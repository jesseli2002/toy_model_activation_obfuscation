"""Pins the vendored StableAdamW against the upstream `optimi` implementation
it was derived from, so that `d` is the only behavioural difference."""

import io

import optimi
import pytest
import torch

from stableadamw import StableAdamW


def _make_params(seed: int = 0) -> list[torch.Tensor]:
    """A few differently-shaped tensors -- clipping is per-tensor, so the RMS
    (and hence the clipped lr) must differ between them for the comparison to
    have any teeth."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(8, 4), (4,), (3, 3, 2)]
    return [torch.randn(*s, generator=g, requires_grad=True) for s in shapes]


def _grads(params: list[torch.Tensor], step: int, scale: float = 1.0):
    """Deterministic per-step gradients, with one tensor driven far harder than
    the others so at least one of them exceeds the clipping threshold."""
    g = torch.Generator().manual_seed(1000 + step)
    for i, p in enumerate(params):
        p.grad = torch.randn(p.shape, generator=g) * scale * (10.0 if i == 0 else 1.0)


def _run(optimizer_factory, *, steps: int = 10, scale: float = 1.0, seed: int = 0):
    params = _make_params(seed)
    opt = optimizer_factory(params)
    for step in range(steps):
        _grads(params, step, scale)
        opt.step()
    return params


@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_matches_optimi_at_d_1(weight_decay):
    ours = _run(
        lambda p: StableAdamW(
            p, lr=3e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=weight_decay, d=1.0
        )
    )
    theirs = _run(
        lambda p: optimi.StableAdamW(
            p,
            lr=3e-3,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=weight_decay,
            triton=False,
            kahan_sum=False,
            foreach=False,
        )
    )
    for a, b in zip(ours, theirs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_smaller_d_clips_harder():
    """With gradients large enough to push RMS above the threshold, lowering `d`
    must shrink the distance travelled -- the whole point of exposing it."""
    start = _make_params()
    dists = []
    for d in [2.0, 1.0, 0.25]:
        end = _run(
            lambda p, d=d: StableAdamW(p, lr=3e-3, weight_decay=0.0, d=d), scale=100.0
        )
        with torch.no_grad():
            dists.append(sum((e - s).norm().item() for e, s in zip(end, start)))
    assert dists[0] > dists[1] > dists[2]


def test_large_d_is_inert():
    """A threshold far above any observed RMS should never trigger, leaving the
    update identical to an unclipped one."""
    loose = _run(lambda p: StableAdamW(p, lr=3e-3, weight_decay=0.0, d=1e9), scale=0.01)
    looser = _run(
        lambda p: StableAdamW(p, lr=3e-3, weight_decay=0.0, d=1e12), scale=0.01
    )
    for a, b in zip(loose, looser):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_round_trips_its_own_state_dict():
    """`--resume` reloads optimizer state into a freshly built optimizer, so a
    save/load cycle must be a no-op on the resulting step."""
    params = _make_params()
    opt = StableAdamW(params, lr=3e-3, d=0.5)
    _grads(params, 0)
    opt.step()

    # Through a save/load cycle, as --resume does: loading in-process would
    # alias the source optimizer's state tensors rather than copying them.
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)

    resumed_params = [p.detach().clone().requires_grad_(True) for p in params]
    resumed = StableAdamW(resumed_params, lr=3e-3, d=0.5)
    resumed.load_state_dict(torch.load(buf, weights_only=True))

    for step in (1, 2):
        _grads(params, step)
        opt.step()
        _grads(resumed_params, step)
        resumed.step()
    for a, b in zip(params, resumed_params):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rejects_nonpositive_d():
    params = _make_params()
    with pytest.raises(ValueError, match="d="):
        StableAdamW(params, lr=3e-3, d=0.0)
