import torch

from config import ResidualMLPConfig
from model import ResidualMLP


def _make_model(num_blocks=4):
    cfg = ResidualMLPConfig(num_x=3, d_model=8, d_mlp=16, num_blocks=num_blocks)
    return ResidualMLP(cfg)


def test_noise_none_matches_omitted():
    model = _make_model()
    x = torch.randn(5, model.d_in)
    assert torch.equal(model.forward(x), model.forward(x, noise=None))


def test_generate_noise_shape_and_scale():
    model = _make_model(num_blocks=4)
    gen = torch.Generator().manual_seed(0)
    noise = model.generate_noise(1000, noise_std=2.0, generator=gen)
    assert noise.shape == (3, 1000, model.d_model)
    assert abs(noise.std().item() - 2.0) < 0.1


def test_tensor_noise_is_replayed_bit_identically():
    """The property the noise-blob refactor buys over `gen`-state replay:
    passing the same blob to two forward calls gives identical output."""
    model = _make_model()
    x = torch.randn(5, model.d_in)
    gen = torch.Generator().manual_seed(0)
    noise = model.generate_noise(x.shape[0], noise_std=0.5, generator=gen)
    out1 = model.forward(x, noise=noise)
    out2 = model.forward(x, noise=noise)
    assert torch.equal(out1, out2)


def test_float_noise_draws_fresh_each_call():
    model = _make_model()
    x = torch.randn(5, model.d_in)
    gen = torch.Generator().manual_seed(0)
    out1 = model.forward(x, noise=0.5, generator=gen)
    out2 = model.forward(x, noise=0.5, generator=gen)
    assert not torch.equal(out1, out2)


def test_num_blocks_one_has_no_injection_points():
    model = _make_model(num_blocks=1)
    noise = model.generate_noise(4, noise_std=1.0)
    assert noise.shape == (0, 4, model.d_model)
    x = torch.randn(4, model.d_in)
    assert torch.equal(model.forward(x, noise=noise), model.forward(x, noise=None))
