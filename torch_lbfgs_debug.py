"""Diagnostic instrumentation for the `RuntimeError: value cannot be
converted to type float without overflow` crash inside torch.optim.LBFGS,
seen in train_adversarial_logreg.tmp.py's warm-started probe refits (see
crash_log/1.txt). Installed via monkeypatch so torch_logreg.py and
probe_backend.py stay untouched -- this is a debugging tool, not a fix.

Two suspected mechanisms, not mutually exclusive:
  1. warm_start=True never resets the LBFGS optimizer's curvature history
     (old_dirs/old_stps/ro/H_diag persist across every .fit() call for the
     whole run). Once the probe is nearly converged and successive
     warm-started fits see near-identical gradients, a curvature pair with
     a tiny-but-positive `ys = y.dot(s)` can blow up `H_diag = ys / y.dot(y)`,
     corrupting the L-BFGS direction `d` until it's astronomically large --
     then even the line search's very first trial (step_size=1) overflows
     when added to the parameters.
  2. The probed activations' per-column std collapses toward the float32
     precision floor (adversarial pressure shrinks class separation, and
     with it the raw variance the StandardScaler divides by), so
     (X - mean) / std amplifies float32 rounding noise into a poorly
     conditioned logistic regression problem.

install() patches:
  - TorchProbePipeline.fit: records per-call health stats (raw activation
    column-std range, probe coef/intercept norms, LBFGS internal state) to
    an in-memory ring buffer, periodically flushed to `log_path`; on any
    exception, dumps the ring buffer plus full internal state to a
    dedicated crash file under `crash_dump_dir` before re-raising.
  - torch.optim.lbfgs.LBFGS._add_grad: prints the exact (step_size, update)
    pair the instant it's about to overflow, i.e. right at the failure
    site, independent of the ring buffer above.
"""

import json
import os
import time
from collections import deque

import torch

from probe_backend import TorchProbePipeline

_RING = deque(maxlen=500)
_CALL_COUNT = 0
_FLUSH_EVERY = 50


def _lbfgs_state(logreg) -> dict:
    """Whatever's in torch.optim.LBFGS's internal state dict for the weight
    param -- {} if the optimizer hasn't run a step yet."""
    opt = logreg._optimizer
    if opt is None:
        return {}
    w = opt.param_groups[0]["params"][0]
    st = opt.state.get(w, {})
    if not st:
        return {}
    old_dirs = st.get("old_dirs") or []
    old_stps = st.get("old_stps") or []
    ro = st.get("ro") or []
    d = st.get("d")
    t = st.get("t")
    prev_flat_grad = st.get("prev_flat_grad")
    H_diag = st.get("H_diag")
    return {
        "n_iter": st.get("n_iter"),
        "func_evals": st.get("func_evals"),
        "H_diag": float(H_diag) if isinstance(H_diag, torch.Tensor) else H_diag,
        "history_len": len(old_dirs),
        "last_ro": float(ro[-1]) if ro else None,
        "last_y_norm": float(old_dirs[-1].norm()) if old_dirs else None,
        "last_s_norm": float(old_stps[-1].norm()) if old_stps else None,
        "d_norm": float(d.norm()) if d is not None else None,
        "t": float(t) if t is not None else None,
        "prev_flat_grad_norm": (
            float(prev_flat_grad.norm()) if prev_flat_grad is not None else None
        ),
        "prev_loss": st.get("prev_loss"),
    }


def _col_stats(X: torch.Tensor) -> dict:
    std = X.std(dim=0, unbiased=False)
    mean_abs = X.abs().mean(dim=0)
    return {
        "min_std": float(std.min()),
        "max_std": float(std.max()),
        "min_mean_abs": float(mean_abs.min()),
        "max_mean_abs": float(mean_abs.max()),
        "any_nan": bool(torch.isnan(X).any()),
        "any_inf": bool(torch.isinf(X).any()),
    }


def _dump(path: str, extra: dict) -> None:
    payload = {"time": time.time(), "recent_history": list(_RING), **extra}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[lbfgs_debug] dumped diagnostics to {path}")


def _install_add_grad_probe() -> None:
    """Print the exact (step_size, update) pair the instant it's about to
    overflow -- catches the failure at its exact source, independent of the
    per-call ring buffer in _install_pipeline_probe."""
    orig_add_grad = torch.optim.LBFGS._add_grad

    def patched_add_grad(self, step_size, update):
        try:
            finite = bool(torch.isfinite(update).all())
            max_abs = float(update.abs().max())
        except Exception:
            finite, max_abs = None, None
        step_f = float(step_size) if isinstance(step_size, (int, float)) else None
        about_to_overflow = (finite is False) or (
            max_abs is not None and step_f is not None and max_abs * abs(step_f) > 1e30
        )
        if about_to_overflow:
            print(
                f"[lbfgs_debug] _add_grad about to overflow: "
                f"step_size={step_size!r} update_max_abs={max_abs} "
                f"update_finite={finite}",
                flush=True,
            )
        return orig_add_grad(self, step_size, update)

    torch.optim.LBFGS._add_grad = patched_add_grad


def _install_pipeline_probe(log_path: str, crash_dump_dir: str) -> None:
    orig_fit = TorchProbePipeline.fit

    def instrumented_fit(self, X, y):
        global _CALL_COUNT
        _CALL_COUNT += 1

        pre_raw_X = _col_stats(X)
        logreg = self._logreg
        w_prev = float(logreg.coef_.norm()) if logreg.coef_ is not None else None
        b_prev = float(logreg.intercept_) if logreg.intercept_ is not None else None
        pre_lbfgs = _lbfgs_state(logreg)

        entry = {
            "call": _CALL_COUNT,
            "pre_raw_X": pre_raw_X,
            "w_prev_norm": w_prev,
            "b_prev": b_prev,
            "pre_lbfgs": pre_lbfgs,
        }

        try:
            result = orig_fit(self, X, y)
        except Exception as e:
            entry["exception"] = repr(e)
            _RING.append(entry)
            dump_path = f"{crash_dump_dir}/lbfgs_debug_crash_call{_CALL_COUNT}.json"
            _dump(dump_path, extra={"reason": repr(e), "call_count": _CALL_COUNT})
            raise

        entry["w_new_norm"] = float(logreg.coef_.norm())
        entry["b_new"] = float(logreg.intercept_)
        entry["post_lbfgs"] = _lbfgs_state(logreg)
        _RING.append(entry)

        if _CALL_COUNT % _FLUSH_EVERY == 0:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

        return result

    TorchProbePipeline.fit = instrumented_fit


def install(
    log_path: str = "crash_log/lbfgs_debug.jsonl",
    crash_dump_dir: str = "crash_log",
) -> None:
    """Call once, before the training loop starts."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    os.makedirs(crash_dump_dir, exist_ok=True)
    _install_add_grad_probe()
    _install_pipeline_probe(log_path, crash_dump_dir)
    print(
        f"[lbfgs_debug] instrumentation installed (health log flushed every "
        f"{_FLUSH_EVERY} probe fits -> {log_path}; crash dump -> {crash_dump_dir}/)"
    )
