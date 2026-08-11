"""Reading and smoothing a run's training history from logs/history.jsonl.

For loss trajectories over training. Contrast sweep_lib.metrics, which
ignores history entirely and recomputes metrics from a checkpoint.
"""

import functools
import json
import os

import numpy as np
from jaxtyping import Float

from paths import log_dir

LOSS_FIELDS = {"total": "loss", "task": "l_task", "probe": "l_probe"}


def load_history(
    tag: str, loss_type: str = "task"
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    """(iters, losses) for a run, where `loss_type` selects a LOSS_FIELDS key.

    Skips the trailing 'final' summary record, which repeats the last iter
    with its loss fields nulled out.
    """
    if loss_type not in LOSS_FIELDS:
        raise ValueError(f"{loss_type=} not one of {sorted(LOSS_FIELDS)}")
    field = LOSS_FIELDS[loss_type]

    iters, losses = [], []
    with open(os.path.join(log_dir(tag), "history.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("final"):
                continue
            iters.append(rec["iter"])
            losses.append(rec[field])
    return np.array(iters), np.array(losses)


def causal_lowpass(
    iters: Float[np.ndarray, " n"],
    values: Float[np.ndarray, " n"],
    window: int,
    reduce=np.mean,
) -> Float[np.ndarray, " n"]:
    """Reduce `values` over the logged points within the trailing `window`
    *iterations* of each point.

    Keys off `iters` rather than array index because --log-interval varies
    between runs, so a fixed number of logged points is a different amount of
    training in each.
    """
    out = np.empty_like(values)
    lo = 0
    for i in range(len(iters)):
        cutoff = iters[i] - window + 1
        while iters[lo] < cutoff:
            lo += 1
        out[i] = reduce(values[lo : i + 1])
    return out


def median_trajectory(
    trials: list[tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]],
) -> tuple[Float[np.ndarray, " m"], Float[np.ndarray, " m"]]:
    """Per-iteration median across trials' (iters, values).

    Trials log at different iters and run to different lengths, so this
    interpolates each onto the union of their iteration grids, clipped to the
    range every trial covers.
    """
    lo = max(iters[0] for iters, _ in trials)
    hi = min(iters[-1] for iters, _ in trials)
    grid = functools.reduce(np.union1d, [iters for iters, _ in trials])
    grid = grid[(grid >= lo) & (grid <= hi)]
    stacked = np.stack([np.interp(grid, iters, values) for iters, values in trials])
    return grid, np.median(stacked, axis=0)
