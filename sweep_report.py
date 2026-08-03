"""Aggregate several same-settings, different-seed training runs into one
sweep report: superimposed loss curves and last-iteration probe AUROC.

Settings live in the constants below rather than a CLI -- this script's
shape is still changing, so argparse would just be churn for now. Prints to
console / plt.show() only; nothing is written to disk.
"""

RUN_TAGS = [f"sweep3_lam0.1_tr{i}" for i in range(10)]
# RUN_TAGS = [f"sweep3_lam0.05_tr{i}" for i in range(10)]
# RUN_TAGS = [f"sweep3_lam0.01_tr{i}" for i in range(10)]
# RUN_TAGS = [f"sweep3_lam0.003_tr{i}" for i in range(10)]
CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
LOSS_LOWPASS_WINDOW = 2000  # running min of loss over the past this-many iters
EVAL_NOISE_MULT = 1.0  # see adversarial_report.py's --eval-noise-mult
N_TRAIN = 20_000  # per class, fit size for the refit probe
PROBE_BACKEND = "auto"  # see adversarial_report.py's --probe-backend

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from paths import log_dir
from probe_backend import resolve_probe_backend
from probe_lib import RefitProbeScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def _load_history(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """(iters, loss) from a run's logs/history.jsonl. Skips the trailing
    'final' summary record, which repeats the last iter with loss fields
    nulled out."""
    iters, losses = [], []
    with open(os.path.join(log_dir(tag), "history.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("final"):
                continue
            iters.append(rec["iter"])
            losses.append(rec["loss"])
    return np.array(iters), np.array(losses)


def _running_min(iters: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """Causal low-pass: minimum value among logged points within the trailing
    `window` iterations (not `window` logged points -- --log-interval
    defaults to 100 and can vary run to run, so this has to key off `iters`,
    not array index)."""
    out = np.empty_like(values)
    lo = 0
    for i in range(len(iters)):
        cutoff = iters[i] - window + 1
        while iters[lo] < cutoff:
            lo += 1
        # out[i] = values[lo : i + 1].min()
        out[i] = values[lo : i + 1].mean()
    return out


def main():
    fig, (ax_loss, ax_auroc) = plt.subplots(1, 2, figsize=(13, 5))

    for tag in RUN_TAGS:
        iters, losses = _load_history(tag)
        filtered = _running_min(iters, losses, LOSS_LOWPASS_WINDOW)
        ax_loss.plot(iters, filtered, label=tag, alpha=0.8)
    ax_loss.set_xlabel("iter")
    ax_loss.set_ylabel(f"loss (min over past {LOSS_LOWPASS_WINDOW} iters)")
    ax_loss.set_yscale("log")
    ax_loss.set_title("training loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    print(f"refit probe AUROC at {CKPT} checkpoint:")
    scorer = RefitProbeScorer(
        CKPT,
        DEVICE,
        n_train=N_TRAIN,
        probe_backend_name=resolve_probe_backend(PROBE_BACKEND, DEVICE),
        eval_noise_mult=EVAL_NOISE_MULT,
        seed=SEED,
    )
    aurocs = {tag: scorer.auroc(tag) for tag in RUN_TAGS}
    for tag in RUN_TAGS:
        if aurocs[tag] is None:
            print(f"  {tag}: n/a (no probe in checkpoint)")
        else:
            print(f"  {tag}: {aurocs[tag]:.4f}")

    tags_with_auroc = [t for t in RUN_TAGS if aurocs[t] is not None]
    ax_auroc.bar(tags_with_auroc, [aurocs[t] for t in tags_with_auroc])
    ax_auroc.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax_auroc.set_ylim(0.45, 1.02)
    ax_auroc.set_ylabel("AUROC")
    ax_auroc.set_title(f"probe AUROC ({CKPT} checkpoint)")
    ax_auroc.tick_params(axis="x", rotation=45)
    ax_auroc.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
