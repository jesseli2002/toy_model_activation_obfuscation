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

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import sample_fixed_c
from model import ResidualMLP
from paths import ckpt_dir, log_dir
from probe_lib import capture_layers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def _last_iter_auroc(tag: str, eval_n: int = 4096, seed: int = 0) -> float | None:
    """Probe AUROC at the run's CKPT checkpoint, scored with the probe
    (probe_w/probe_b/probe_layers) already fit and stored in the checkpoint --
    same computation as adversarial_report.py's _auroc_snapshots, but for one
    checkpoint instead of a sweep across training."""
    path = os.path.join(ckpt_dir(tag), f"{CKPT}.pt")
    model, ck = ResidualMLP.load(path, map_location=DEVICE)
    if "probe_w" not in ck:
        return None
    model = model.to(DEVICE).eval()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    noise_gen = torch.Generator(device=DEVICE).manual_seed(seed)
    num_x = model.config.num_x
    xf_lo, _ = sample_fixed_c(eval_n, num_x, 1.0, generator=gen, device=DEVICE)
    xf_hi, _ = sample_fixed_c(eval_n, num_x, 2.0, generator=gen, device=DEVICE)
    eval_x = torch.cat([xf_lo, xf_hi], dim=0)
    eval_label = np.concatenate([np.zeros(eval_n), np.ones(eval_n)])
    w = ck["probe_w"].to(DEVICE)
    b = ck["probe_b"].to(DEVICE)
    noise_std = ck["adv_config"]["resid_noise_std"] * EVAL_NOISE_MULT
    with torch.no_grad():
        feats = capture_layers(
            model, eval_x, ck["probe_layers"], noise_std=noise_std, generator=noise_gen
        )
        s = (feats @ w + b).cpu().numpy()
    return roc_auc_score(eval_label, s)


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

    print(f"probe AUROC at {CKPT} checkpoint:")
    aurocs = {}
    for tag in RUN_TAGS:
        aurocs[tag] = _last_iter_auroc(tag)
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
