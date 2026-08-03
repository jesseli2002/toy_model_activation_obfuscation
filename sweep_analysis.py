"""Aggregate runs/sweep3_lam*_tr* into a stacked-fraction-vs-lambda chart:
for each lambda, what fraction of its trials failed the task (loss), and
among those that succeeded, how well they hid from the layer-PROBE_LAYER
linear probe (AUROC), at one or more AUROC thresholds.

x-axis is lam / (1 - lam) -- the probe-to-task loss weight ratio (see
`(1 - lam_eff) * l_task + lam_eff * l_probe` in
train_adversarial_logreg._compute_losses) -- rather than lam itself, since
it's unbounded above (lam is capped at 1) and matches what the adversarial
penalty actually trades off against. Plotted symlog so lam=0 (ratio 0) has a
finite position while still spreading out the many-orders-of-magnitude tail.

LOSS_THRESHOLD / AUROC_THRESHOLDS are provisional -- a separate exercise is
picking the final values -- so they're constants here, not a CLI, alongside
everything else that may still change. Prints a summary table; plt.show()
only, nothing written to disk.
"""

RUN_GLOB = "sweep3_lam*_tr*"
EXCLUDE_LAMBDAS: set[float] = set()
CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json
LOSS_LOWPASS_WINDOW = 2000  # matches sweep_report.py's smoothing window
EVAL_NOISE_MULT = 1.0  # see adversarial_report.py's --eval-noise-mult

LOSS_THRESHOLD = 0.05  # task "succeeded" iff final loss below this -- placeholder
AUROC_THRESHOLDS = [
    0.6,
    0.75,
]  # ascending; splits "succeeded" runs into len+1 hiding bins -- placeholder

SYMLOG_LINTHRESH = 0.002  # linear-region half-width (in ratio units) around lam=0

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import sample_fixed_c
from paths import ckpt_dir, log_dir
from probe_lib import capture_layers, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TAG_RE = re.compile(r"sweep3_lam([0-9.]+)_tr(\d+)$")

# Sequential blue ramp (references/palette.md), lightest -> darkest, 100..700.
SEQ_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
FAILED_COLOR = "#eb6834"  # categorical slot 2 (orange) -- distinct "problem" hue


def _discover_tags_by_lambda() -> dict[float, list[str]]:
    by_lambda: dict[float, list[str]] = {}
    for path in sorted(glob.glob(os.path.join("runs", RUN_GLOB))):
        tag = os.path.basename(path)
        m = TAG_RE.match(tag)
        if not m:
            continue
        lam = float(m.group(1))
        if lam in EXCLUDE_LAMBDAS:
            continue
        by_lambda.setdefault(lam, []).append(tag)
    return by_lambda


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


def _running_mean(iters: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """Causal low-pass: mean value among logged points within the trailing
    `window` iterations (not `window` logged points -- --log-interval
    defaults to 100 and can vary run to run, so this has to key off `iters`,
    not array index). Matches sweep_report.py's smoothing."""
    out = np.empty_like(values, dtype=float)
    lo = 0
    for i in range(len(iters)):
        cutoff = iters[i] - window + 1
        while iters[lo] < cutoff:
            lo += 1
        out[i] = values[lo : i + 1].mean()
    return out


def _final_loss(tag: str) -> float:
    iters, losses = _load_history(tag)
    return float(_running_mean(iters, losses, LOSS_LOWPASS_WINDOW)[-1])


def _probe_auroc(tag: str, eval_n: int = 4096, seed: int = 0) -> float | None:
    """Probe AUROC at the run's CKPT checkpoint, scored with the probe
    (probe_w/probe_b/probe_layers) already fit and stored in the checkpoint --
    same computation as sweep_report.py / sweep_threshold_report.py."""
    model, ck = load_model(tag, CKPT, DEVICE)
    if "probe_w" not in ck:
        return None
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    noise_gen = torch.Generator(device=DEVICE).manual_seed(seed)
    num_x = model.num_x
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


def _classify(loss: float, auroc: float, thresholds: list[float]) -> int:
    """Bucket index, 0 = failed task, 1 = succeeded but not hidden (auroc
    above the highest threshold), ..., len(thresholds) + 1 = succeeded and
    most hidden (auroc below the lowest threshold)."""
    if loss >= LOSS_THRESHOLD:
        return 0
    for i, t in enumerate(reversed(thresholds)):
        if auroc >= t:
            return i + 1
    return len(thresholds) + 1


def _band_labels(thresholds: list[float]) -> list[str]:
    labels = [f"failed task (loss >= {LOSS_THRESHOLD:g})"]
    desc = reversed(thresholds)
    prev = None
    for t in desc:
        if prev is None:
            labels.append(f"not hidden (auroc >= {t:g})")
        else:
            labels.append(f"partially hidden ({t:g} <= auroc < {prev:g})")
        prev = t
    labels.append(f"hidden (auroc < {prev:g})")
    return labels


def _band_colors(thresholds: list[float]) -> list[str]:
    n_hiding_bins = len(thresholds) + 1
    idxs = np.linspace(0, len(SEQ_RAMP) - 1, n_hiding_bins).astype(int)
    return [FAILED_COLOR] + [SEQ_RAMP[i] for i in idxs]


def main():
    by_lambda = _discover_tags_by_lambda()
    thresholds = sorted(AUROC_THRESHOLDS)
    n_bands = len(thresholds) + 2

    ratios: list[float] = []
    fractions: list[np.ndarray] = []  # one length-n_bands array per lambda

    print(f"{'lambda':>10s} {'ratio':>10s} {'n_ok':>5s} {'n_total':>7s}")
    for lam in sorted(by_lambda):
        tags = by_lambda[lam]
        counts = np.zeros(n_bands)
        n_ok = 0
        for tag in tags:
            try:
                loss = _final_loss(tag)
                auroc = _probe_auroc(tag)
            except FileNotFoundError:
                print(f"  {tag}: no history/checkpoint yet, skipped")
                continue
            if auroc is None:
                print(f"  {tag}: no probe in checkpoint, skipped")
                continue
            counts[_classify(loss, auroc, thresholds)] += 1
            n_ok += 1
        if n_ok == 0:
            print(f"  {lam}: no usable runs, skipped")
            continue
        ratio = lam / (1 - lam)
        ratios.append(ratio)
        fractions.append(counts / n_ok)
        print(f"{lam:10g} {ratio:10.4g} {n_ok:5d} {len(tags):7d}")

    order = np.argsort(ratios)
    x = np.array(ratios)[order]
    frac_matrix = np.array(fractions)[order]  # (n_lambda, n_bands)

    labels = _band_labels(thresholds)
    colors = _band_colors(thresholds)

    fig, ax = plt.subplots(figsize=(8, 5))
    # stackplot stacks bottom-to-top in argument order; we want "failed" on
    # top and "most hidden" at the bottom, so feed bands in reverse.
    ax.stackplot(
        x,
        *frac_matrix.T[::-1],
        labels=labels[::-1],
        colors=colors[::-1],
        alpha=0.9,
    )
    ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH)
    ax.axvline(0, ls="--", lw=1, color="#52514e", label="lam=0")
    ax.set_xlabel("lam / (1 - lam)  (probe-loss weight / task-loss weight)")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1)
    ax.set_title(f"task success / probe hiding vs. adversarial weight ({RUN_GLOB})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
