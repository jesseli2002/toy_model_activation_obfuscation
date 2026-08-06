"""Analysis of an adversarially-trained checkpoint's task fidelity and probe strength.

Loads an adversarial checkpoint (from train_adversarial_logreg.py) and
produces a report covering:

  1. Task fidelity           — the PRICE of hiding (expected near-zero).
  2. Probe-strength gap at the probed points, per hidden layer.
  3. Held-out c recovery — binary probes at held-out c pairs chosen
     asymmetric about the training midpoint so an affine cancellation
     can't masquerade as a genuinely weakened probe.

The held-out-pairs table is the most expensive part of the script and is
only printed/written with --detailed.

Two optional deep-dive analyses, each opt-in since they cost extra compute:
  --detailed        more detailed analysis.
  --steer L1,L2,...  causal test: inject the DoM- and logreg-discovered c=1->c=2
                     directions at each given layer and see whether the model's
                     output actually moves toward the c=2 target -- a probe that
                     merely correlates with c but isn't causally used would fail
                     this even at high probe accuracy.
"""

import argparse
import dataclasses
import glob
import json
import os

import config


def _parse_pairs(s: str) -> list[tuple[float, float]]:
    pairs = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        lo, hi = tok.split("-")
        pairs.append((float(lo), float(hi)))
    return pairs


def _parse_int_list(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(
        description="Adversarial analysis: task fidelity vs. probe strength.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="last", choices=["best", "last"])
    p.add_argument(
        "--held-out-pairs",
        type=_parse_pairs,
        default="1.0-1.5,1.0-1.75,1.25-2.0",
        help="binary held-out c pairs, comma-separated 'lo-hi'. Kept ASYMMETRIC "
        "about 1.5 on purpose (see module docstring).",
    )
    p.add_argument("--n-train", type=int, default=20_000, help="per class/set")
    p.add_argument("--n-test", type=int, default=50_000, help="per class/set")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument(
        "--probe-backend",
        choices=config.PROBE_BACKEND_CHOICES,
        default="auto",
        help="'auto': torch (GPU-resident) probe iff CUDA is available, else "
        "sklearn. 'sklearn'/'torch' force a backend regardless of device.",
    )
    p.add_argument("--show", action="store_true")
    p.add_argument(
        "--detailed",
        action="store_true",
        help="more detailed analysis.",
    )
    p.add_argument(
        "--steer",
        type=_parse_int_list,
        default=None,
        help="comma-separated hidden-layer indices to run the causal steering "
        "test at (e.g. '3,6,9'). For each layer, injects the DoM- and "
        "logreg-discovered c=1->c=2 directions (magnitude-matched) into c=1 "
        "inputs and plots the resulting y(x). Must be a subset of the hidden "
        "layers already probed at c in {1,2} (no extra forward passes needed).",
    )
    p.add_argument(
        "--steer-scale",
        type=float,
        default=1.0,
        help="multiple of the full c=1->c=2 shift magnitude to inject (1.0 = full).",
    )
    p.add_argument(
        "--eval-noise-mult",
        type=float,
        default=1.0,
        help="multiplier on the checkpoint's own adv_config.resid_noise_std, "
        "injected into the residual stream only when EVALUATING probes (never "
        "when fitting them, since a probe fit is a large-n limit that noise "
        "doesn't move) -- replicates the noisy environment the model itself "
        "saw at train time. 0 disables.",
    )
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.collections import PolyCollection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

from analytic import reference_task_losses
from data import sample_batch
from paths import ckpt_dir, log_dir
from paths import plot_dir as get_plot_dir
from data import eval_max_err
from probe_backend import resolve_probe_backend
from probe_lib import (
    LinearBoundary,
    binary_probe_metrics_all_layers,
    forward_steered,
    load_model,
    load_model_path,
    make_binary_eval_set,
    resolve_adv_config,
    save_plot,
    stored_probe_auroc,
)
from probe_lib import plot_probe as plot_probe_separation
from probe_lib import plot_probe_pca as plot_probe_pca_separation


@torch.no_grad()
def plot_learned_curves(
    model,
    tag,
    plot_dir,
    c_values=(1.0, 1.333, 1.667, 2.0),
    show=False,
):
    """Plot learned y(x) per coordinate at fixed c, for an already-loaded model."""
    num_x = model.num_x
    device = next(model.parameters()).device
    xs = torch.linspace(-3, 3, 400, device=device)
    fig, axes = plt.subplots(
        1, len(c_values), figsize=(4 * len(c_values), 4), sharey=True
    )
    if len(c_values) == 1:
        axes = [axes]
    for ax, c in zip(axes, c_values):
        # Build inputs: sweep x on every coordinate simultaneously is not valid
        # (coords are independent), so sweep coordinate 0 and hold others at 0.
        for j in range(num_x):
            x = torch.zeros(len(xs), num_x, device=device)
            x[:, j] = xs
            x_full = torch.cat([x, torch.full((len(xs), 1), c, device=device)], dim=1)
            y = model.task_output(x_full)[:, j]
            ax.plot(xs.cpu().numpy(), y.cpu().numpy(), alpha=0.5, zorder=5)
        ax.plot(
            xs.cpu().numpy(),
            torch.clamp(xs, -c, c).cpu().numpy(),
            "k--",
            lw=1,
            label="target sat",
            zorder=2,
        )
        ax.set_title(f"c = {c:.3f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("y")
    fig.suptitle(f"learned y(x) per coordinate, fixed c ({tag}); {num_x} lines/panel")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_curves.png", close=not show)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _auroc_snapshots(
    tag, device, eval_noise_mult, n_snapshots=1000, eval_n=4096, seed=0
):
    """Probe AUROC over the course of training, computed here (not logged
    during training) from the numbered checkpoint snapshots
    train_adversarial_logreg.py already writes every --ckpt-interval iters --
    each carries the probe's affine (probe_w/probe_b/probe_layers) alongside
    the model weights.

    Snapshots are approximately-uniformly strided down to ~n_snapshots (a
    simple stride, not an exact pick) since loading + a forward pass per
    snapshot is the expensive part. All snapshots share one eval set, so the
    trace is comparable point to point (see probe_lib.BinaryEvalSet), with
    `eval_noise_mult` * each snapshot's own adv_config.resid_noise_std
    injected into the eval forward pass."""
    paths = sorted(
        glob.glob(os.path.join(ckpt_dir(tag), "iter_*.pt")),
        key=lambda p: int(os.path.basename(p)[len("iter_") : -len(".pt")]),
    )
    stride = max(1, len(paths) // n_snapshots)
    paths = paths[::stride]

    g = torch.Generator(device=device).manual_seed(seed)
    eval_set = None
    out = []
    for path in paths:
        model, ck = load_model_path(path, device)
        if eval_set is None:
            eval_set = make_binary_eval_set(
                model.config.num_x, eval_n, generator=g, device=device
            )
        auroc = stored_probe_auroc(
            model, ck, eval_set, eval_noise_mult=eval_noise_mult, generator=g
        )
        if auroc is None:
            continue  # not a logreg-probe checkpoint (some have no adv_config at all)
        out.append((ck["iter"], auroc))
    return out


def _plot_training_traces(
    tag, history, plot_dir, device, class_threshold, eval_noise_mult, show=False
):
    pts = [h for h in history if h.get("l_task") is not None]
    if not pts:
        return
    its = [h["iter"] for h in pts]
    fig, (ax_err, ax_auroc, ax_loss) = plt.subplots(1, 3, figsize=(16, 4.2))

    ax_err.semilogy(its, [h["max_err"] for h in pts], color="crimson")
    ax_err.set_title("task fidelity (price of hiding)")
    ax_err.set_xlabel("iter")
    ax_err.set_ylabel("max abs error")
    ax_err.grid(True, alpha=0.3)

    snapshots = _auroc_snapshots(tag, device, eval_noise_mult)
    if snapshots:
        auroc_its = [it for it, _ in snapshots]
        auroc_ys = [y for _, y in snapshots]
        ax_auroc.plot(auroc_its, auroc_ys, color="tab:purple", label="probe AUROC")
        ax_auroc.axhline(0.5, color="k", ls="--", lw=1, label="chance")
        ax_auroc.set_ylim(0.45, 1.02)
        ax_auroc.legend(fontsize=8)
    else:
        ax_auroc.text(
            0.5,
            0.5,
            "no checkpoint snapshots (iter_*.pt) found",
            ha="center",
            va="center",
            transform=ax_auroc.transAxes,
        )
    if class_threshold is not None:
        ax_auroc.set_title(f"probe AUROC for c > {class_threshold:g}")
    else:
        ax_auroc.set_title("probe AUROC (adversary strength)")
    ax_auroc.set_xlabel("iter")
    ax_auroc.set_ylabel("AUROC")
    ax_auroc.grid(True, alpha=0.3)

    ax_loss.semilogy(its, [h["l_task"] for h in pts], label="L_task")
    # ax_loss.semilogy(its, [max(h["loss"], 1e-30) for h in pts], label="L_total")
    ref_losses = reference_task_losses(
        config.X_LOW, config.X_HIGH, config.C_LOW, config.C_HIGH
    )
    for key, style in [
        ("do_nothing", ":"),
        ("linreg", "--"),
        ("clamp", "-."),
    ]:
        ax_loss.axhline(
            ref_losses[key], color="gray", ls=style, lw=1, label=key, zorder=0
        )
    ax_loss.set_title("loss terms")
    ax_loss.set_xlabel("iter")
    # ax_loss.set_ylim(bottom=3e-3)
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)

    fig.suptitle(f"adversarial training traces ({tag})")
    fig.tight_layout()
    save_plot(fig, plot_dir, f"{tag}_training.png", close=not show)


def _plot_probe_gap(tag, hidden_layers, gap, plot_dir):
    x = np.arange(len(hidden_layers))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for off, key, lbl in [
        (-0.25, "dom", "DoM"),
        (0.0, "logreg", "logreg"),
        (0.25, "lda", "LDA"),
    ]:
        ax.bar(x + off, [gap[l][key] for l in hidden_layers], 0.25, label=lbl)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in hidden_layers])
    ax.set_ylabel("accuracy at c in {1,2}")
    ax.set_title(f"probe-strength gap ({tag})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_plot(fig, plot_dir, f"{tag}_probe_gap.png")


def _plot_heldout_gap(
    tag, hidden_layers, held_out_pairs, gap, heldout, plot_dir, metric="logreg"
):
    """Bar chart analogous to _plot_probe_gap, but grouped by c-pair instead of
    by probe type -- one metric (logreg, the pass/fail gate used everywhere
    else in this codebase) compared across the baseline {1,2} pair and every
    --held-out-pairs entry, per layer."""
    groups = [(1.0, 2.0, gap)] + [
        (lo, hi, {lyr: heldout[(lo, hi, lyr)] for lyr in hidden_layers})
        for lo, hi in held_out_pairs
    ]
    n = len(groups)
    x = np.arange(len(hidden_layers))
    width = 0.8 / n
    offsets = (np.arange(n) - (n - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(hidden_layers)), 4.2))
    for off, (lo, hi, m) in zip(offsets, groups):
        ax.bar(
            x + off,
            [m[l][metric] for l in hidden_layers],
            width,
            label=f"c={lo:g}/{hi:g}",
        )
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in hidden_layers])
    ax.set_ylabel(f"{metric} accuracy")
    ax.set_title(f"held-out c-pair probe gap ({tag}, {metric})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_plot(fig, plot_dir, f"{tag}_heldout_gap.png")


def _plot_layer_distributions(tag, c_lo, c_hi, layers, plot_inputs, plot_dir):
    """Per-layer, per-class signed-distance-to-boundary distributions (raw
    data units, boundary at 0, shared y-axis across layers -- so a collapse
    in spread at a penalized layer is visually comparable to neighboring
    layers), plus a companion panel tracking the mean-gap/pooled-std collapse
    numerically."""
    lo_label, hi_label = f"c={c_lo:g}", f"c={c_hi:g}"
    rows = []
    for lyr in layers:
        for dist, label in (
            (plot_inputs[lyr]["dist_lo"], lo_label),
            (plot_inputs[lyr]["dist_hi"], hi_label),
        ):
            rows.extend(
                {"layer": f"L{lyr}", "distance": d, "class": label} for d in dist
            )
    df = pd.DataFrame(rows)

    layer_gap = [
        float(plot_inputs[l]["dist_hi"].mean() - plot_inputs[l]["dist_lo"].mean())
        for l in layers
    ]
    pooled_std = [
        float(
            np.sqrt(
                (
                    plot_inputs[l]["dist_lo"].std() ** 2
                    + plot_inputs[l]["dist_hi"].std() ** 2
                )
                / 2
            )
        )
        for l in layers
    ]

    fig_width = round(max(7, 1.4 * len(layers)) * 2 / 3)
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # density_norm="width" caps every violin at the same max width regardless
    # of how peaked/spread its KDE is -- otherwise a near-degenerate layer
    # (tiny std) balloons to full width in a sliver of y-range (reads as a
    # flat horizontal bar) while a widely-spread layer's per-point density is
    # low and shrinks to a near-invisible vertical thread, under the default
    # "area" normalization (equal probability mass -> equal area).
    sns.violinplot(
        data=df,
        x="layer",
        y="distance",
        hue="class",
        split=True,
        density_norm="width",
        inner=None,
        linewidth=1.5,
        ax=ax_top,
    )
    # Force each violin's outline to match its own fill color: seaborn's
    # linecolor="auto" default renders dark/near-black, which swallows the
    # class color-coding entirely once a violin collapses to a thin sliver
    # (the fill area vanishes and only the outline remains visible).
    for artist in ax_top.collections:
        if isinstance(artist, PolyCollection):
            artist.set_edgecolor(artist.get_facecolor())
    ax_top.axhline(0.0, color="k", ls="--", lw=1, label="boundary")
    ax_top.set_title(
        f"probe signed distance to boundary, per layer ({tag}, {lo_label} vs {hi_label})"
    )
    ax_top.set_xlabel("")
    ax_top.set_ylabel("signed distance (data units)")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # Twin-x: mean gap and pooled std differ by ~an order of magnitude, so a
    # shared axis makes pooled std unreadable -- each line gets its own axis,
    # color-matched to its label so the split reads unambiguously.
    x = np.arange(len(layers))
    ax_bot2 = ax_bot.twinx()
    (line_gap,) = ax_bot.plot(
        x, layer_gap, marker="o", color="tab:blue", label="mean gap"
    )
    (line_std,) = ax_bot2.plot(
        x, pooled_std, marker="o", color="tab:orange", label="pooled std"
    )
    ax_bot.axhline(0.0, color="k", lw=0.8)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"L{l}" for l in layers])
    ax_bot.set_xlabel("layer")
    ax_bot.set_ylabel("mean gap (data units)", color="tab:blue")
    ax_bot.tick_params(axis="y", labelcolor="tab:blue")
    ax_bot2.set_ylabel("pooled std (data units)", color="tab:orange")
    ax_bot2.tick_params(axis="y", labelcolor="tab:orange")
    ax_bot.legend(
        [line_gap, line_std], [line_gap.get_label(), line_std.get_label()], fontsize=8
    )
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    save_plot(fig, plot_dir, f"{tag}_c{c_lo:g}-{c_hi:g}_layer_dist.png")


def _steer_vectors(w_dom, w_probe, scale):
    """DoM steer vector = scale * (mu_hi - mu_lo), raw units. Logreg steer
    vector = scale * ||w_dom|| * unit(w_probe) -- normalized to the SAME
    causal magnitude as the DoM shift, so a difference in downstream effect
    reflects the direction the probe found, not an arbitrary scale (w_probe's
    own norm is set by the regularization strength, not a meaningful shift
    size)."""
    w_dom_vec = scale * w_dom
    w_probe_unit = w_probe / np.linalg.norm(w_probe)
    w_logreg_vec = scale * np.linalg.norm(w_dom) * w_probe_unit
    return w_dom_vec, w_logreg_vec


@torch.no_grad()
def _plot_steer_comparison(
    tag, steer_layer, num_x, model, w_dom, w_probe, steer_scale, plot_dir, device
):
    """Causal steering test, DoM direction vs logreg direction side by side.
    Only the steered curves are shown (no unsteered/reference panels) --
    each panel already carries both targets
    (sat(x,1), sat(x,2)) so steering effectiveness reads directly off how far
    the steered curve moved from the c=1 target toward the c=2 one."""
    w_dom_vec, w_logreg_vec = _steer_vectors(w_dom, w_probe, steer_scale)
    dtype = next(model.parameters()).dtype
    vecs = {
        "DoM": torch.tensor(w_dom_vec, dtype=dtype, device=device),
        "logreg": torch.tensor(w_logreg_vec, dtype=dtype, device=device),
    }
    xs = torch.linspace(-3, 3, 400, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (label, vec) in zip(axes, vecs.items()):
        for j in range(num_x):
            x = torch.zeros(len(xs), num_x, device=device)
            x[:, j] = xs
            x_full = torch.cat([x, torch.full((len(xs), 1), 1.0, device=device)], dim=1)
            y = forward_steered(model, x_full, steer_layer, vec)[:, j]
            ax.plot(
                xs.cpu().numpy(),
                y.cpu().numpy(),
                color="blue",
                alpha=0.3,
                zorder=5,
            )
        for t, (ls, lbl) in {
            1.0: ("k--", "target sat(x,1)"),
            2.0: ("r--", "target sat(x,2)"),
        }.items():
            ax.plot(
                xs.cpu().numpy(),
                torch.clamp(xs, -t, t).cpu().numpy(),
                ls,
                lw=1.5,
                label=lbl,
                zorder=2,
            )
        ax.set_title(f"{label} direction")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("y (c=1 input, steered toward c=2)")
    axes[0].set_ylim([-2.8, 2.8])
    fig.suptitle(
        f"steering effectiveness, DoM vs logreg direction ({tag}, layer {steer_layer})"
    )
    fig.tight_layout()
    save_plot(fig, plot_dir, f"{tag}_L{steer_layer}_steer_cmp.png")


@torch.no_grad()
def _linear_y_reconstruction(model, num_x, num_blocks, n_train, n_test, g, device):
    """Fit a linear map residual[layer] -> model's own final task output y,
    for every residual-stream layer 0..num_blocks, with c ~ U[1,2] (the
    training distribution, not pinned pairs). Layer num_blocks should score
    ~1 (y is exactly linear in it); see module docstring for what an
    early-layer score implies.

    Always fit and evaluate noise-free (unlike the binary probe metrics
    above, which do inject eval-time noise): this analysis asks whether c's
    contribution to y is linearly decodable at a given layer, which OLS
    answers on its own (a genuinely nonlinear map just gets a low R²). Eval
    noise the fit never saw at train time instead measures how much a
    layer's fitted coefficients amplify an unrelated perturbation -- a
    layer whose fit happens to need large coefficients (e.g. because its
    encoding is scaled down) can swing to wildly negative R² for reasons
    having nothing to do with linearity."""
    layers = list(range(0, num_blocks + 1))

    def _sample(n):
        x_full, _ = sample_batch(n, num_x, generator=g, device=device)
        pred, caches = model.forward(x_full, return_cache=True, generator=g)
        y = pred[:, :num_x]
        return {lyr: caches[lyr].cpu().numpy() for lyr in layers}, y.cpu().numpy()

    train_caches, y_train = _sample(n_train)
    test_caches, y_test = _sample(n_test)

    r2 = {}
    for lyr in tqdm(layers, desc="linear-y per layer", leave=False):
        reg = LinearRegression().fit(train_caches[lyr], y_train)
        pred_te = reg.predict(test_caches[lyr])
        r2[lyr] = float(r2_score(y_test, pred_te))
    return r2


def _plot_linear_y_reconstruction(tag, r2, penalty_layers, plot_dir):
    """Plots R² relative to the L0 (embedding) baseline, not raw R² -- sat(x,c)
    is already close to linear (identity off-saturation), so even a linear map
    from the raw input achieves a high raw R². Rescaling so L0 -> 0 and perfect
    reconstruction -> 1 isolates how much each layer improves ON TOP OF that
    trivial baseline; a layer scoring BELOW the baseline shows negative."""
    layers = sorted(r2)
    baseline = r2[0]
    rel_r2 = {l: (r2[l] - baseline) / (1 - baseline + 1e-9) for l in layers}

    x = np.arange(len(layers))
    colors = ["crimson" if l in penalty_layers else "steelblue" for l in layers]
    fig, ax = plt.subplots(figsize=(max(3.5, 0.65 * len(layers)), 4.2))
    ax.bar(x, [rel_r2[l] for l in layers], color=colors)
    line_ref = ax.axhline(
        1.0, color="k", ls="--", lw=1, label="perfect linear reconstruction"
    )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("R² relative to L0 (embedding) baseline")
    ax.set_title(f"linear reconstruction of final y, per layer ({tag})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="crimson", label="penalized layer"),
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="unpenalized layer"),
        line_ref,
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_plot(fig, plot_dir, f"{tag}_linear_y.png")


# ----------------------------------------------------------------------------
def _build_report(args, model, adv_cfg, result: "AnalysisResult"):
    """Assemble the full report text from already-computed data. Returns the
    report as a list of lines; produces no side effects (no printing)."""
    lines = []

    def emit(s=""):
        lines.append(s)

    emit(f"# Adversarial analysis — tag={args.tag} ckpt={args.ckpt}")
    emit()
    lam = adv_cfg.lam if adv_cfg is not None else None
    penalty_layers = adv_cfg.penalty_layers if adv_cfg is not None else "n/a"
    emit(
        f"config: num_x={model.num_x} d_model={model.d_model} d_mlp={model.d_mlp} "
        f"num_blocks={model.num_blocks} lam={lam} "
        f"penalty_layers={penalty_layers}"
    )
    emit()

    # --- 1. task fidelity ---
    emit(f"## 1. Task fidelity")
    emit(f"max abs elementwise error (c~U[1,2]): {result.me:.3e}")
    emit()

    if adv_cfg is None:
        emit(
            "## 2+. Probe analysis skipped — c-blind demonstration run "
            "(train_no_c.py): c was never given to the model, so there is "
            "no c signal in its activations to probe for."
        )
        emit()
        return lines

    hidden_layers = list(range(1, model.num_blocks))

    # --- 2. probe-strength gap at c in {1,2} ---
    emit("## 2. Probe-strength gap at c in {1,2} (per hidden layer)")
    emit("layer | penalized | DoM ||Δμ|| |  DoM acc | logreg acc |  LDA acc")
    emit("------|-----------|-----------|----------|------------|---------")
    for lyr in hidden_layers:
        m = result.gap[lyr]
        pen = "yes" if lyr in penalty_layers else "no"
        emit(
            f"  L{lyr}  |   {pen:>3s}     | {m['delta_norm']:.3e} | "
            f"{m['dom']:.4f}  |   {m['logreg']:.4f}   | {m['lda']:.4f}"
        )
    emit()

    if args.detailed:
        # --- 3b. binary held-out pairs (asymmetric about 1.5) ---
        emit("## 3b. Binary held-out c pairs (NOT in {1,2}; asymmetric about 1.5)")
        emit("pair       | layer | DoM acc | logreg acc | LDA acc")
        emit("-----------|-------|---------|------------|--------")
        for c_lo, c_hi in args.held_out_pairs:
            for lyr in hidden_layers:
                m = result.heldout[(c_lo, c_hi, lyr)]
                emit(
                    f"{c_lo:.2f}/{c_hi:.2f} |  L{lyr}  | {m['dom']:.4f}  |  "
                    f"{m['logreg']:.4f}   | {m['lda']:.4f}"
                )
        emit()

    if result.linear_y_r2 is not None:
        # --- 4. linear reconstruction of final y from each layer ---
        emit("## 4. Linear reconstruction of final y (c ~ U[1,2], per layer)")
        emit("layer | penalized | R²")
        emit("------|-----------|-----")
        for lyr in sorted(result.linear_y_r2):
            pen = "yes" if lyr in penalty_layers else "no"
            emit(f"  L{lyr}  |   {pen:>3s}     | {result.linear_y_r2[lyr]:.4f}")
        emit()

    return lines


@dataclasses.dataclass
class AnalysisResult:
    """Everything computed in phase 1, needed by both the report (phase 2)
    and the plots (phase 3). The probe-derived fields are None for a c-blind
    checkpoint (adv_cfg is None) -- see `_run_analysis`."""

    me: float
    gap: dict | None
    gap_plot_inputs: dict | None
    heldout: dict | None
    linear_y_r2: dict | None


def _validate_steer_layers(steer_layers, hidden_layers):
    for lyr in steer_layers:
        assert lyr in hidden_layers, (
            f"--steer layer {lyr} must be one of the hidden layers already "
            f"probed at c in {{1,2}}: {hidden_layers}"
        )


def _run_analysis(
    model, adv_cfg, args, g, device, probe_backend_name
) -> AnalysisResult:
    """Phase 1: all data generation, no plotting or printing.

    A c-blind checkpoint (adv_cfg is None) has no c signal anywhere in its
    activations to probe for (train_no_c.py zeroes c's input coordinate
    entirely) -- only task fidelity is meaningful there, so the probe-derived
    fields come back None."""
    me = eval_max_err(model, g, device=device)

    if adv_cfg is None:
        return AnalysisResult(me, None, None, None, None)

    hidden_layers = list(range(1, model.num_blocks))
    # Multiplier on the model's OWN training-time noise, injected only when
    # evaluating probes (see --eval-noise-mult help).
    eval_noise_std = adv_cfg.resid_noise_std * args.eval_noise_mult

    gap, gap_plot_inputs = binary_probe_metrics_all_layers(
        model,
        1.0,
        2.0,
        hidden_layers,
        args.n_train,
        args.n_test,
        g,
        probe_backend_name,
        desc="probe gap @ {1,2}",
        eval_noise_std=eval_noise_std,
    )

    heldout = {}
    if args.detailed:
        for c_lo, c_hi in tqdm(args.held_out_pairs, desc="held-out pairs"):
            pair_metrics, _ = binary_probe_metrics_all_layers(
                model,
                c_lo,
                c_hi,
                hidden_layers,
                args.n_train,
                args.n_test,
                g,
                probe_backend_name,
                desc=f"held-out {c_lo:g}-{c_hi:g}",
                eval_noise_std=eval_noise_std,
            )
            for lyr in hidden_layers:
                heldout[(c_lo, c_hi, lyr)] = pair_metrics[lyr]

    linear_y_r2 = None
    if args.detailed:
        linear_y_r2 = _linear_y_reconstruction(
            model,
            model.num_x,
            model.num_blocks,
            args.n_train,
            args.n_test,
            g,
            device,
        )

    return AnalysisResult(me, gap, gap_plot_inputs, heldout, linear_y_r2)


def _make_plots(args, model, adv_cfg, result: AnalysisResult, plot_dir, device):
    """Phase 3: everything currently after the report is written in main().

    Plots that depend on probing c (probe gap, layer distributions,
    separation, steering, held-out gap, linear-y reconstruction) are skipped
    for a c-blind checkpoint (adv_cfg is None) -- see `_run_analysis`."""
    class_threshold = adv_cfg.class_threshold if adv_cfg is not None else None

    out_log = log_dir(args.tag)
    hist_path = os.path.join(out_log, "history.jsonl")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = [json.loads(line) for line in f if line.strip()]
        _plot_training_traces(
            args.tag,
            history,
            plot_dir,
            device,
            class_threshold,
            args.eval_noise_mult,
            show=args.show,
        )
    plot_learned_curves(model, args.tag, plot_dir, show=args.show)

    if adv_cfg is not None:
        hidden_layers = list(range(1, model.num_blocks))
        gap = result.gap
        gap_plot_inputs = result.gap_plot_inputs
        heldout = result.heldout
        linear_y_r2 = result.linear_y_r2

        _plot_probe_gap(args.tag, hidden_layers, gap, plot_dir)
        _plot_layer_distributions(
            args.tag, 1.0, 2.0, hidden_layers, gap_plot_inputs, plot_dir
        )
        for lyr in hidden_layers:
            pi = gap_plot_inputs[lyr]
            # LinearBoundary scores as `w . x + b`, not `w . x - threshold`, so
            # its bias is the midpoint threshold negated.
            b_dom = -pi["midpoint"]
            plot_probe_separation(
                "c1-2",
                [lyr],
                LinearBoundary(pi["w_dom"], b_dom),
                LinearBoundary(pi["w_probe"], pi["b_probe"]),
                pi["X_te"],
                pi["y_te"],
                plot_dir,
            )
            if args.detailed:
                plot_probe_pca_separation(
                    "c1-2",
                    [lyr],
                    LinearBoundary(pi["w_dom"], b_dom),
                    LinearBoundary(pi["w_probe"], pi["b_probe"]),
                    pi["X_te"],
                    pi["y_te"],
                    plot_dir,
                )
        if args.steer:
            for lyr in args.steer:
                pi = gap_plot_inputs[lyr]
                _plot_steer_comparison(
                    args.tag,
                    lyr,
                    model.num_x,
                    model,
                    pi["w_dom"],
                    pi["w_probe"],
                    args.steer_scale,
                    plot_dir,
                    device,
                )

        if args.detailed:
            _plot_heldout_gap(
                args.tag, hidden_layers, args.held_out_pairs, gap, heldout, plot_dir
            )

        if linear_y_r2 is not None:
            _plot_linear_y_reconstruction(
                args.tag, linear_y_r2, adv_cfg.penalty_layers, plot_dir
            )

    if args.show:
        plt.show()
        plt.close("all")


# ----------------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.tag, args.ckpt, device)
    adv_cfg = resolve_adv_config(ck)
    if args.steer:
        assert adv_cfg is not None, (
            "--steer needs probe directions from an adversarial-trained "
            "checkpoint; this checkpoint has no adv_config (c-blind "
            "demonstration run) so there's nothing to steer along."
        )
        _validate_steer_layers(args.steer, list(range(1, model.num_blocks)))
    plot_dir = get_plot_dir(args.tag)
    os.makedirs(plot_dir, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(args.seed)
    probe_backend_name = resolve_probe_backend(args.probe_backend, device)

    result = _run_analysis(model, adv_cfg, args, g, device, probe_backend_name)

    lines = _build_report(args, model, adv_cfg, result)
    print("\n".join(lines))
    out_log = log_dir(args.tag)
    os.makedirs(out_log, exist_ok=True)
    report_path = os.path.join(out_log, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {report_path}")

    _make_plots(args, model, adv_cfg, result, plot_dir, device)


if __name__ == "__main__":
    main(args)
