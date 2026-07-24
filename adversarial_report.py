"""Step 3 diagnostics — did the adversarially-trained model HIDE c or ERASE it?

Loads an adversarial checkpoint (from train_adversarial.py) and produces the
report that IS the Step-3 deliverable. It distinguishes the outcomes the plan
cares about:

  1. Task fidelity           — the PRICE of hiding (expected near-zero).
  2. Probe-strength gap at the probed points, per hidden layer.
  3. Held-out c recovery (the hidden-vs-erased test) — binary probes at
     held-out c pairs chosen asymmetric about the training midpoint so an
     affine cancellation can't masquerade as erasure.

The held-out-pairs table is the most expensive part of the script and is
only printed/written with --detailed.

Optionally pass --baseline-path to run the same probes on the pre-adversarial
model for a before/after contrast.

Three optional deep-dive diagnostics, each opt-in since they cost extra compute:
  --detailed        also plots the held-out-pair accuracy gap across layers
                     (extends the table above into a bar chart).
  --steer L1,L2,...  causal test: inject the DoM- and logreg-discovered c=1->c=2
                     directions at each given layer and see whether the model's
                     output actually moves toward the c=2 target -- a probe that
                     merely correlates with c but isn't causally used would fail
                     this even at high probe accuracy.
  --linear-y-probe   fits a linear map residual[layer] -> model's own final y,
                     for every layer. Tests whether y is already linearly
                     recoverable well before the final unembed: if so, the
                     remaining blocks don't need to keep carrying c information,
                     they can just carry y forward through always-on neurons.
"""

import argparse
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
        description="Step 3 diagnostics: hidden vs. erased c.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="last", choices=["best", "last"])
    p.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="optional checkpoint to run the same probes on for a before/after "
        "contrast (e.g. the pre-adversarial model).",
    )
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
        help="also compute the report-only statistics that feed no saved plot: "
        "binary held-out c pairs (3b), plus the held-out-pair accuracy-gap bar "
        "chart. Skipped by default since they're the most expensive part of "
        "the script.",
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
        "--linear-y-probe",
        action="store_true",
        help="fit a linear map residual[layer] -> model's own final y at every "
        "layer (0..num_blocks) and report/plot R^2. Tests whether y is already "
        "linearly recoverable well before the final unembed.",
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

from data import sample_batch, sample_fixed_c
from model import ResidualMLP
from paths import log_dir
from paths import plot_dir as get_plot_dir
from data import eval_max_err
from probe_backend import build_probe_pipeline, resolve_probe_backend
from train_model_plot import plot_learned_curves
from train_probe import capture_layers_dict, forward_steered, load_model
from train_probe import plot_probe as plot_probe_separation


# ----------------------------------------------------------------------------
# Probe primitives (reuse train_probe's harness where possible)
# ----------------------------------------------------------------------------
def _dom_accuracy(r_lo_tr, r_hi_tr, r_lo_te, r_hi_te):
    """Raw difference-of-means classifier (train direction, test accuracy)."""
    mu_lo = r_lo_tr.mean(dim=0)
    mu_hi = r_hi_tr.mean(dim=0)
    w = (mu_hi - mu_lo).cpu().numpy()
    midpoint = float(((mu_hi + mu_lo) / 2).cpu().numpy() @ w)
    X_te = np.concatenate([r_lo_te.cpu().numpy(), r_hi_te.cpu().numpy()], axis=0)
    y_te = np.concatenate([np.zeros(len(r_lo_te)), np.ones(len(r_hi_te))])
    pred = (X_te @ w > midpoint).astype(float)
    delta_norm = float(np.linalg.norm(w))
    return float((pred == y_te).mean()), delta_norm


def _raw_signed_distance(w_probe, b_probe, X):
    """Signed distance to the probe's decision boundary in raw (unstandardized)
    data units, boundary at 0 -- same fold as train_probe.plot_probe's
    raw-space panel."""
    w_hat = w_probe / np.linalg.norm(w_probe)
    threshold = -b_probe / np.linalg.norm(w_probe)
    return X @ w_hat - threshold


@torch.no_grad()
def _binary_dataset_all_layers(model, num_x, n, c_lo, c_hi, layers, generator, device):
    """One forward pass over c_lo and one over c_hi, shared across all
    `layers` -- returns {layer: (r_lo, r_hi)}."""
    xf_lo, _ = sample_fixed_c(n, num_x, c_lo, generator=generator, device=device)
    xf_hi, _ = sample_fixed_c(n, num_x, c_hi, generator=generator, device=device)
    r_lo = capture_layers_dict(model, xf_lo, layers)
    r_hi = capture_layers_dict(model, xf_hi, layers)
    return {layer: (r_lo[layer], r_hi[layer]) for layer in layers}


def _binary_probe_metrics_all_layers(
    model,
    c_lo,
    c_hi,
    layers,
    n_train,
    n_test,
    g,
    probe_backend_name,
    desc="layers",
):
    """DoM / logreg / LDA accuracy for every layer in `layers`, one (c_lo,
    c_hi) pair, from a single shared forward pass per train/test set.

    Pure data generation -- no plotting. Returns `(metrics, plot_inputs)`:
    `metrics` is {layer: {"dom", "delta_norm", "logreg", "lda"}}, and
    `plot_inputs` is {layer: {"w_dom", "midpoint", "w_probe", "b_probe",
    "X_te", "y_te", "dist_lo", "dist_hi"}}, everything a caller needs to
    later feed train_probe.plot_probe or _plot_layer_distributions for this
    (c_lo, c_hi) pair -- returned unconditionally since the forward pass and
    the fit already happened regardless of whether the caller wants a plot.

    The logreg probe is fit via `probe_backend` (sklearn or GPU-resident
    torch, per `probe_backend_name`): X/y are kept as torch tensors on
    `device` end-to-end for the torch backend, so it actually skips the numpy
    round-trip rather than just wrapping the same CPU path under a new name.
    LDA has no GPU variant and stays on the numpy/sklearn path.
    """
    num_x = model.num_x
    device = next(model.parameters()).device
    train_ds = _binary_dataset_all_layers(
        model, num_x, n_train, c_lo, c_hi, layers, g, device
    )
    test_ds = _binary_dataset_all_layers(
        model, num_x, n_test, c_lo, c_hi, layers, g, device
    )

    metrics = {}
    plot_inputs = {}
    for layer in tqdm(layers, desc=desc, leave=False):
        r_lo_tr, r_hi_tr = train_ds[layer]
        r_lo_te, r_hi_te = test_ds[layer]
        dom_acc, delta_norm = _dom_accuracy(r_lo_tr, r_hi_tr, r_lo_te, r_hi_te)

        X_tr = np.concatenate([r_lo_tr.cpu().numpy(), r_hi_tr.cpu().numpy()], axis=0)
        y_tr = np.concatenate([np.zeros(n_train), np.ones(n_train)])
        X_te = np.concatenate([r_lo_te.cpu().numpy(), r_hi_te.cpu().numpy()], axis=0)
        y_te = np.concatenate([np.zeros(n_test), np.ones(n_test)])
        lda = LinearDiscriminantAnalysis().fit(X_tr, y_tr)

        X_tr_t = torch.cat([r_lo_tr, r_hi_tr], dim=0)
        y_tr_t = torch.cat(
            [
                torch.zeros(n_train, dtype=torch.bool, device=device),
                torch.ones(n_train, dtype=torch.bool, device=device),
            ]
        )
        X_te_t = torch.cat([r_lo_te, r_hi_te], dim=0)
        y_te_t = torch.cat(
            [
                torch.zeros(n_test, dtype=torch.bool, device=device),
                torch.ones(n_test, dtype=torch.bool, device=device),
            ]
        )
        pipeline = build_probe_pipeline(
            C=1.0, max_iter=2000, backend=probe_backend_name
        )
        pipeline.fit(X_tr_t, y_tr_t)
        w_probe_t, b_probe_t = pipeline.get_affine(device)
        logreg_pred = (X_te_t @ w_probe_t + b_probe_t) > 0
        logreg_acc = float((logreg_pred == y_te_t).float().mean())
        w_probe = w_probe_t.cpu().numpy()
        b_probe = float(b_probe_t.cpu())

        mu_lo = r_lo_tr.mean(dim=0)
        mu_hi = r_hi_tr.mean(dim=0)
        w_dom = (mu_hi - mu_lo).cpu().numpy()
        midpoint = float(((mu_hi + mu_lo) / 2).cpu().numpy() @ w_dom)

        metrics[layer] = {
            "dom": dom_acc,
            "delta_norm": delta_norm,
            "logreg": logreg_acc,
            "lda": float(lda.score(X_te, y_te)),
        }
        plot_inputs[layer] = {
            "w_dom": w_dom,
            "midpoint": midpoint,
            "w_probe": w_probe,
            "b_probe": b_probe,
            "X_te": X_te,
            "y_te": y_te,
            "dist_lo": _raw_signed_distance(w_probe, b_probe, r_lo_te.cpu().numpy()),
            "dist_hi": _raw_signed_distance(w_probe, b_probe, r_hi_te.cpu().numpy()),
        }
    return metrics, plot_inputs


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _plot_training_traces(tag, history, hidden_layers, plot_dir):
    pts = [h for h in history if h.get("l_task") is not None]
    if not pts:
        return
    its = [h["iter"] for h in pts]
    fig, (ax_err, ax_dom, ax_loss) = plt.subplots(1, 3, figsize=(16, 4.2))

    ax_err.semilogy(its, [h["max_err"] for h in pts], color="crimson")
    ax_err.set_title("task fidelity (price of hiding)")
    ax_err.set_xlabel("iter")
    ax_err.set_ylabel("max abs error")
    ax_err.grid(True, alpha=0.3)

    if any("delta_norms" in h for h in pts):
        for lyr in hidden_layers:
            key = str(lyr)
            ys = [h.get("delta_norms", {}).get(key, float("nan")) for h in pts]
            ax_dom.semilogy(its, ys, label=f"layer {lyr}")
        ax_dom.legend(fontsize=8)
    else:
        ax_dom.text(
            0.5,
            0.5,
            "no delta_norms in history",
            ha="center",
            va="center",
            transform=ax_dom.transAxes,
        )
    ax_dom.set_title("penalized DoM  ||Δμ||  per hidden layer")
    ax_dom.set_xlabel("iter")
    ax_dom.set_ylabel("||mean(c=2) - mean(c=1)||")
    ax_dom.grid(True, alpha=0.3)

    ax_loss.semilogy(its, [h["l_task"] for h in pts], label="L_task")
    ax_loss.semilogy(its, [max(h["l_probe"], 1e-30) for h in pts], label="L_probe")
    ax_loss.set_title("loss terms")
    ax_loss.set_xlabel("iter")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    fig.suptitle(f"adversarial training traces ({tag})")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{tag}_training.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


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
    path = os.path.join(plot_dir, f"{tag}_probe_gap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


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
    path = os.path.join(plot_dir, f"{tag}_heldout_gap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


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
    path = os.path.join(plot_dir, f"{tag}_c{c_lo:g}-{c_hi:g}_layer_dist.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


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
    Only the steered curves are shown (not the unsteered/reference panels
    train_probe.py's version has) -- each panel already carries both targets
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
                color="steelblue",
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
    fig.suptitle(
        f"steering effectiveness, DoM vs logreg direction ({tag}, layer {steer_layer})"
    )
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{tag}_L{steer_layer}_steer_cmp.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


@torch.no_grad()
def _linear_y_reconstruction(model, num_x, num_blocks, n_train, n_test, g, device):
    """Fit a linear map residual[layer] -> model's own final task output y,
    for every residual-stream layer 0..num_blocks (embedding through the
    final residual, inclusive). Layer num_blocks is a sanity anchor: y IS a
    linear map of it (y = r_num_blocks @ W_U), so R^2 there should be ~1
    regardless of anything else. Tests whether y is already linearly
    recoverable well before that point -- if so, downstream blocks don't need
    to keep encoding c, they can just carry y forward through always-on
    neurons. Samples c ~ U[1,2] (the training distribution), not pinned pairs,
    since this asks about the model's actual behavior, not a probe contrast."""
    layers = list(range(0, num_blocks + 1))

    def _sample(n):
        x_full, _ = sample_batch(n, num_x, generator=g, device=device)
        pred, caches = model.forward(x_full, return_cache=True)
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
    layers = sorted(r2)
    x = np.arange(len(layers))
    colors = ["crimson" if l in penalty_layers else "steelblue" for l in layers]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(layers)), 4.2))
    ax.bar(x, [r2[l] for l in layers], color=colors)
    line_ref = ax.axhline(
        1.0, color="k", ls="--", lw=1, label="perfect linear reconstruction"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("R² (linear map -> model's y)")
    ax.set_title(f"linear reconstruction of final y, per layer ({tag})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="crimson", label="penalized layer"),
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="unpenalized layer"),
        line_ref,
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{tag}_linear_y.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


# ----------------------------------------------------------------------------
def _build_report(
    args,
    num_x,
    model,
    ck,
    num_blocks,
    penalty_layers,
    hidden_layers,
    me,
    me_b,
    gap,
    heldout,
    linear_y_r2,
):
    """Assemble the full report text from already-computed data. Returns the
    report as a list of lines; produces no side effects (no printing)."""
    lines = []

    def emit(s=""):
        lines.append(s)

    emit(f"# Step 3 adversarial diagnostics — tag={args.tag} ckpt={args.ckpt}")
    emit()
    emit(
        f"config: num_x={num_x} d_model={model.d_model} d_mlp={model.d_mlp} "
        f"num_blocks={num_blocks} lam={ck.get('lam')} init={ck.get('init')} "
        f"penalty_layers={penalty_layers}"
    )
    emit()

    # --- 1. task fidelity ---
    emit(f"## 1. Task fidelity")
    emit(f"max abs elementwise error (c~U[1,2]): {me:.3e}")
    if me_b is not None:
        emit(f"  baseline max abs error: {me_b:.3e}")
    emit()

    # --- 2. probe-strength gap at c in {1,2} ---
    emit("## 2. Probe-strength gap at c in {1,2} (per hidden layer)")
    emit("layer | penalized | DoM ||Δμ|| |  DoM acc | logreg acc |  LDA acc")
    emit("------|-----------|-----------|----------|------------|---------")
    for lyr in hidden_layers:
        m = gap[lyr]
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
                m = heldout[(c_lo, c_hi, lyr)]
                emit(
                    f"{c_lo:.2f}/{c_hi:.2f} |  L{lyr}  | {m['dom']:.4f}  |  "
                    f"{m['logreg']:.4f}   | {m['lda']:.4f}"
                )
        emit()

    if linear_y_r2 is not None:
        # --- 4. linear reconstruction of final y from each layer ---
        emit("## 4. Linear reconstruction of final y (c ~ U[1,2], per layer)")
        emit("layer | penalized | R²")
        emit("------|-----------|-----")
        for lyr in sorted(linear_y_r2):
            pen = "yes" if lyr in penalty_layers else "no"
            emit(f"  L{lyr}  |   {pen:>3s}     | {linear_y_r2[lyr]:.4f}")
        emit()

    return lines


# ----------------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ck = load_model(args.tag, args.ckpt, device)
    num_x = model.num_x
    num_blocks = model.num_blocks
    penalty_layers = ck.get("penalty_layers") or list(range(1, num_blocks))
    hidden_layers = list(range(1, num_blocks))

    if args.steer:
        for lyr in args.steer:
            assert lyr in hidden_layers, (
                f"--steer layer {lyr} must be one of the hidden layers already "
                f"probed at c in {{1,2}}: {hidden_layers}"
            )

    plot_dir = get_plot_dir(args.tag)
    os.makedirs(plot_dir, exist_ok=True)

    base_model = None
    if args.baseline_path:
        base_model, _ = ResidualMLP.load(args.baseline_path, map_location=device)
        base_model = base_model.to(device)
        base_model.eval()

    g = torch.Generator(device=device).manual_seed(args.seed)
    probe_backend_name = resolve_probe_backend(args.probe_backend, device)

    # --- phase 1: generate all data ---
    me = eval_max_err(model, num_x, g, device=device)
    me_b = eval_max_err(base_model, num_x, g, device=device) if base_model else None

    gap, gap_plot_inputs = _binary_probe_metrics_all_layers(
        model,
        1.0,
        2.0,
        hidden_layers,
        args.n_train,
        args.n_test,
        g,
        probe_backend_name,
        desc="probe gap @ {1,2}",
    )

    heldout = {}
    if args.detailed:
        for c_lo, c_hi in tqdm(args.held_out_pairs, desc="held-out pairs"):
            pair_metrics, _ = _binary_probe_metrics_all_layers(
                model,
                c_lo,
                c_hi,
                hidden_layers,
                args.n_train,
                args.n_test,
                g,
                probe_backend_name,
                desc=f"held-out {c_lo:g}-{c_hi:g}",
            )
            for lyr in hidden_layers:
                heldout[(c_lo, c_hi, lyr)] = pair_metrics[lyr]

    linear_y_r2 = None
    if args.linear_y_probe:
        linear_y_r2 = _linear_y_reconstruction(
            model, num_x, num_blocks, args.n_train, args.n_test, g, device
        )

    # --- phase 2: build + write the report ---
    lines = _build_report(
        args,
        num_x,
        model,
        ck,
        num_blocks,
        penalty_layers,
        hidden_layers,
        me,
        me_b,
        gap,
        heldout,
        linear_y_r2,
    )
    print("\n".join(lines))

    out_log = log_dir(args.tag)
    os.makedirs(out_log, exist_ok=True)
    report_path = os.path.join(out_log, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {report_path}")

    # --- phase 3: generate all plots ---
    hist_path = os.path.join(out_log, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        _plot_training_traces(args.tag, history, hidden_layers, plot_dir)
    _plot_probe_gap(args.tag, hidden_layers, gap, plot_dir)
    _plot_layer_distributions(
        args.tag, 1.0, 2.0, hidden_layers, gap_plot_inputs, plot_dir
    )
    for lyr in hidden_layers:
        pi = gap_plot_inputs[lyr]
        plot_probe_separation(
            "c1-2",
            [lyr],
            pi["w_dom"],
            pi["midpoint"],
            pi["w_probe"],
            pi["b_probe"],
            pi["X_te"],
            pi["y_te"],
            plot_dir,
        )
    plot_learned_curves(model, args.tag, plot_dir)
    if base_model is not None:
        plot_learned_curves(base_model, f"{args.tag}_baseline", plot_dir)

    if args.detailed:
        _plot_heldout_gap(
            args.tag, hidden_layers, args.held_out_pairs, gap, heldout, plot_dir
        )

    if args.steer:
        for lyr in args.steer:
            pi = gap_plot_inputs[lyr]
            _plot_steer_comparison(
                args.tag,
                lyr,
                num_x,
                model,
                pi["w_dom"],
                pi["w_probe"],
                args.steer_scale,
                plot_dir,
                device,
            )

    if linear_y_r2 is not None:
        _plot_linear_y_reconstruction(args.tag, linear_y_r2, penalty_layers, plot_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main(args)
