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
        "binary held-out c pairs (3b). Skipped by default since they're the "
        "most expensive part of the script.",
    )
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data import sample_fixed_c
from model import ResidualMLP
from paths import log_dir
from paths import plot_dir as get_plot_dir
from data import eval_max_err
from probe_backend import build_probe_pipeline, resolve_probe_backend
from train_model_plot import plot_learned_curves
from train_probe import capture_layers_dict, load_model
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
    "X_te", "y_te"}}, everything a caller needs to later feed
    train_probe.plot_probe for this (c_lo, c_hi) pair -- returned
    unconditionally since the forward pass and the fit already happened
    regardless of whether the caller wants a plot.

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

    return lines


# ----------------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ck = load_model(args.tag, args.ckpt, device)
    num_x = model.num_x
    num_blocks = model.num_blocks
    penalty_layers = ck.get("penalty_layers") or list(range(1, num_blocks))
    hidden_layers = list(range(1, num_blocks))

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

    if args.show:
        plt.show()


if __name__ == "__main__":
    main(args)
