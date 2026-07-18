"""Step 3 diagnostics — did the adversarially-trained model HIDE c or ERASE it?

Loads an adversarial checkpoint (from train_adversarial.py) and produces the
report that IS the Step-3 deliverable. It distinguishes the outcomes the plan
cares about:

  1. Task fidelity     — max-abs elementwise error (the PRICE of hiding; expected
                         near-zero, i.e. hiding is nearly free).
  2. Probe-strength gap at the probed points c in {1,2}, per hidden layer:
                         raw DoM vs logreg vs LDA accuracy. If DoM -> chance but
                         logreg/LDA still separate, the model defeated the
                         *specific* (first-moment) probe while leaving
                         second-moment structure readable — a reportable outcome.
  3. Held-out c recovery (the hidden-vs-erased test):
       - continuous ridge probe on c ~ U[1,2] (R^2 of recovered c) per layer:
         the cleanest "is c linearly present across the whole range" measure.
       - binary logreg/LDA at held-out c pairs NOT in {1,2}. Pairs are chosen
         ASYMMETRIC about c=1.5 on purpose: the residual of an affine
         cancellation is a quadratic in c with vertex at 1.5, so a symmetric
         pair (e.g. 1.25/1.75) reads ~0 even when c is merely hidden — you'd
         falsely conclude "erased".

Reading the outcomes (heuristic, printed at the end):
  - penalized DoM -> chance, but ridge R^2 / held-out binary stay high  => HIDDEN
    (point-hiding: class means cancelled only at {1,2}; c still represented).
  - no linear probe recovers c anywhere in [1,2]                        => ERASED.
  - DoM -> chance but logreg/LDA at {1,2} stay high                     => defeated
    the specific probe only (means match, covariances differ).

Optionally pass --baseline-path to run the same probes on the pre-adversarial
model (e.g. runs/nx32/checkpoints/best.pt) for a before/after contrast.

Usage:
    python adversarial_report.py --tag adv1
    python adversarial_report.py --tag adv1 --baseline-path runs/nx32/checkpoints/best.pt
"""

import argparse
import json
import os


def parse_pairs(s: str) -> list[tuple[float, float]]:
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
        type=parse_pairs,
        default="1.0-1.5,1.0-1.75,1.25-2.0",
        help="binary held-out c pairs, comma-separated 'lo-hi'. Kept ASYMMETRIC "
        "about 1.5 on purpose (see module docstring).",
    )
    p.add_argument("--n-train", type=int, default=20_000, help="per class/set")
    p.add_argument("--n-test", type=int, default=50_000, help="per class/set")
    p.add_argument(
        "--n-ridge", type=int, default=50_000, help="samples for the ridge probe"
    )
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score

from data import sample_batch
from model import ResidualMLP
from paths import log_dir
from train_model import eval_max_err
from train_probe import binary_dataset, capture_layers, load_model


# ----------------------------------------------------------------------------
# Probe primitives (reuse train_probe's harness where possible)
# ----------------------------------------------------------------------------
def dom_accuracy(r_lo_tr, r_hi_tr, r_lo_te, r_hi_te):
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


def binary_probe_metrics(
    model, num_x, c_lo, c_hi, layer, n_train, n_test, g_train, g_test, device
):
    """DoM / logreg / LDA accuracy for one layer, one (c_lo, c_hi) pair."""
    r_lo_tr, r_hi_tr = binary_dataset(
        model, num_x, n_train, c_lo, c_hi, [layer], g_train, device
    )
    r_lo_te, r_hi_te = binary_dataset(
        model, num_x, n_test, c_lo, c_hi, [layer], g_test, device
    )
    dom_acc, delta_norm = dom_accuracy(r_lo_tr, r_hi_tr, r_lo_te, r_hi_te)

    X_tr = np.concatenate([r_lo_tr.cpu().numpy(), r_hi_tr.cpu().numpy()], axis=0)
    y_tr = np.concatenate([np.zeros(n_train), np.ones(n_train)])
    X_te = np.concatenate([r_lo_te.cpu().numpy(), r_hi_te.cpu().numpy()], axis=0)
    y_te = np.concatenate([np.zeros(n_test), np.ones(n_test)])

    logreg = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    lda = LinearDiscriminantAnalysis().fit(X_tr, y_tr)
    return {
        "dom": dom_acc,
        "delta_norm": delta_norm,
        "logreg": float(logreg.score(X_te, y_te)),
        "lda": float(lda.score(X_te, y_te)),
    }


def ridge_r2(model, num_x, layer, n_train, n_test, alpha, g_train, g_test, device):
    """R^2 of a ridge probe recovering continuous c ~ U[1,2] from layer l."""

    def ds(n, g):
        x_full, _ = sample_batch(n, num_x, generator=g, device=device)
        r = capture_layers(model, x_full, [layer])
        c = x_full[:, num_x]
        return r.cpu().numpy(), c.cpu().numpy()

    X_tr, c_tr = ds(n_train, g_train)
    X_te, c_te = ds(n_test, g_test)
    reg = Ridge(alpha=alpha).fit(X_tr, c_tr)
    return float(r2_score(c_te, reg.predict(X_te)))


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_training_traces(tag, history, hidden_layers, out_dir):
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

    for lyr in hidden_layers:
        key = str(lyr)
        ys = [h["delta_norms"].get(key, float("nan")) for h in pts]
        ax_dom.semilogy(its, ys, label=f"layer {lyr}")
    ax_dom.set_title("penalized DoM  ||Δμ||  per hidden layer")
    ax_dom.set_xlabel("iter")
    ax_dom.set_ylabel("||mean(c=2) - mean(c=1)||")
    ax_dom.legend(fontsize=8)
    ax_dom.grid(True, alpha=0.3)

    ax_loss.semilogy(its, [h["l_task"] for h in pts], label="L_task")
    ax_loss.semilogy(its, [max(h["l_probe"], 1e-30) for h in pts], label="L_probe")
    ax_loss.set_title("loss terms")
    ax_loss.set_xlabel("iter")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    fig.suptitle(f"adversarial training traces ({tag})")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}_training.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def plot_heldout_r2(tag, layers, r2_adv, r2_base, out_dir):
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if r2_base is not None:
        ax.bar(x - 0.2, [r2_base[l] for l in layers], 0.4, label="baseline", alpha=0.8)
        ax.bar(
            x + 0.2, [r2_adv[l] for l in layers], 0.4, label="adversarial", alpha=0.8
        )
        ax.legend()
    else:
        ax.bar(x, [r2_adv[l] for l in layers], 0.6, label="adversarial")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("ridge R^2 for continuous c")
    ax.set_title(f"held-out c recovery across [1,2] ({tag})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}_heldout_r2.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def plot_probe_gap(tag, hidden_layers, gap, out_dir):
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
    path = os.path.join(out_dir, f"{tag}_probe_gap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}")


# ----------------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = load_model(args.tag, args.ckpt, device)
    num_x = cfg["num_x"]
    num_blocks = cfg.get("num_blocks", 4)
    penalty_layers = cfg.get("penalty_layers") or list(range(1, num_blocks))
    hidden_layers = list(range(1, num_blocks))
    all_layers = list(range(0, num_blocks + 1))  # 0=embed .. num_blocks=output

    base_model = None
    if args.baseline_path:
        bck = torch.load(args.baseline_path, map_location=device)
        bcfg = bck["config"]
        base_model = ResidualMLP(
            bcfg["num_x"],
            bcfg["d_model"],
            bcfg["d_mlp"],
            leaky_relu_slope=bcfg.get("leaky_relu_slope", 0.0),
            num_blocks=bcfg.get("num_blocks", 4),
        ).to(device)
        base_model.load_state_dict(bck["model"])
        base_model.eval()

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Step 3 adversarial diagnostics — tag={args.tag} ckpt={args.ckpt}")
    emit()
    emit(
        f"config: num_x={num_x} d_model={cfg['d_model']} d_mlp={cfg['d_mlp']} "
        f"num_blocks={num_blocks} lam={cfg.get('lam')} init={cfg.get('init')} "
        f"penalty_layers={penalty_layers}"
    )
    emit()

    # --- 1. task fidelity ---
    me = eval_max_err(model, num_x, device=device)
    emit(f"## 1. Task fidelity")
    emit(f"max abs elementwise error (c~U[1,2]): {me:.3e}")
    if base_model is not None:
        me_b = eval_max_err(base_model, num_x, device=device)
        emit(f"  baseline max abs error: {me_b:.3e}")
    emit()

    g = lambda off: torch.Generator(device=device).manual_seed(args.seed + off)

    # --- 2. probe-strength gap at c in {1,2} ---
    emit("## 2. Probe-strength gap at c in {1,2} (per hidden layer)")
    emit("layer | penalized | DoM ||Δμ|| |  DoM acc | logreg acc |  LDA acc")
    emit("------|-----------|-----------|----------|------------|---------")
    gap = {}
    for lyr in hidden_layers:
        m = binary_probe_metrics(
            model,
            num_x,
            1.0,
            2.0,
            lyr,
            args.n_train,
            args.n_test,
            g(10 + lyr),
            g(100 + lyr),
            device,
        )
        gap[lyr] = m
        pen = "yes" if lyr in penalty_layers else "no"
        emit(
            f"  L{lyr}  |   {pen:>3s}     | {m['delta_norm']:.3e} | "
            f"{m['dom']:.4f}  |   {m['logreg']:.4f}   | {m['lda']:.4f}"
        )
    emit()

    # --- 3a. continuous ridge R^2 across [1,2], every layer ---
    emit("## 3a. Continuous ridge probe R^2 for c ~ U[1,2] (per layer)")
    emit("layer | role      | adv R^2  | baseline R^2")
    emit("------|-----------|----------|-------------")
    r2_adv, r2_base = {}, ({} if base_model is not None else None)
    for lyr in all_layers:
        r2a = ridge_r2(
            model,
            num_x,
            lyr,
            args.n_ridge,
            args.n_ridge,
            args.ridge_alpha,
            g(200 + lyr),
            g(300 + lyr),
            device,
        )
        r2_adv[lyr] = r2a
        role = "embed" if lyr == 0 else "output" if lyr == num_blocks else "hidden"
        if base_model is not None:
            r2b = ridge_r2(
                base_model,
                num_x,
                lyr,
                args.n_ridge,
                args.n_ridge,
                args.ridge_alpha,
                g(200 + lyr),
                g(300 + lyr),
                device,
            )
            r2_base[lyr] = r2b
            emit(f"  L{lyr}  | {role:>6s}    | {r2a:+.4f} | {r2b:+.4f}")
        else:
            emit(f"  L{lyr}  | {role:>6s}    | {r2a:+.4f} |     -")
    emit()

    # --- 3b. binary held-out pairs (asymmetric about 1.5) ---
    emit("## 3b. Binary held-out c pairs (NOT in {1,2}; asymmetric about 1.5)")
    emit("pair       | layer | DoM acc | logreg acc | LDA acc")
    emit("-----------|-------|---------|------------|--------")
    heldout = {}
    for c_lo, c_hi in args.held_out_pairs:
        for lyr in hidden_layers:
            m = binary_probe_metrics(
                model,
                num_x,
                c_lo,
                c_hi,
                lyr,
                args.n_train,
                args.n_test,
                g(400 + lyr),
                g(500 + lyr),
                device,
            )
            heldout[(c_lo, c_hi, lyr)] = m
            emit(
                f"{c_lo:.2f}/{c_hi:.2f} |  L{lyr}  | {m['dom']:.4f}  |  "
                f"{m['logreg']:.4f}   | {m['lda']:.4f}"
            )
    emit()

    # --- interpretation heuristic ---
    emit("## Interpretation (heuristic — confirm by eye)")
    pen_hidden = [l for l in hidden_layers if l in penalty_layers]
    dom_at_chance = (
        all(gap[l]["dom"] < 0.6 for l in pen_hidden) if pen_hidden else False
    )
    ridge_hidden_max = max((r2_adv[l] for l in pen_hidden), default=0.0)
    logreg_high = any(gap[l]["logreg"] > 0.9 for l in pen_hidden)
    heldout_high = any(
        heldout[(c_lo, c_hi, l)]["logreg"] > 0.9
        for (c_lo, c_hi) in args.held_out_pairs
        for l in pen_hidden
    )
    emit(f"- penalized-layer DoM at chance (<0.6): {dom_at_chance}")
    emit(f"- max ridge R^2 on penalized layers: {ridge_hidden_max:+.4f}")
    emit(f"- any penalized-layer logreg > 0.9 at {{1,2}}: {logreg_high}")
    emit(f"- any held-out-pair logreg > 0.9 on penalized layers: {heldout_high}")
    if dom_at_chance and ridge_hidden_max > 0.5:
        verdict = "HIDDEN (point-hiding): DoM cancelled at {1,2} but c still linearly present across [1,2]."
    elif dom_at_chance and (logreg_high or heldout_high):
        verdict = "SPECIFIC-PROBE DEFEATED: DoM at chance but a stronger linear probe still separates."
    elif dom_at_chance and ridge_hidden_max < 0.1 and not (logreg_high or heldout_high):
        verdict = (
            "ERASED: no linear probe recovers c on penalized layers anywhere in [1,2]."
        )
    else:
        verdict = (
            "MIXED / penalty not yet biting — inspect traces and per-layer numbers."
        )
    emit(f"=> {verdict}")
    emit()

    # --- write report + plots ---
    out_log = log_dir(args.tag)
    os.makedirs(out_log, exist_ok=True)
    report_path = os.path.join(out_log, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {report_path}")

    out_dir = "plot"
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_log, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_traces(args.tag, history, hidden_layers, out_dir)
    plot_probe_gap(args.tag, hidden_layers, gap, out_dir)
    plot_heldout_r2(args.tag, all_layers, r2_adv, r2_base, out_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main(args)
