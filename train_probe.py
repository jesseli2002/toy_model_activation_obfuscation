"""Step 2: probe the residual stream for the hidden scalar c.

Loads a trained checkpoint (Step 1), captures the residual stream at one or more
layers (--layers, concatenated if more than one) on datasets pinned at c=1 and
c=2 (x resampled each time), and fits two probe types:
    - raw difference-of-means (DoM): direction = mean(r1|c=2) - mean(r1|c=1),
      classified by comparing projection to the train-set midpoint.
    - logistic regression (binary c=1 vs c=2)

Gate 2 (logreg accuracy) is the pass/fail signal that c is linearly decodable;
raw DoM is reported as a diagnostic, not gated on. See plans/detailed_plan.md
for the rationale behind the probe choices and gate thresholds.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float

from data import sample_fixed_c
from model import ResidualMLP
from paths import ckpt_dir
from paths import plot_dir as get_plot_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def _parse_layers(s: str) -> list[int]:
    return [int(v) for v in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="nx32")
    p.add_argument("--ckpt", type=str, default="best", choices=["best", "last"])
    p.add_argument(
        "--layers",
        type=_parse_layers,
        default=[1],
        help=(
            "comma-separated residual-stream indices to probe, e.g. '1' or '0,2' "
            "(0 = embedding, i = after block i-1; concatenated if more than one)"
        ),
    )
    p.add_argument("--n-train", type=int, default=20_000, help="per class/set")
    p.add_argument("--n-test", type=int, default=50_000, help="per class/set")
    p.add_argument("--seed", type=int, default=20260717)
    p.add_argument("--dom-gate", type=float, default=0.70)
    p.add_argument("--logreg-gate", type=float, default=0.99)
    p.add_argument(
        "--steer",
        action="store_true",
        help=(
            "causal test: add the DoM(c=1->c=2) direction at the probed layer to "
            "c=1 inputs and plot the resulting y(x), vs c=1/c=2 targets. Requires "
            "a single --layers entry (the injection point)."
        ),
    )
    p.add_argument(
        "--steer-scale",
        type=float,
        default=1.0,
        help="multiple of the full DoM(c=1->c=2) shift to inject (1.0 = full)",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_model(tag: str, ckpt: str, device: str) -> tuple[ResidualMLP, dict]:
    """Returns (model, full checkpoint dict) -- the dict carries any non-
    architecture fields (opt state, iter, adversarial-run metadata, ...) that
    rode along in the checkpoint; architecture lives on model.config."""
    path = os.path.join(ckpt_dir(tag), f"{ckpt}.pt")
    model, ck = ResidualMLP.load(path, map_location=device)
    model = model.to(device)
    model.eval()
    return model, ck


@torch.no_grad()
def capture_layers_dict(
    model: ResidualMLP, x_full: torch.Tensor, layers: list[int]
) -> dict[int, torch.Tensor]:
    """Returns each requested layer's residual-stream activations, keyed by
    layer index, from a single shared forward pass."""
    _, caches = model.forward(x_full, return_cache=True)
    return {i: caches[i] for i in layers}


@torch.no_grad()
def capture_layers(
    model: ResidualMLP, x_full: torch.Tensor, layers: list[int]
) -> torch.Tensor:
    """Like capture_layers_dict, but concatenates the requested layers into a
    single flat feature tensor."""
    d = capture_layers_dict(model, x_full, layers)
    return torch.cat([d[i] for i in layers], dim=-1)


@torch.no_grad()
def binary_dataset(model, num_x, n, c_lo, c_hi, layers, generator, device):
    xf_lo, _ = sample_fixed_c(n, num_x, c_lo, generator=generator, device=device)
    xf_hi, _ = sample_fixed_c(n, num_x, c_hi, generator=generator, device=device)
    r_lo = capture_layers(model, xf_lo, layers)
    r_hi = capture_layers(model, xf_hi, layers)
    return r_lo, r_hi


@torch.no_grad()
def _forward_steered(
    model: ResidualMLP,
    x_full: torch.Tensor,
    steer_layer: int,
    steer_vec: torch.Tensor | None,
) -> torch.Tensor:
    """Manual replay of ResidualMLP.forward, injecting steer_vec into the
    residual stream at index steer_layer (0=embedding, i=after block i-1)."""
    r = x_full @ model.W_E
    if steer_layer == 0 and steer_vec is not None:
        r = r + steer_vec
    for i, block in enumerate(model.blocks):
        r = r + block(r)
        if (i + 1) == steer_layer and steer_vec is not None:
            r = r + steer_vec
    y = r @ model.W_U
    return y[:, : model.num_x]


@torch.no_grad()
def _plot_steering(model, num_x, steer_layer, steer_vec, tag, plot_dir):
    xs = torch.linspace(-3, 3, 400, device=device)
    panels = [
        ("c=1, unsteered", 1.0, None, [1.0]),
        (
            f"c=1, steered @ layer {steer_layer} toward c=2",
            1.0,
            steer_vec,
            [1.0, 2.0],
        ),
        ("c=2, unsteered (reference)", 2.0, None, [2.0]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True)
    for ax, (title, c_val, vec, targets) in zip(axes, panels):
        for j in range(num_x):
            x = torch.zeros(len(xs), num_x)
            x[:, j] = xs
            x_full = torch.cat([x, torch.full((len(xs), 1), c_val)], dim=1)
            y = _forward_steered(model, x_full, steer_layer, vec)[:, j]
            ax.plot(
                xs.cpu().numpy(),
                y.cpu().numpy(),
                color="steelblue",
                alpha=0.3,
                zorder=5,
            )
        styles = {1.0: ("k--", "target sat(x,1)"), 2.0: ("r--", "target sat(x,2)")}
        for t in targets:
            ls, label = styles[t]
            ax.plot(
                xs.cpu().numpy(),
                torch.clamp(xs, -t, t).cpu().numpy(),
                ls,
                lw=1.5,
                label=label,
                zorder=2,
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("y")
    fig.suptitle(
        f"causal steering test ({tag}); {num_x} lines/panel, "
        f"steer @ layer {steer_layer}"
    )
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{tag}_L{steer_layer}_steer.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")


def plot_probe(
    tag,
    layers,
    w_dom: Float[np.ndarray, "d"],
    midpoint,
    logreg,
    X_test: Float[np.ndarray, "n d"],
    y_test: Bool[np.ndarray, "n"],
    plot_dir,
):
    """n = test-set size (both classes concatenated), d = probed feature dim
    (sum of d_model over --layers)."""
    from sklearn.decomposition import PCA

    proj_dom = X_test @ w_dom - midpoint
    proj_logreg = logreg.decision_function(X_test)
    pca_xy = PCA(n_components=2).fit_transform(X_test)

    # logreg's decision boundary direction in raw (unstandardized) feature
    # space, so we can project it out and PCA the residual: this extends the
    # logreg histogram into a second axis showing whether any of the
    # remaining (logreg-orthogonal) variance still separates the classes.
    scaler = logreg.named_steps["standardscaler"]
    clf = logreg.named_steps["logisticregression"]
    w_logreg: Float[np.ndarray, "d"] = clf.coef_[0] / scaler.scale_
    w_hat: Float[np.ndarray, "d"] = w_logreg / np.linalg.norm(w_logreg)
    X_resid: Float[np.ndarray, "n d"] = X_test - np.outer(X_test @ w_hat, w_hat)
    pc1_resid: Float[np.ndarray, "n"] = PCA(n_components=1).fit_transform(X_resid)[:, 0]

    # raw (unstandardized) projection onto the logreg direction, for the
    # scatter plot's x-axis: decision_function() reports values in the
    # StandardScaler's space, whereas w_hat @ X_test is in data coordinates.
    proj_logreg_raw: Float[np.ndarray, "n"] = X_test @ w_hat
    logreg_raw_threshold = float(
        (np.dot(scaler.mean_, w_logreg) - clf.intercept_[0]) / np.linalg.norm(w_logreg)
    )

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0

    fig, (ax_dom, ax_logreg, ax_pca, ax_logreg_resid) = plt.subplots(
        1, 4, figsize=(20, 6)
    )

    for ax, proj, title in (
        (ax_dom, proj_dom, "DoM projection"),
        (ax_logreg, proj_logreg, "logreg decision function"),
    ):
        ax.hist(proj[lo_mask], bins=60, alpha=0.6, label="c=1")
        ax.hist(proj[hi_mask], bins=60, alpha=0.6, label="c=2")

        # Set plot limit to avoid outliers
        percentile_5, percentile_95 = np.percentile(proj, [5, 95])
        percentile_diff = percentile_95 - percentile_5
        ax.set_xlim(
            [
                percentile_5 - percentile_diff * 0.5,
                percentile_95 + percentile_diff * 0.5,
            ]
        )

        ax.axvline(0.0, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(title)
        ax.set_xlabel("projection (test set)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_pca.scatter(pca_xy[lo_mask, 0], pca_xy[lo_mask, 1], s=4, alpha=0.4, label="c=1")
    ax_pca.scatter(pca_xy[hi_mask, 0], pca_xy[hi_mask, 1], s=4, alpha=0.4, label="c=2")
    ax_pca.set_title("PCA (top 2 components)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(fontsize=8)
    ax_pca.grid(True, alpha=0.3)
    ax_pca.set_aspect("equal", adjustable="datalim")

    ax_logreg_resid.scatter(
        proj_logreg_raw[lo_mask], pc1_resid[lo_mask], s=4, alpha=0.4, label="c=1"
    )
    ax_logreg_resid.scatter(
        proj_logreg_raw[hi_mask], pc1_resid[hi_mask], s=4, alpha=0.4, label="c=2"
    )
    ax_logreg_resid.axvline(
        logreg_raw_threshold, color="k", ls="--", lw=1, label="threshold"
    )
    ax_logreg_resid.set_title("logreg vs residual PCA")
    ax_logreg_resid.set_xlabel("logreg projection (data coords)")
    ax_logreg_resid.set_ylabel("PC1 of logreg-orthogonal residual")
    ax_logreg_resid.legend(fontsize=8)
    ax_logreg_resid.grid(True, alpha=0.3)
    # ax_logreg_resid.set_aspect("equal", adjustable="datalim")

    layer_str = "-".join(str(i) for i in layers)
    fig.suptitle(f"probe separation ({tag}, layers={layer_str})")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{tag}_L{layer_str}_probe.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")


def main():
    args = parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model, ck = load_model(args.tag, args.ckpt, device)
    num_x = model.num_x
    num_blocks = model.num_blocks
    for layer in args.layers:
        assert 0 <= layer <= num_blocks, (
            f"--layers index {layer} out of range [0, {num_blocks}] "
            f"(0=embedding, i=after block i-1)"
        )

    g = torch.Generator(device=device).manual_seed(args.seed)

    r_lo_tr, r_hi_tr = binary_dataset(  # training sets
        model, num_x, args.n_train, 1.0, 2.0, args.layers, g, device
    )
    r_lo_te, r_hi_te = binary_dataset(  # test sets
        model, num_x, args.n_test, 1.0, 2.0, args.layers, g, device
    )

    X_train = torch.cat([r_lo_tr, r_hi_tr], dim=0).cpu().numpy()
    y_train = (
        torch.cat([torch.zeros(args.n_train), torch.ones(args.n_train)]).cpu().numpy()
    )
    X_test = torch.cat([r_lo_te, r_hi_te], dim=0).cpu().numpy()
    y_test = (
        torch.cat([torch.zeros(args.n_test), torch.ones(args.n_test)]).cpu().numpy()
    )

    # --- raw difference-of-means ---
    mu_lo = r_lo_tr.mean(dim=0)
    mu_hi = r_hi_tr.mean(dim=0)
    w_dom = (mu_hi - mu_lo).cpu().numpy()
    midpoint = float(((mu_hi + mu_lo) / 2).cpu().numpy() @ w_dom)
    proj_test = X_test @ w_dom
    pred_dom = (proj_test > midpoint).astype(float)
    dom_acc = float((pred_dom == y_test).mean())

    # --- logistic regression (DoM above needs no normalization: it's just a
    # difference of means, invariant to a shared affine rescaling of features) ---
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logreg.fit(X_train, y_train)
    logreg_acc = float(logreg.score(X_test, y_test))

    print(
        f"[Gate 2] tag={args.tag} ckpt={args.ckpt} num_x={num_x} layers={args.layers} "
        f"n_train={args.n_train}/class n_test={args.n_test}/class"
    )
    results = [
        ("raw DoM", dom_acc, args.dom_gate),
        ("logreg", logreg_acc, args.logreg_gate),
    ]
    for name, val, gate in results:
        status = "PASS" if val > gate else "FAIL"
        print(f"[Gate 2] {name:>10s} = {val:.6f}  gate > {gate:.2f}  -> {status}")
    all_pass_key = logreg_acc > args.logreg_gate

    print(
        f"[Gate 2] overall (keyed on logreg, DoM reported as diagnostic) -> "
        f"{'PASS' if all_pass_key else 'FAIL'}"
    )

    plot_dir = get_plot_dir(args.tag)
    os.makedirs(plot_dir, exist_ok=True)
    plot_probe(args.tag, args.layers, w_dom, midpoint, logreg, X_test, y_test, plot_dir)

    if args.steer:
        assert len(args.layers) == 1, "--steer needs a single --layers entry"
        steer_vec = args.steer_scale * torch.tensor(
            w_dom, dtype=r_lo_tr.dtype, device=device
        )
        _plot_steering(model, num_x, args.layers[0], steer_vec, args.tag, plot_dir)

    if args.show:
        plt.show()

    return all_pass_key


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
