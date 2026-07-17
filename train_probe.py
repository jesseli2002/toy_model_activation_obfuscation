"""Step 2: probe the residual stream for the hidden scalar c.

Loads a trained checkpoint (Step 1), captures the residual stream at one or more
layers (--layers, concatenated if more than one) on datasets pinned at c=1 and
c=2 (x resampled each time), and fits two probe types:
    - raw difference-of-means (DoM): direction = mean(r1|c=2) - mean(r1|c=1),
      classified by comparing projection to the train-set midpoint.
    - logistic regression (binary c=1 vs c=2)

(LDA and a continuous-c ridge probe were considered but dropped for Step 2: LDA
is redundant with logreg here -- both are linear classifiers on the same two
point-masses, and LDA's Gaussian-class-conditional assumption doesn't hold for
ReLU-transformed activations anyway -- and ridge-on-continuous-c tests a
different, Step-3-relevant question (is c decodable in between the two probed
points, needed to distinguish "erased" from "hidden" once the adversary is
pinning DoM at {1,2}) that's premature before Step 3 exists. See
plans/detailed_plan.md pitfall #4.)

Gate 2 thresholds (calibrated against the ideal analytic construction, see
plans/detailed_plan.md -- do NOT expect raw DoM to be near-perfect, ~0.77 is
correct behavior since ~67% of the DoM signal at r_1 lives in the x-directions):
    raw DoM      > 0.70   (reported as a diagnostic, not gated on)
    logreg       > 0.99   (expected ~1.000; this is what Gate 2 keys on)

Usage:
    python train_probe.py --tag nx32 --ckpt best
    python train_probe.py --tag nx32 --ckpt best --layers 0,2  # concat r_0 and r_2
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch

from data import sample_fixed_c
from model import ResidualMLP
from paths import ckpt_dir


def parse_layers(s: str) -> list[int]:
    return [int(v) for v in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="nx32")
    p.add_argument("--ckpt", type=str, default="best", choices=["best", "last"])
    p.add_argument(
        "--layers",
        type=parse_layers,
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
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_model(tag: str, ckpt: str, device: str) -> tuple[ResidualMLP, dict]:
    path = os.path.join(ckpt_dir(tag), f"{ckpt}.pt")
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = ResidualMLP(
        cfg["num_x"],
        cfg["d_model"],
        cfg["d_mlp"],
        leaky_relu_slope=cfg.get("leaky_relu_slope", 0.0),
        num_blocks=cfg.get("num_blocks", 4),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def capture_layers(
    model: ResidualMLP, x_full: torch.Tensor, layers: list[int]
) -> torch.Tensor:
    _, caches = model.forward(x_full, return_cache=True)
    return torch.cat([caches[i] for i in layers], dim=-1)


@torch.no_grad()
def binary_dataset(model, num_x, n, c_lo, c_hi, layers, generator, device):
    xf_lo, _ = sample_fixed_c(n, num_x, c_lo, generator=generator, device=device)
    xf_hi, _ = sample_fixed_c(n, num_x, c_hi, generator=generator, device=device)
    r_lo = capture_layers(model, xf_lo, layers)
    r_hi = capture_layers(model, xf_hi, layers)
    return r_lo, r_hi


def plot_probe(tag, layers, w_dom, midpoint, logreg, X_test, y_test, out_dir):
    from sklearn.decomposition import PCA

    proj_dom = X_test @ w_dom - midpoint
    proj_logreg = logreg.decision_function(X_test)
    pca_xy = PCA(n_components=2).fit_transform(X_test)

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0

    fig, (ax_dom, ax_logreg, ax_pca) = plt.subplots(1, 3, figsize=(15, 4))

    for ax, proj, title in (
        (ax_dom, proj_dom, "DoM projection"),
        (ax_logreg, proj_logreg, "logreg decision function"),
    ):
        ax.hist(proj[lo_mask], bins=60, alpha=0.6, label="c=1")
        ax.hist(proj[hi_mask], bins=60, alpha=0.6, label="c=2")
        ax.axvline(0.0, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(title)
        ax.set_xlabel("projection (test set)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_pca.scatter(
        pca_xy[lo_mask, 0], pca_xy[lo_mask, 1], s=4, alpha=0.4, label="c=1"
    )
    ax_pca.scatter(
        pca_xy[hi_mask, 0], pca_xy[hi_mask, 1], s=4, alpha=0.4, label="c=2"
    )
    ax_pca.set_title("PCA (top 2 components)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(fontsize=8)
    ax_pca.grid(True, alpha=0.3)

    layer_str = "-".join(str(i) for i in layers)
    fig.suptitle(f"probe separation ({tag}, layers={layer_str})")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}_L{layer_str}_probe.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")


def main():
    args = parse_args()
    from sklearn.linear_model import LogisticRegression

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.tag, args.ckpt, device)
    num_x = cfg["num_x"]
    num_blocks = cfg.get("num_blocks", 4)
    for layer in args.layers:
        assert 0 <= layer <= num_blocks, (
            f"--layers index {layer} out of range [0, {num_blocks}] "
            f"(0=embedding, i=after block i-1)"
        )

    g_train = torch.Generator(device=device).manual_seed(args.seed)
    g_test = torch.Generator(device=device).manual_seed(args.seed + 1)

    r_lo_tr, r_hi_tr = binary_dataset(  # training sets
        model, num_x, args.n_train, 1.0, 2.0, args.layers, g_train, device
    )
    r_lo_te, r_hi_te = binary_dataset(  # test sets
        model, num_x, args.n_test, 1.0, 2.0, args.layers, g_test, device
    )

    X_train = torch.cat([r_lo_tr, r_hi_tr], dim=0).cpu().numpy()
    y_train = torch.cat([torch.zeros(args.n_train), torch.ones(args.n_train)]).numpy()
    X_test = torch.cat([r_lo_te, r_hi_te], dim=0).cpu().numpy()
    y_test = torch.cat([torch.zeros(args.n_test), torch.ones(args.n_test)]).numpy()

    # --- raw difference-of-means ---
    mu_lo = r_lo_tr.mean(dim=0)
    mu_hi = r_hi_tr.mean(dim=0)
    w_dom = (mu_hi - mu_lo).cpu().numpy()
    midpoint = float(((mu_hi + mu_lo) / 2).cpu().numpy() @ w_dom)
    proj_test = X_test @ w_dom
    pred_dom = (proj_test > midpoint).astype(float)
    dom_acc = float((pred_dom == y_test).mean())

    # --- logistic regression ---
    logreg = LogisticRegression(max_iter=2000)
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

    out_dir = "plot"
    os.makedirs(out_dir, exist_ok=True)
    plot_probe(
        args.tag, args.layers, w_dom, midpoint, logreg, X_test, y_test, out_dir
    )
    if args.show:
        plt.show()

    return all_pass_key


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
