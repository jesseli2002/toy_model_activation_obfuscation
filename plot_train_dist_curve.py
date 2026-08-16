"""Publication plot: learned y(x) in- and out-of-distribution, for a single
fixed tag.

2x2 grid: top row samples x on the TRAINING distribution (x ~ U[X_LOW,
X_HIGH]^num_x, c pinned) and scatters (x, y_pred) pooled over every
coordinate -- see plot_train_dist_curves.tmp.py, which this supersedes.
Bottom row is probe_lib.plot_learned_curves' OOD sweep: one
coordinate swept at a time with the rest held at 0 (a corner no training
example actually occupies). Two versions are written: one with the OOD sweep
shown noise-free and at the checkpoint's own noise level superimposed (so the
noise's effect on the OOD extrapolation is visible directly), and a
noise-free-only variant. Titles are quieted for a writeup (no tag, no
per-panel MSE) -- see make_one_run_plots.py for the same convention.
"""

import argparse

TAG = "sweep11_lr0.0015_iter200k_lam0.01_tr0"
C_VALUES = (1.0, 2.0)

LEARNED_COLOR = "tab:blue"
NOISEFREE_ALPHA = 1
NOISY_ALPHA = 0.15
ID_SCATTER_ALPHA = 0.05


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", choices=["last", "best", "both"], default="last")
    p.add_argument(
        "--n-id",
        type=int,
        default=2000,
        help="in-distribution samples per c panel (x num_x coords/sample)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import os

import matplotlib.pyplot as plt
import torch

from checkpoint_lib import load_model, resolve_adv_config
from config import X_LOW, X_HIGH
from data import sample_fixed_c
from model import Noise
from paths import plot_dir as get_plot_dir
from probe_lib import save_plot

plt.rcParams["savefig.dpi"] = 200


@torch.no_grad()
def _plot_id_panel(
    ax, model, c: float, n: int, noise: Noise, g: torch.Generator, show_legend
):
    """Scatter (x, y_pred) pooled over every coordinate, for x sampled on the
    training distribution at fixed c."""
    num_x = model.num_x
    device = next(model.parameters()).device
    x_full, _ = sample_fixed_c(n, num_x, c, generator=g, device=device)
    pred = model.task_output(x_full, noise=noise, generator=g)
    ax.scatter(
        x_full[:, :num_x].flatten().cpu().numpy(),
        pred.flatten().cpu().numpy(),
        s=1,
        alpha=ID_SCATTER_ALPHA,
        color=LEARNED_COLOR,
        linewidths=0,
        rasterized=True,
        zorder=5,
    )
    xs = torch.linspace(X_LOW, X_HIGH, 400, device=device)
    ax.plot(
        xs.cpu().numpy(),
        torch.clamp(xs, -c, c).cpu().numpy(),
        color="red",
        ls="--",
        lw=1.5,
        zorder=10,
        label="target\nfunction" if show_legend else None,
    )
    ax.set_title(f"c = {c:g}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


@torch.no_grad()
def _plot_ood_panel(
    ax, model, c: float, noise: Noise, g: torch.Generator, show_noisy: bool = True
):
    """One coordinate swept at a time, others held at 0. If `show_noisy`, both
    the noise-free and the checkpoint's own noise level are superimposed in
    the same color; otherwise only the noise-free curve is shown."""
    num_x = model.num_x
    device = next(model.parameters()).device
    xs = torch.linspace(X_LOW, X_HIGH, 400, device=device)
    levels = [(0.0, NOISEFREE_ALPHA)]
    if show_noisy:
        levels.append((noise, NOISY_ALPHA))
    for noise_level, alpha in levels:
        for j in range(num_x):
            x = torch.zeros(len(xs), num_x, device=device)
            x[:, j] = xs
            x_full = torch.cat([x, torch.full((len(xs), 1), c, device=device)], dim=1)
            y = model.task_output(x_full, noise=noise_level, generator=g)[:, j]
            ax.plot(
                xs.cpu().numpy(),
                y.cpu().numpy(),
                color=LEARNED_COLOR,
                alpha=alpha,
                zorder=5,
                lw=0.5,
            )
    ax.plot(
        xs.cpu().numpy(),
        torch.clamp(xs, -c, c).cpu().numpy(),
        color="red",
        ls="--",
        lw=1.5,
        zorder=1,
    )
    ax.set_title(f"c = {c:g}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def _make_plot(
    model, noise: Noise, n_id: int, g: torch.Generator, show_noisy_ood: bool = True
):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    _plot_id_panel(axes[0][0], model, C_VALUES[0], n_id, noise, g, show_legend=True)
    _plot_id_panel(axes[0][1], model, C_VALUES[1], n_id, noise, g, show_legend=False)

    for ax, c in zip(axes[1], C_VALUES):
        _plot_ood_panel(ax, model, c, noise, g, show_noisy=show_noisy_ood)

    for ax in axes[-1]:
        ax.set_xlabel("x")
    axes[0, 0].set_ylabel("y\n(in-distribution)")
    axes[1, 0].set_ylabel("y\n(out-of-distribution)")

    fig.tight_layout()
    return fig


def _run_one(args, ckpt, device):
    model, ck = load_model(TAG, ckpt, device)
    adv_cfg = resolve_adv_config(ck)
    assert adv_cfg is not None, f"{TAG}/{ckpt} has no adv_config -- nothing to plot."
    plot_dir = get_plot_dir(TAG, ckpt)
    os.makedirs(plot_dir, exist_ok=True)

    g = torch.Generator(device=device).manual_seed(args.seed)
    fig = _make_plot(model, adv_cfg.resid_noise_std, args.n_id, g, show_noisy_ood=True)
    save_plot(fig, plot_dir, f"{TAG}_train_dist_curve.png", close=not args.show)
    save_plot(fig, plot_dir, f"{TAG}_train_dist_curve.svg", close=not args.show)

    g = torch.Generator(device=device).manual_seed(args.seed)
    fig = _make_plot(model, adv_cfg.resid_noise_std, args.n_id, g, show_noisy_ood=False)
    save_plot(
        fig, plot_dir, f"{TAG}_train_dist_curve_noisefree.png", close=not args.show
    )
    save_plot(
        fig, plot_dir, f"{TAG}_train_dist_curve_noisefree.svg", close=not args.show
    )


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = ["last", "best"] if args.ckpt == "both" else [args.ckpt]
    for ckpt in ckpts:
        _run_one(args, ckpt, device)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main(args)
