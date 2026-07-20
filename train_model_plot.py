"""Step 1 plots: training dynamics and learned y(x) curves.

Writes to plot/<tag>/:
  - <tag>_dynamics.png : iteration vs loss, iteration vs max-abs-error
  - <tag>_curves.png   : y vs x for fixed c in [1, 1.333, 1.667, 2], num_x lines each

Usage:
    python train_plot.py --tag nx1 --ckpt best
"""

import argparse
import json
import os

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import ResidualMLP
from paths import ckpt_dir, log_dir
from paths import plot_dir as get_plot_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--ckpt", type=str, default="best", choices=["best", "last"])
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def plot_dynamics(tag, plot_dir):
    hist_path = os.path.join(log_dir(tag), "history.json")
    with open(hist_path) as f:
        hist = json.load(f)
    its = [h[0] for h in hist]
    loss = [h[1] for h in hist]
    err = [h[2] for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogy(its, loss)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("MSE loss")
    ax1.set_title("loss")
    ax1.grid(True, which="both", alpha=0.3)
    ax2.semilogy(its, err)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("max abs elementwise error")
    ax2.set_title("max abs error")
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"training dynamics ({tag})")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{tag}_dynamics.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")


@torch.no_grad()
def plot_learned_curves(
    model,
    tag,
    plot_dir,
    c_values=(1.0, 1.333, 1.667, 2.0),
):
    """Plot learned y(x) per coordinate at fixed c, for an already-loaded model.

    Split out of plot_curves so callers who already have a model in memory
    (e.g. adversarial_report.py) can reuse it without a checkpoint round-trip.
    """
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
    p = os.path.join(plot_dir, f"{tag}_curves.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {p}")
    return p


def plot_curves(tag, ckpt, plot_dir):
    path = os.path.join(ckpt_dir(tag), f"{ckpt}.pt")
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    model = ResidualMLP(
        cfg["num_x"],
        cfg["d_model"],
        cfg["d_mlp"],
        leaky_relu_slope=cfg.get("leaky_relu_slope", 0.0),
        num_blocks=cfg.get("num_blocks", 4),  # 4 = pre-num_blocks-config default
    )
    model.load_state_dict(ck["model"])
    model.eval()
    plot_learned_curves(model, tag, plot_dir)


def main():
    args = parse_args()
    plot_dir = get_plot_dir(args.tag)
    os.makedirs(plot_dir, exist_ok=True)
    plot_dynamics(args.tag, plot_dir)
    plot_curves(args.tag, args.ckpt, plot_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
