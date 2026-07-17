"""Step 1 plots: training dynamics and learned y(x) curves.

Writes to plot/:
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--ckpt", type=str, default="best", choices=["best", "last"])
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def plot_dynamics(tag, out_dir):
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
    path = os.path.join(out_dir, f"{tag}_dynamics.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")


@torch.no_grad()
def plot_curves(tag, ckpt, out_dir):
    path = os.path.join(ckpt_dir(tag), f"{ckpt}.pt")
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    num_x = cfg["num_x"]
    model = ResidualMLP(
        num_x,
        cfg["d_model"],
        cfg["d_mlp"],
        leaky_relu_slope=cfg.get("leaky_relu_slope", 0.0),
    )
    model.load_state_dict(ck["model"])
    model.eval()

    c_values = [1.0, 1.333, 1.667, 2.0]
    xs = torch.linspace(-3, 3, 400)
    fig, axes = plt.subplots(
        1, len(c_values), figsize=(4 * len(c_values), 4), sharey=True
    )
    if len(c_values) == 1:
        axes = [axes]
    for ax, c in zip(axes, c_values):
        # Build inputs: sweep x on every coordinate simultaneously is not valid
        # (coords are independent), so sweep coordinate 0 and hold others at 0.
        for j in range(num_x):
            x = torch.zeros(len(xs), num_x)
            x[:, j] = xs
            x_full = torch.cat([x, torch.full((len(xs), 1), c)], dim=1)
            y = model.task_output(x_full)[:, j]
            ax.plot(xs.numpy(), y.numpy(), alpha=0.5, zorder=5)
        ax.plot(
            xs.numpy(),
            torch.clamp(xs, -c, c).numpy(),
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
    p = os.path.join(out_dir, f"{tag}_curves.png")
    fig.savefig(p, dpi=120)
    print(f"[plot] wrote {p}")


def main():
    args = parse_args()
    out_dir = "plot"
    os.makedirs(out_dir, exist_ok=True)
    plot_dynamics(args.tag, out_dir)
    plot_curves(args.tag, args.ckpt, out_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
