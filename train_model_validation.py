"""Gate 1: confirm the trained model reproduces sat(x, -c, c).

Loads a checkpoint and reports the maximum absolute elementwise error over a large
fresh eval set. Small error => Gate 1 PASS. Plateau far from zero => FAIL (stop and
report; do not tune indefinitely).

Usage:
    python train_validation.py --tag nx1 --ckpt best
"""

import argparse
import os

import torch

import config
from data import sample_batch
from model import ResidualMLP
from paths import ckpt_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--ckpt", type=str, default="best", choices=["best", "last"])
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--batch", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument(
        "--gate", type=float, default=1e-2, help="max-abs-error threshold for a PASS"
    )
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(ckpt_dir(args.tag), f"{args.ckpt}.pt")
    model, ck = ResidualMLP.load(path, map_location=device)
    model = model.to(device)
    model.eval()
    num_x = model.num_x

    g = torch.Generator(device=device).manual_seed(args.seed)
    worst = 0.0
    sse = 0.0
    done = 0
    while done < args.n:
        b = min(args.batch, args.n - done)
        x_full, y = sample_batch(b, num_x, generator=g, device=device)
        pred = model.task_output(x_full)
        err = (pred - y).abs()
        worst = max(worst, err.max().item())
        sse += (err**2).sum().item()
        done += b
    mse = sse / (done * num_x)

    status = "PASS" if worst < args.gate else "FAIL"
    print(
        f"[Gate 1] tag={args.tag} ckpt={args.ckpt} iter={ck['iter']} "
        f"num_x={num_x} d_model={model.d_model} d_mlp={model.d_mlp}"
    )
    print(
        f"[Gate 1] n={done}  MSE={mse:.3e}  max_abs_err={worst:.3e}  "
        f"threshold={args.gate:.1e}  -> {status}"
    )
    return worst < args.gate


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
