"""
Simple utility to check for dead ReLUs.

Copied from Claude transcript:

Method, so you can reuse it on other checkpoints:
1. Load the model and its ResidualMLPConfig (d_mlp, leaky_relu_slope, layer_norm).
2. Draw a large batch from data.sample_batch (the actual training input distribution — dead-ReLU checks are only meaningful relative to the data the model actually sees).
3. Manually replay forward() block-by-block, but stop before the nonlinearity: preact = layer_norm(r) @ W_in + b_in.
4. Per neuron, compute the fraction of samples with preact <= 0. A neuron with that fraction == 1.0 is dead — for plain ReLU (leaky_relu_slope=0.0, which this run uses), such a neuron outputs exactly 0 for every input in your distribution, so both ∂loss/∂W_in[:,j] and ∂loss/∂b_in[j] vanish and it can never recover via gradient descent (it can only escape if W_out is later reused, which doesn't happen since MLP grads are also zero).
5. Cross-check with std(activation) < eps across the batch — a second symptom (this can also catch "almost dead" neurons whose fraction is like 0.999 rather than exactly 1.0).

Since leaky_relu_slope = 0.0 in this run, dead neurons here are a genuine dead-end (zero gradient in the off-region) rather than a slow-leaking one. If you switch to a nonzero leaky_relu_slope, this same script still works but "dead" becomes a matter of degree (gradient is small but nonzero for negative inputs), so I'd switch the diagnostic to frac-nonpositive percentiles rather than a hard 100% cutoff.

usage: python check_dead_relu.py <checkpoint.pt> [--n SAMPLES] [--dead-thresh FRAC])
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--n", type=int, default=100_000, help="number of samples")
    p.add_argument(
        "--dead-thresh",
        type=float,
        default=1.0,
        help="fraction of samples with preact<=0 above which a neuron counts as dead",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import sys

import torch
from model import ResidualMLP
from data import sample_batch

torch.manual_seed(0)

model, ck = ResidualMLP.load(args.checkpoint, map_location="cpu")
model.eval()
cfg = model.config
print(
    f"config: num_x={cfg.num_x} d_model={cfg.d_model} d_mlp={cfg.d_mlp} "
    f"num_blocks={cfg.num_blocks} leaky_relu_slope={cfg.leaky_relu_slope} "
    f"layer_norm={cfg.layer_norm}"
)
if "iter" in ck:
    print(f"checkpoint iter: {ck['iter']}")

x_full, y = sample_batch(args.n, cfg.num_x)

with torch.no_grad():
    r = x_full @ model.W_E
    for i, block in enumerate(model.blocks):
        r_in = block.layer_norm(r) if block.layer_norm is not None else r
        preact = r_in @ block.W_in + block.b_in  # (N, d_mlp)
        frac_nonpositive = (preact <= 0).float().mean(dim=0)  # per-neuron, over samples
        dead_mask = frac_nonpositive >= args.dead_thresh
        n_dead = int(dead_mask.sum().item())

        act = torch.nn.functional.leaky_relu(
            preact, negative_slope=cfg.leaky_relu_slope
        )
        mean_act = act.mean(dim=0)
        std_act = act.std(dim=0)

        print(
            f"\nblock {i}: {n_dead}/{cfg.d_mlp} neurons dead "
            f"(preact<=0 on >={args.dead_thresh:.0%} of {args.n} samples)"
        )
        if n_dead:
            print(f"  dead neuron indices: {dead_mask.nonzero().flatten().tolist()}")
        print(
            f"  frac-nonpositive per neuron: min={frac_nonpositive.min():.3f} "
            f"median={frac_nonpositive.median():.3f} max={frac_nonpositive.max():.3f}"
        )
        print(
            f"  mean|act| across neurons: {mean_act.abs().mean():.4g}, "
            f"neurons with std(act)<1e-6: {int((std_act < 1e-6).sum())}/{cfg.d_mlp}"
        )

        r = r + (act @ block.W_out + block.b_out)
