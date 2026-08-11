"""Train the saturation task with c withheld from the model, to empirically
approach the c-blind loss floor.

Purpose: `analytic.no_c_task_loss` derives, in closed form, the lowest
per-coordinate L_task any predictor of x alone can reach. That makes it a
one-sided certificate -- an adversarially-trained model scoring below it must
be reading c, no matter what its activations look like. This script checks the
floor is actually reachable by this architecture under gradient descent, so a
model that fails to beat it can't be excused as "the optimizer just couldn't
get there".

c is withheld by zeroing its input coordinate (the embedding is [I; 0], so
that removes it from the residual stream entirely) while the target still
uses the true c. Residual-stream noise is on, unlike the analytic derivation:
noise can only raise the achievable loss, so the bound stays one-sided.

There is no probe and no adversarial penalty here -- pure task training. The
output is the per-iteration loss log plus a periodic `last.pt` checkpoint
(same on-disk layout as `train_adversarial_logreg.py`'s minus the
adversarial-only fields, so `adversarial_report.py --tag <tag>` still loads
it); there's no pass/fail check.

Deliberately standalone rather than a mode of train_adversarial_logreg.py:
every hyperparameter besides the architecture is a module-level constant
below, so this file needs no edit when the real training config gains fields.
Constants are set to values representative of a real adversarial run, not
tuned.
"""

import argparse
import json
import os

# --- hyperparameters (edit here; deliberately not configurable, see docstring) ---
SEED = 10000
BATCH_SIZE = 4096 * 4
LR = 3e-3
ADAM_EPS = 1e-5
ADAM_BETA1 = 0.7
ADAM_BETA2 = 0.97
GRAD_CLIP = 1.0  # per-block grad-norm clip; 0 disables
RESID_NOISE_STD = 0.1
STABLEADAMW_D = 1
# Architecture knobs other than the four CLI dimensions.
ACTIVATION = "gelu"
LEAKY_RELU_SLOPE = 0.0
OUT_INIT_SCALE = 0.1
LAYER_NORM = False
# x-sampling bias (see data.sample_batch); None = plain uniform.
X_P_OUTER = None
X_THRESHOLD = 1.0
# Low-variance eval of the same loss, alongside the noisy per-iteration one.
EVAL_BATCH = 4096 * 4
EVAL_BATCHES = 32


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the saturation task WITHOUT c as a model input, to "
        "check the analytic c-blind loss floor is reachable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-x", type=int, required=True)
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--d-mlp", type=int, required=True)
    p.add_argument("--num-blocks", type=int, required=True)
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    p.add_argument("--tag", type=str, default="no-c")
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import torch
from jaxtyping import Float
from torch import Tensor

import config
from analytic import no_c_task_loss
from data import sample_batch
from model import ResidualMLP
from paths import ckpt_dir, log_dir

from stableadamw import StableAdamW


def blind_c(
    x_full: Float[Tensor, "batch num_x_plus_1"],
) -> Float[Tensor, "batch num_x_plus_1"]:
    x_blind = x_full.clone()
    x_blind[:, -1] = 0.0
    return x_blind


def task_loss(
    model: ResidualMLP,
    x_full: Float[Tensor, "batch num_x_plus_1"],
    y: Float[Tensor, "batch num_x"],
    noise: Float[Tensor, "num_blocks_minus_1 batch d_model"],
) -> Float[Tensor, ""]:
    """Per-coordinate MSE of the c-blind forward pass against the true target."""
    pred = model.task_output(blind_c(x_full), noise=noise)
    return torch.mean((pred - y) ** 2)


@torch.no_grad()
def eval_loss(model: ResidualMLP, gen: torch.Generator, device: str) -> float:
    """Same loss as training, averaged over EVAL_BATCHES fresh batches -- the
    per-iteration loss is too noisy to compare against the analytic floor
    directly."""
    total = 0.0
    for _ in range(EVAL_BATCHES):
        x_full, y = sample_batch(
            EVAL_BATCH,
            model.num_x,
            generator=gen,
            device=device,
            x_p_outer=X_P_OUTER,
            x_threshold=X_THRESHOLD,
        )
        noise = model.generate_noise(EVAL_BATCH, RESID_NOISE_STD, gen)
        total += task_loss(model, x_full, y, noise).item()
    return total / EVAL_BATCHES


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    gen = torch.Generator(device=device).manual_seed(SEED + 1)

    model = ResidualMLP(
        config.ResidualMLPConfig(
            num_x=args.num_x,
            d_model=args.d_model,
            d_mlp=args.d_mlp,
            num_blocks=args.num_blocks,
            out_init_scale=OUT_INIT_SCALE,
            activation=ACTIVATION,
            leaky_relu_slope=LEAKY_RELU_SLOPE,
            layer_norm=LAYER_NORM,
        )
    ).to(device)
    opt = StableAdamW(
        model.parameters(),
        lr=LR,
        eps=ADAM_EPS,
        betas=(ADAM_BETA1, ADAM_BETA2),
        d=STABLEADAMW_D,
    )

    bound = no_c_task_loss(
        config.X_LOW,
        config.X_HIGH,
        config.C_LOW,
        config.C_HIGH,
        X_P_OUTER,
        X_THRESHOLD,
    )
    print(f"[bound] analytic c-blind minimum L_task = {bound:.6e}")

    out_dir = log_dir(args.tag)
    os.makedirs(out_dir, exist_ok=True)
    # Same name/schema train_adversarial_logreg.py writes (minus the
    # adversarial-only fields, nan/None instead), so
    # adversarial_report.py's training-trace plot picks it up unmodified.
    hist_path = os.path.join(out_dir, "history.jsonl")
    out_ckpt_dir = ckpt_dir(args.tag)
    os.makedirs(out_ckpt_dir, exist_ok=True)
    last_path = os.path.join(out_ckpt_dir, "last.pt")
    print(f"[run] tag={args.tag} device={device} iters={args.max_iters}")

    best = float("inf")
    with open(hist_path, "w") as hist:
        for it in range(args.max_iters):
            x_full, y = sample_batch(
                BATCH_SIZE,
                args.num_x,
                generator=gen,
                device=device,
                x_p_outer=X_P_OUTER,
                x_threshold=X_THRESHOLD,
            )
            noise = model.generate_noise(BATCH_SIZE, RESID_NOISE_STD, gen)
            loss = task_loss(model, x_full, y, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP > 0:
                # Per-block, matching train_adversarial_logreg.py's clipping.
                for block in model.blocks:
                    torch.nn.utils.clip_grad_norm_(block.parameters(), GRAD_CLIP)
            opt.step()

            best = min(best, loss.item())
            if it % args.log_interval == 0 or it == args.max_iters - 1:
                ev_loss = eval_loss(model, gen, device)
                print(
                    f"iter {it:>6d}  loss {loss.item():.6e}  "
                    f"eval {ev_loss:.6e}  "
                    f"eval/bound {ev_loss / bound:.4f}"
                )
                # Same keys train_adversarial_logreg.py's history.jsonl
                # entries have (see TrainRecord/_history_entry) -- no probe
                # here, so l_probe/lam_eff/n_exploded are nan/None/0.
                entry = {
                    "iter": it,
                    "loss": loss.item(),
                    "l_task": ev_loss,
                    "l_probe": float("nan"),
                    "lam_eff": None,
                    "n_exploded": 0,
                }
                hist.write(json.dumps(entry) + "\n")

            if it % args.ckpt_interval == 0 and it > 0:
                model.save(last_path, iter=it, best_loss=best)

    model.save(last_path, iter=args.max_iters - 1, best_loss=best)
    print(f"[done] best per-iter loss {best:.6e}  vs bound {bound:.6e}")
    print(f"[done] loss log in {hist_path}, checkpoint in {last_path}")
    print(f"[next] python adversarial_report.py --tag {args.tag}")


if __name__ == "__main__":
    main(args)
