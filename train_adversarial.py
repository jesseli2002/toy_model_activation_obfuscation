"""Step 3 — adversarial training: model vs. difference-of-means (DoM) probe.

We add a penalty that pressures the model to make c=1 and c=2 activations share
the *same mean* at every hidden residual layer (defeating a difference-of-means
probe) while still computing sat(x,-c,c) correctly across the full c-range.

    L = lam * L_probe(pinned c in {1,2})  +  (1 - lam) * L_task(full-range c ~ U[1,2])

  lam=1 optimizes ENTIRELY for hiding c (task loss has zero weight) — useful for
  testing whether hiding is achievable *in principle* at a given set of layers
  before worrying about how much task fidelity it costs. lam=0 recovers plain
  task training with no probe pressure at all.

  L_task  — MSE over the first num_x outputs on a batch with c ~ U[1,2]. Keeping
            the task trained across ALL c is the single most important design
            choice: it stops "c not recoverable at held-out c" from being
            confounded with "the model was never trained there".
  L_probe — sum over the *hidden* residual layers of the squared L2 norm of the
            difference of class means, on a SEPARATE pinned sub-batch (half c=1,
            half c=2, x resampled):
                sum_{l in hidden} || mean(r_l | c=2) - mean(r_l | c=1) ||^2
            This *is* the DoM probe's separability; driving it to 0 makes the two
            class means coincide at every hidden layer -> DoM accuracy -> chance.
            Closed form, no inner probe-training loop.

"Hidden" layers are the residual-stream caches strictly between the embedding
(cache 0 = input; c sits verbatim in a fixed coordinate, nothing to hide/train)
and the final residual (cache num_blocks -> y; the task *requires* it to encode
c, since sat(x,-1,1) != sat(x,-2,2)). So hidden = caches 1 .. num_blocks-1.
Penalizing all of them by default closes the "relocate c to an unpenalized
layer" escape route from the start; per-layer logging still confirms it.

This is NOT gated. The deliverable is the trained checkpoint + diagnostics
(see adversarial_report.py); run once, then stop and review. The interesting
science is not "can it hide c" (expected: yes) but HOW: does it hide c only at
the probed points {1,2} (recoverable elsewhere in [1,2] -> "hidden"), or genuinely
erase linear c-information across the range ("erased")?

Usage:
    # primary run: warm-start a capable model, then apply probe pressure
    python train_adversarial.py --tag adv1 --lam 0.5 \
        --warmstart-path runs/nx32/checkpoints/best.pt --max-iters 6000
    # from scratch (conflates learning + hiding; kept for contrast)
    python train_adversarial.py --tag advscratch --init scratch --lam 0.5
    python train_adversarial.py --resume --tag adv1 --max-iters 12000
    # feasibility check: can c be hidden at layer 1 at all, ignoring task loss
    python train_adversarial.py --tag adv-feas-l1 --lam 1.0 --penalty-layers 1 \
        --warmstart-path runs/nx32/checkpoints/best.pt --max-iters 6000
"""

import argparse
import json
import math
import os
import shutil
import time

import config


def parse_penalty_layers(s: str) -> str | list[int]:
    if s.strip().lower() == "all":
        return "all"
    return [int(v) for v in s.split(",") if v.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 3 adversarial training (model vs. DoM probe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tag", type=str, default="adv")
    p.add_argument(
        "--init",
        choices=["warmstart", "scratch"],
        default="warmstart",
        help="warmstart (default) loads a capable model then applies probe "
        "pressure; scratch conflates learning the task with hiding c.",
    )
    p.add_argument(
        "--warmstart-path",
        type=str,
        default="runs/nx32/checkpoints/best.pt",
        help="checkpoint to warm-start from (only used when --init warmstart). "
        "Architecture is taken from this checkpoint's config.",
    )
    p.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="convex-combination weight: loss = lam * L_probe + (1-lam) * L_task. "
        "lam=1 optimizes purely for hiding c (task loss ignored) -- use this to "
        "test whether hiding is achievable in principle at a given set of "
        "layers. lam=0 is plain task training. Tune in between so the penalty "
        "bites without wrecking the task.",
    )
    p.add_argument(
        "--lam-warmup-iters",
        type=int,
        default=0,
        help="linearly ramp the penalty weight 0 -> lam over this many iters "
        "(task weight ramps 1 -> 1-lam correspondingly). Warm-starting from a "
        "near-exact solution and hitting it with the full penalty at once knocks "
        "the model off the task manifold into a bad basin it can't climb back "
        "from; ramping keeps the task intact while probe pressure grows in. "
        "0 = no ramp (constant lam).",
    )
    p.add_argument(
        "--penalty-layers",
        type=parse_penalty_layers,
        default="all",
        help="'all' = every hidden layer (1..num_blocks-1), or a comma-separated "
        "subset e.g. '1,2,3'.",
    )
    # Architecture (only used for --init scratch; warmstart reads the checkpoint).
    p.add_argument("--num-x", type=int, default=config.NUM_X)
    p.add_argument("--d-model", type=int, default=config.D_MODEL)
    p.add_argument("--d-mlp", type=int, default=None, help="default: num_x")
    p.add_argument("--num-blocks", type=int, default=config.NUM_BLOCKS)
    p.add_argument("--leaky-relu-slope", type=float, default=config.LEAKY_RELU_SLOPE)
    p.add_argument("--out-init-scale", type=float, default=0.1)
    # Optimization
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument(
        "--probe-batch-size",
        type=int,
        default=4096,
        help="per-class size of the pinned sub-batch used for L_probe.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final", type=float, default=1e-3)
    p.add_argument("--max-iters", type=int, default=6000)
    p.add_argument("--seed", type=int, default=913768)
    # Bookkeeping
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--tag-force",
        action="store_true",
        help="delete an existing runs/<tag> directory before a fresh run.",
    )
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import torch

from data import sample_batch, sample_fixed_c
from model import ResidualMLP
from paths import ckpt_dir, log_dir, run_dir
from train_model import eval_max_err


def cosine_lr(step: int, total: int, lr0: float, lr1: float) -> float:
    if total <= 1:
        return lr0
    t = min(step, total) / total
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * t))


def resolve_hidden_layers(penalty_layers, num_blocks: int) -> list[int]:
    """Hidden residual layers = 1 .. num_blocks-1 (see module docstring)."""
    all_hidden = list(range(1, num_blocks))
    if penalty_layers == "all":
        return all_hidden
    layers = sorted(set(penalty_layers))
    for lyr in layers:
        if lyr == 0:
            raise SystemExit(
                "[error] layer 0 is the embedding: c sits in a fixed coordinate, "
                "so its class-mean gap is a constant 1.0 with no gradient. "
                "Penalizing it is a no-op; drop it."
            )
        if lyr == num_blocks:
            print(
                f"[warn] layer {num_blocks} is the final residual (-> y). The task "
                f"REQUIRES it to encode c (sat differs by c), so penalizing it "
                f"fights the task directly. Proceeding as explicitly requested."
            )
        if not (0 <= lyr <= num_blocks):
            raise SystemExit(
                f"[error] penalty layer {lyr} out of range [0, {num_blocks}]."
            )
    return layers


def delta_means_from_x(model, x_lo, x_hi, layers):
    """Per-layer difference of class means for pre-sampled c=1 / c=2 batches.

    Returns {layer: mean(r_l|c=2) - mean(r_l|c=1)}. Differentiable; the caller
    controls grad via torch.no_grad(). Shared by the training penalty (fresh x
    each step) and the eval trace (fixed x, built once).
    """
    _, caches_lo = model.forward(x_lo, return_cache=True)
    _, caches_hi = model.forward(x_hi, return_cache=True)
    return {lyr: caches_hi[lyr].mean(0) - caches_lo[lyr].mean(0) for lyr in layers}


def probe_delta_means(model, num_x, n_per_class, layers, generator, device):
    """Differentiable per-layer difference of class means at pinned c in {1,2},
    with x resampled from `generator` each call."""
    xf_lo, _ = sample_fixed_c(n_per_class, num_x, 1.0, generator, device)
    xf_hi, _ = sample_fixed_c(n_per_class, num_x, 2.0, generator, device)
    return delta_means_from_x(model, xf_lo, xf_hi, layers)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(run_dir(args.tag)) and not args.resume:
        if args.tag_force:
            shutil.rmtree(run_dir(args.tag))
        else:
            raise SystemExit(
                f"[error] runs/{args.tag} already exists. Use --resume to continue, "
                f"--tag-force to overwrite, or pick a different --tag."
            )

    run_ckpt_dir = ckpt_dir(args.tag)
    run_log_dir = log_dir(args.tag)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # --- build / initialize the model ---
    if args.init == "warmstart":
        if not os.path.exists(args.warmstart_path):
            raise SystemExit(
                f"[error] --init warmstart but checkpoint not found: "
                f"{args.warmstart_path}"
            )
        ck = torch.load(args.warmstart_path, map_location=device)
        cfg = ck["config"]
        num_x = cfg["num_x"]
        d_model = cfg["d_model"]
        d_mlp = cfg["d_mlp"]
        num_blocks = cfg.get("num_blocks", 4)
        leaky = cfg.get("leaky_relu_slope", 0.0)
        model = ResidualMLP(
            num_x, d_model, d_mlp, leaky_relu_slope=leaky, num_blocks=num_blocks
        ).to(device)
        model.load_state_dict(ck["model"])
        print(f"[init] warm-started from {args.warmstart_path} (cfg={cfg})")
    else:
        num_x = args.num_x
        d_model = args.d_model
        d_mlp = args.d_mlp if args.d_mlp is not None else config.d_mlp_for(num_x)
        num_blocks = args.num_blocks
        leaky = args.leaky_relu_slope
        model = ResidualMLP(
            num_x,
            d_model,
            d_mlp,
            out_init_scale=args.out_init_scale,
            leaky_relu_slope=leaky,
            num_blocks=num_blocks,
        ).to(device)
        print(f"[init] scratch model num_x={num_x} d_model={d_model} d_mlp={d_mlp}")

    hidden_layers = resolve_hidden_layers(args.penalty_layers, num_blocks)
    if not hidden_layers:
        raise SystemExit(
            f"[error] no penalty layers (num_blocks={num_blocks} has no hidden "
            f"layers). Nothing to hide against."
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    start_iter = 0
    history = []  # list of dicts
    best_loss = float("inf")
    last_path = os.path.join(run_ckpt_dir, "last.pt")
    best_path = os.path.join(run_ckpt_dir, "best.pt")
    hist_path = os.path.join(run_log_dir, "history.json")

    if args.resume and os.path.exists(last_path):
        rck = torch.load(last_path, map_location=device)
        model.load_state_dict(rck["model"])
        opt.load_state_dict(rck["opt"])
        start_iter = rck["iter"]
        best_loss = rck.get("best_loss", float("inf"))
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        print(f"[resume] from iter {start_iter}, best_loss={best_loss:.3e}")

    def config_dict():
        return {
            "num_x": num_x,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "num_blocks": num_blocks,
            "leaky_relu_slope": leaky,
            "seed": args.seed,
            # adversarial metadata
            "lam": args.lam,
            "lam_warmup_iters": args.lam_warmup_iters,
            "penalty_layers": hidden_layers,
            "init": args.init,
            "warmstart_path": args.warmstart_path if args.init == "warmstart" else None,
        }

    def save(path, it):
        torch.save(
            {
                "iter": it,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_loss": best_loss,
                "config": config_dict(),
            },
            path,
        )

    # Fixed eval batch, drawn once from gen so the delta-mean trace reflects the
    # model changing, not the batch.
    eval_x_lo, _ = sample_fixed_c(20_000, num_x, 1.0, gen, device)
    eval_x_hi, _ = sample_fixed_c(20_000, num_x, 2.0, gen, device)

    @torch.no_grad()
    def eval_delta_norms():
        """Clean per-layer ||Δmean|| on the fixed eval batch (for stable traces)."""
        deltas = delta_means_from_x(model, eval_x_lo, eval_x_hi, hidden_layers)
        return {lyr: float(d.norm().item()) for lyr, d in deltas.items()}

    print(
        f"[adv] tag={args.tag} init={args.init} lam={args.lam} "
        f"penalty_layers={hidden_layers} num_blocks={num_blocks} "
        f"bs={args.batch_size} probe_bs={args.probe_batch_size}/class "
        f"lr={args.lr} device={device} iters {start_iter}->{args.max_iters}"
    )

    t0 = time.time()
    it = start_iter
    for it in range(start_iter, args.max_iters):
        lr = cosine_lr(it, args.max_iters, args.lr, args.lr_final)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # task loss on the FULL c-range
        x_full, y = sample_batch(args.batch_size, num_x, generator=gen, device=device)
        pred = model.task_output(x_full)
        l_task = torch.mean((pred - y) ** 2)

        # probe penalty on a separate pinned sub-batch (x resampled)
        deltas = probe_delta_means(
            model, num_x, args.probe_batch_size, hidden_layers, gen, device
        )
        l_probe = sum((d**2).sum() for d in deltas.values())

        if args.lam_warmup_iters > 0:
            lam_eff = args.lam * min(1.0, it / args.lam_warmup_iters)
        else:
            lam_eff = args.lam
        loss = lam_eff * l_probe + (1 - lam_eff) * l_task

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            save(best_path, it)

        if it % args.log_interval == 0:
            me = eval_max_err(model, num_x, gen, device=device)
            dn = eval_delta_norms()
            history.append(
                {
                    "iter": it,
                    "loss": lv,
                    "l_task": float(l_task.item()),
                    "l_probe": float(l_probe.item()),
                    "lam_eff": lam_eff,
                    "max_err": me,
                    "delta_norms": {str(k): v for k, v in dn.items()},
                }
            )
            with open(hist_path, "w") as f:
                json.dump(history, f)
            rate = (it - start_iter + 1) / (time.time() - t0 + 1e-9)
            dn_str = " ".join(f"L{k}:{v:.2e}" for k, v in dn.items())
            print(
                f"iter {it:>6d}  loss {lv:.3e}  task {l_task.item():.3e}  "
                f"probe {l_probe.item():.3e}  λ {lam_eff:.2f}  max_err {me:.3e}  "
                f"|Δμ| [{dn_str}]  lr {lr:.2e}  {rate:.1f} it/s"
            )

        if it % args.ckpt_interval == 0 and it > start_iter:
            save(last_path, it)

    # final logging + save
    save(last_path, it)
    me = eval_max_err(model, num_x, gen, device=device)
    dn = eval_delta_norms()
    history.append(
        {
            "iter": it,
            "loss": best_loss,
            "l_task": None,
            "l_probe": None,
            "max_err": me,
            "delta_norms": {str(k): v for k, v in dn.items()},
            "final": True,
        }
    )
    with open(hist_path, "w") as f:
        json.dump(history, f)
    dn_str = " ".join(f"L{k}:{v:.2e}" for k, v in dn.items())
    print(
        f"[done] iter {it}  best_loss {best_loss:.3e}  final max_err {me:.3e}  "
        f"|Δμ| [{dn_str}]  elapsed {time.time()-t0:.1f}s"
    )
    print(f"[done] checkpoints in {run_ckpt_dir}, history in {hist_path}")
    print(f"[next] python adversarial_report.py --tag {args.tag}")


if __name__ == "__main__":
    main(args)
