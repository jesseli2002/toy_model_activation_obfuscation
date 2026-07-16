"""Train the residual MLP on the saturation task (Step 1).

Fresh synthetic data every step (infinite data). MSE over the first num_x outputs
only. Early-stop when loss < EARLY_STOP_LOSS. Checkpoints best + last to
checkpoints/<tag>/, resumable with --resume.

Convergence note: an exact zero-loss solution exists (see analytic.py), so any
plateau is an OPTIMIZATION problem, not a capacity one. Escalation ladder if a
given num_x plateaus above the gate:
    1. widen LR sweep (try up to ~3e-2) with cosine decay to ~1e-6
    2. longer schedule (more --max-iters)
    3. warm-start from analytic.py (--warm-start) to confirm it's optimization
    4. STOP and report — do not inflate d_mlp past num_x+1 (that breaks Steps 2-3)

Usage examples:
    python train.py --num-x 1 --tag nx1 --lr 3e-3 --max-iters 20000
    python train.py --resume --tag nx1 --max-iters 40000
"""

import argparse
import json
import os
import time

import torch

import config
from data import sample_batch
from model import ResidualMLP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-x", type=int, default=config.NUM_X)
    p.add_argument("--d-model", type=int, default=config.D_MODEL)
    p.add_argument("--d-mlp", type=int, default=None, help="default: num_x+1")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument(
        "--lr-final", type=float, default=1e-6, help="cosine-decay target LR"
    )
    p.add_argument("--max-iters", type=int, default=config.MAX_ITERS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--early-stop-loss", type=float, default=config.EARLY_STOP_LOSS)
    p.add_argument("--out-init-scale", type=float, default=0.1)
    p.add_argument(
        "--warm-start",
        action="store_true",
        help="initialize from analytic.py exact weights (diagnostic)",
    )
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    return p.parse_args()


def cosine_lr(step, total, lr0, lr1):
    import math

    if total <= 1:
        return lr0
    t = min(step, total) / total
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def eval_max_err(model, num_x, n=100_000, batch=20_000, seed=12345, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    worst = 0.0
    done = 0
    while done < n:
        b = min(batch, n - done)
        x_full, y = sample_batch(b, num_x, generator=g, device=device)
        pred = model.task_output(x_full)
        worst = max(worst, (pred - y).abs().max().item())
        done += b
    return worst


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_x = args.num_x
    d_mlp = args.d_mlp if args.d_mlp is not None else config.d_mlp_for(num_x)

    ckpt_dir = os.path.join("checkpoints", args.tag)
    log_dir = os.path.join("logs", args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    model = ResidualMLP(
        num_x, args.d_model, d_mlp, out_init_scale=args.out_init_scale
    ).to(device)
    if args.warm_start:
        from analytic import build_exact_model

        exact = build_exact_model(num_x, args.d_model, d_mlp)
        model.load_state_dict(exact.state_dict())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    start_iter = 0
    history = []  # list of (iter, loss, max_err)
    best_loss = float("inf")
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    hist_path = os.path.join(log_dir, "history.json")

    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"]
        best_loss = ck.get("best_loss", float("inf"))
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        print(f"[resume] from iter {start_iter}, best_loss={best_loss:.3e}")

    def save(path, it):
        torch.save(
            {
                "iter": it,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_loss": best_loss,
                "config": {
                    "num_x": num_x,
                    "d_model": args.d_model,
                    "d_mlp": d_mlp,
                    "seed": args.seed,
                },
            },
            path,
        )

    print(
        f"[train] tag={args.tag} num_x={num_x} d_model={args.d_model} "
        f"d_mlp={d_mlp} bs={args.batch_size} lr={args.lr} device={device} "
        f"iters {start_iter}->{args.max_iters}"
    )

    t0 = time.time()
    it = start_iter
    stopped_early = False
    for it in range(start_iter, args.max_iters):
        lr = cosine_lr(it, args.max_iters, args.lr, args.lr_final)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x_full, y = sample_batch(args.batch_size, num_x, generator=gen, device=device)
        pred = model.task_output(x_full)
        loss = torch.mean((pred - y) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            save(best_path, it)

        if it % args.log_interval == 0 or lv < args.early_stop_loss:
            me = eval_max_err(model, num_x, device=device)
            history.append((it, lv, me))
            with open(hist_path, "w") as f:
                json.dump(history, f)
            rate = (it - start_iter + 1) / (time.time() - t0 + 1e-9)
            print(
                f"iter {it:>7d}  loss {lv:.3e}  max_err {me:.3e}  "
                f"lr {lr:.2e}  {rate:.1f} it/s"
            )

        if it % args.ckpt_interval == 0 and it > start_iter:
            save(last_path, it)

        if lv < args.early_stop_loss:
            print(
                f"[early stop] loss {lv:.3e} < {args.early_stop_loss:.1e} at iter {it}"
            )
            stopped_early = True
            break

    save(last_path, it)
    me = eval_max_err(model, num_x, device=device)
    history.append((it, best_loss, me))
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(
        f"[done] iter {it}  best_loss {best_loss:.3e}  final max_err {me:.3e}  "
        f"early_stop={stopped_early}  elapsed {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
