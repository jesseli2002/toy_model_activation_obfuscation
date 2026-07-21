"""Brute-force search: can a width-n residual MLP solve sat(x,c) exactly with
mean-constant residuals at every probed layer (n=2, n=3)?

Method: float64 optimization of task MSE + lambda * mean-constancy penalty,
lambda ramped to 1e6. Controls validate the method:
  - nohide: same size, lambda=0        -> should reach ~0 task MSE (anchor exists)
  - wide:   d_mlp=2n+1, with hiding    -> should reach ~0 both (known construction)
Verdict signal: hide runs plateau far above controls => impossibility evidence;
hide runs match controls => extract + verify structure symbolically.
"""

import json
import math
import os
import pathlib
import time

import torch

OUT = pathlib.Path(os.environ.get("TMPDIR", "."))

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"


def make_params(n, d_model, d_mlp, num_blocks, gen):
    params = []
    for _ in range(num_blocks):
        W_in = (torch.randn(d_model, d_mlp, generator=gen) / math.sqrt(d_model)).to(
            device
        )
        b_in = torch.zeros(d_mlp, device=device)
        W_out = (
            torch.randn(d_mlp, d_model, generator=gen) * 0.1 / math.sqrt(d_mlp)
        ).to(device)
        b_out = torch.zeros(d_model, device=device)
        for t in (W_in, b_in, W_out, b_out):
            t.requires_grad_(True)
        params.append((W_in, b_in, W_out, b_out))
    return params


def warm_start_block0(params, n):
    """Block 0 = c-erasure + v1 encoding channel on dim n+1 (uses x1=dim0, c=dim n)."""
    W_in, b_in, W_out, b_out = params[0]
    with torch.no_grad():
        W_in.zero_()
        b_in.zero_()
        W_out.zero_()
        b_out.zero_()
        c_dim, v_dim = n, n + 1
        # neuron 0: relu(c+10) always-on
        W_in[c_dim, 0] = 1.0
        b_in[0] = 10.0
        # neuron 1: relu(-x1 - c)
        W_in[0, 1] = -1.0
        W_in[c_dim, 1] = -1.0
        # neuron 2: relu(x1 - 3 + c)
        W_in[0, 2] = 1.0
        W_in[c_dim, 2] = 1.0
        b_in[2] = -3.0
        # erase c: dim c gets -(c+10)+10
        W_out[0, c_dim] = -1.0
        b_out[c_dim] = 10.0
        # v1 = -2*relu(-x1-c) + 2*relu(x1-3+c) - c + 3/2 on v_dim
        W_out[1, v_dim] = -2.0
        W_out[2, v_dim] = 2.0
        W_out[0, v_dim] = -1.0
        b_out[v_dim] = 10.0 + 1.5


def forward_res(params, r):
    caches = []
    for W_in, b_in, W_out, b_out in params:
        h = torch.relu(r @ W_in + b_in)
        r = r + h @ W_out + b_out
        caches.append(r)
    return r, caches


def build_batch(n, n_x_pts, n_c, gen_np_seed):
    sob = torch.quasirandom.SobolEngine(n, scramble=True, seed=gen_np_seed)
    X = (sob.draw(n_x_pts).to(device) * 6.0 - 3.0).to(torch.float64)
    cs = torch.linspace(1.0, 2.0, n_c, device=device)
    return X, cs


def losses(params, X, cs, n, d_model):
    B = X.shape[0]
    n_c = cs.shape[0]
    # rows: (n_c*B, d_model); dims 0..n-1 = x, dim n = c, rest 0
    r0 = torch.zeros(n_c * B, d_model, device=device)
    r0[:, :n] = X.repeat(n_c, 1)
    r0[:, n] = cs.repeat_interleave(B)
    _, caches = forward_res(params, r0)
    y = caches[-1][:, :n]
    target = torch.clamp(
        X.repeat(n_c, 1),
        -cs.repeat_interleave(B)[:, None],
        cs.repeat_interleave(B)[:, None],
    )
    task = ((y - target) ** 2).mean()
    viol = 0.0
    for r in caches:
        m = r.reshape(n_c, B, d_model).mean(dim=1)  # (n_c, d_model)
        viol = viol + ((m - m.mean(dim=0, keepdim=True)) ** 2).mean()
    viol = viol / len(caches)
    return task, viol


def run(cfg):
    n = cfg["n"]
    d_model = cfg["d_model"]
    d_mlp = cfg["d_mlp"]
    nb = cfg["blocks"]
    seed = cfg["seed"]
    gen = torch.Generator().manual_seed(seed)
    params = make_params(n, d_model, d_mlp, nb, gen)
    if cfg.get("warm"):
        warm_start_block0(params, n)
    flat = [t for p in params for t in p]
    X, cs = build_batch(n, cfg.get("n_x_pts", 4096), cfg.get("n_c", 9), seed + 1)

    steps = cfg.get("steps", 18000)
    opt = torch.optim.Adam(flat, lr=3e-3)
    lam_lo, lam_hi = cfg.get("lam_lo", 1e2), cfg.get("lam_hi", 1e6)
    hide = cfg.get("hide", True)
    for step in range(steps):
        t = step / steps
        lam = lam_lo * (lam_hi / lam_lo) ** t if hide else 0.0
        for g in opt.param_groups:
            g["lr"] = 3e-3 * (0.5 * (1 + math.cos(math.pi * t))) + 1e-6
        opt.zero_grad()
        task, viol = losses(params, X, cs, n, d_model)
        (task + lam * viol).backward()
        opt.step()

    # holdout evaluation: bigger fresh sobol sample, finer c grid
    with torch.no_grad():
        Xh, csh = build_batch(n, 32768, 17, seed + 999)
        task_h, viol_h = losses(params, Xh, csh, n, d_model)
    res = {
        **{k: v for k, v in cfg.items()},
        "task_holdout": task_h.item(),
        "viol_holdout": viol_h.item(),
    }
    torch.save(
        [
            {
                "W_in": p[0].detach().cpu(),
                "b_in": p[1].detach().cpu(),
                "W_out": p[2].detach().cpu(),
                "b_out": p[3].detach().cpu(),
            }
            for p in params
        ],
        OUT / f"weights_{cfg['name']}_s{seed}.pt",
    )
    return res


def main():
    configs = []
    # Controls
    for seed in (0, 1):
        configs.append(
            dict(
                name="ctrl_nohide_n3",
                n=3,
                d_model=12,
                d_mlp=3,
                blocks=4,
                hide=False,
                seed=seed,
            )
        )
        configs.append(
            dict(
                name="ctrl_wide_n3",
                n=3,
                d_model=12,
                d_mlp=7,
                blocks=2,
                hide=True,
                seed=seed,
            )
        )
    # n=2 (theory says impossible -> expect floor)
    for seed in (0, 1, 2):
        configs.append(
            dict(
                name="hide_n2_b4",
                n=2,
                d_model=10,
                d_mlp=2,
                blocks=4,
                hide=True,
                seed=seed,
            )
        )
    # n=3 main
    for seed in (0, 1, 2):
        configs.append(
            dict(
                name="hide_n3_b4",
                n=3,
                d_model=12,
                d_mlp=3,
                blocks=4,
                hide=True,
                seed=seed,
            )
        )
    for seed in (0, 1):
        configs.append(
            dict(
                name="hide_n3_b6",
                n=3,
                d_model=16,
                d_mlp=3,
                blocks=6,
                hide=True,
                seed=seed,
            )
        )
    # warm-started v1 encoding
    configs.append(
        dict(
            name="hide_n3_b4_warm",
            n=3,
            d_model=12,
            d_mlp=3,
            blocks=4,
            hide=True,
            seed=7,
            warm=True,
        )
    )
    configs.append(
        dict(
            name="hide_n3_b6_warm",
            n=3,
            d_model=16,
            d_mlp=3,
            blocks=6,
            hide=True,
            seed=8,
            warm=True,
        )
    )

    results = []
    for cfg in configs:
        t0 = time.time()
        res = run(cfg)
        res["secs"] = round(time.time() - t0, 1)
        results.append(res)
        print(
            f"{cfg['name']:20s} seed={cfg['seed']} "
            f"task={res['task_holdout']:.3e} viol={res['viol_holdout']:.3e} "
            f"({res['secs']}s)",
            flush=True,
        )
        with open(OUT / "search_results.json", "w") as f:
            json.dump(results, f, indent=1)

    print("\n=== SUMMARY (best per config) ===")
    best = {}
    for r in results:
        k = r["name"]
        if (
            k not in best
            or r["task_holdout"] + r["viol_holdout"]
            < best[k]["task_holdout"] + best[k]["viol_holdout"]
        ):
            best[k] = r
    for k, r in best.items():
        print(f"{k:20s} task={r['task_holdout']:.3e} viol={r['viol_holdout']:.3e}")


if __name__ == "__main__":
    main()
