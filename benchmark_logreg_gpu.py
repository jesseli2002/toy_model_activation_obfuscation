"""CPU (sklearn) vs. GPU (torch) logistic-regression fit timing, for the
probe's actual usage pattern in train_adversarial_logreg.py: an init-scale
fit (large max_iter, from scratch) and a small warm-started per-step update
(small max_iter, resumed).

Requires user help to run the GPU arm -- this sandbox has no CUDA device; run
this script on a CUDA machine for the real comparison.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="Time sklearn (CPU) vs. torch (GPU) logistic regression "
        "across input dimension.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-points", type=int, default=10000)
    p.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024],
        help="input dimensions to sweep (powers of two, matching the "
        "concatenated-hidden-layer widths the probe actually sees).",
    )
    p.add_argument(
        "--init-iters", type=int, default=1000, help="max_iter for the init-scale fit."
    )
    p.add_argument(
        "--step-iters",
        type=int,
        default=100,
        help="max_iter for the warm-started per-step fit.",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="number of warm-started per-step fits to time (averaged).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="repeats per (dim, backend) timing, min taken.",
    )
    p.add_argument("--out", type=str, default="benchmark_logreg_gpu.csv")
    p.add_argument("--plot", type=str, default="benchmark_logreg_gpu.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probe_backend import SklearnProbePipeline, TorchProbePipeline


def make_data(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w_true = rng.normal(size=d).astype(np.float32)
    y = (X @ w_true + 0.5 * rng.normal(size=n)) >= 0
    assert y.any() and (~y).any()
    return X, y


def time_fit(build_fn, X, y, max_iter: int, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        pipeline = build_fn()
        t0 = time.perf_counter()
        pipeline.set_max_iter(max_iter)
        pipeline.fit(X, y)
        if isinstance(X, torch.Tensor) and X.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def time_warmstart_steps(
    build_fn, X, y, max_iter: int, n_steps: int, repeats: int
) -> float:
    """Average per-step time of n_steps warm-started fit() calls (fresh
    pipeline each repeat, timed after an init fit so the comparison matches
    the actual per-step cost during training)."""
    times = []
    for _ in range(repeats):
        pipeline = build_fn()
        pipeline.set_max_iter(max_iter)
        pipeline.fit(X, y)  # untimed init fit to reach warm-started state
        t0 = time.perf_counter()
        for _ in range(n_steps):
            pipeline.set_max_iter(max_iter)
            pipeline.fit(X, y)
        if isinstance(X, torch.Tensor) and X.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n_steps)
    return min(times)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "[warn] no CUDA device visible -- the 'torch' backend will run on "
            "CPU, which is NOT a GPU timing (only useful as a correctness "
            "smoke test of the code path). Run this script on a CUDA machine "
            "for the real CPU-vs-GPU comparison."
        )

    rows = []  # dict: dim, backend, phase, seconds
    for d in args.dims:
        X_np, y_np = make_data(args.n_points, d, args.seed)
        y_bool = y_np.astype(bool)

        # sklearn (CPU) arm: numpy in, numpy out.
        sk_init = time_fit(
            lambda: SklearnProbePipeline(C=1.0, max_iter=args.init_iters),
            X_np,
            y_bool,
            args.init_iters,
            args.repeats,
        )
        sk_step = time_warmstart_steps(
            lambda: SklearnProbePipeline(C=1.0, max_iter=args.step_iters),
            X_np,
            y_bool,
            args.step_iters,
            args.n_steps,
            args.repeats,
        )
        rows.append(
            {"dim": d, "backend": "sklearn (cpu)", "phase": "init", "seconds": sk_init}
        )
        rows.append(
            {"dim": d, "backend": "sklearn (cpu)", "phase": "step", "seconds": sk_step}
        )

        # torch arm: tensors on `device` (cuda if available, else cpu).
        X_t = torch.as_tensor(X_np, device=device)
        y_t = torch.as_tensor(y_bool, device=device)
        torch_init = time_fit(
            lambda: TorchProbePipeline(C=1.0, max_iter=args.init_iters),
            X_t,
            y_t,
            args.init_iters,
            args.repeats,
        )
        torch_step = time_warmstart_steps(
            lambda: TorchProbePipeline(C=1.0, max_iter=args.step_iters),
            X_t,
            y_t,
            args.step_iters,
            args.n_steps,
            args.repeats,
        )
        backend_label = f"torch ({device})"
        rows.append(
            {"dim": d, "backend": backend_label, "phase": "init", "seconds": torch_init}
        )
        rows.append(
            {"dim": d, "backend": backend_label, "phase": "step", "seconds": torch_step}
        )

        print(
            f"d={d:>5d}  sklearn(cpu) init={sk_init*1e3:8.2f}ms step={sk_step*1e3:7.3f}ms  "
            f"torch({device}) init={torch_init*1e3:8.2f}ms step={torch_step*1e3:7.3f}ms"
        )

    with open(args.out, "w") as f:
        f.write("dim,backend,phase,seconds\n")
        for r in rows:
            f.write(f"{r['dim']},{r['backend']},{r['phase']},{r['seconds']}\n")
    print(f"[done] wrote {args.out}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for phase, ax in zip(["init", "step"], axes):
        backends = sorted({r["backend"] for r in rows})
        for backend in backends:
            xs = [
                r["dim"]
                for r in rows
                if r["backend"] == backend and r["phase"] == phase
            ]
            ys = [
                r["seconds"]
                for r in rows
                if r["backend"] == backend and r["phase"] == phase
            ]
            ax.plot(xs, ys, marker="o", label=backend)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("input dimension")
        ax.set_ylabel("wall time (s)")
        ax.set_title(
            f"{phase}-fit (max_iter={args.init_iters if phase == 'init' else args.step_iters})"
        )
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Logistic regression fit time, n={args.n_points} points")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"[done] wrote {args.plot}")


if __name__ == "__main__":
    main(args)
