"""Relate a run's residual-stream RMS profile to how well it learned the task.

Motivated by the hypothesis that runs whose residual activations "explode"
train worse. Across sweep3_lam0.1 (n=15) that turns out to be the wrong
summary statistic: *peak* RMS is uncorrelated with final task loss
(Spearman rho~0.02), while RMS at the *final* residual layer correlates
strongly (rho~0.73, p~0.002), as does the peak/final decay ratio negatively
(rho~-0.58).

The reading: under an adversarial probe penalty every run inflates the
residual stream at the penalized layer -- that inflation is the hiding
mechanism, not a pathology. What separates a run that learns the task from
one that doesn't is whether it brings the stream back *down* before the
unembed reads it. So an intervention aimed at this failure mode wants to
target the residual scale the output sees, not the peak.

Reports per-run profiles plus Spearman correlations against task loss, for
whichever runs a glob selects.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "run_glob", help="glob under runs/, e.g. 'sweep3_lam0.1_tr*'.", nargs="+"
    )
    p.add_argument("--ckpt", default="last")
    p.add_argument(
        "--threshold",
        type=float,
        default=7e-3,
        help="task loss below which a run counts as having learned the task.",
    )
    p.add_argument("--n-rms", type=int, default=4096, help="batch for the RMS profile.")
    p.add_argument(
        "--n-eval", type=int, default=50_000, help="fresh examples for the task loss."
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import glob
import os
import re

import torch
from scipy.stats import spearmanr

from data import eval_task_loss, sample_batch
from probe_lib import load_model, resolve_adv_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _trial_index(tag: str) -> int:
    m = re.search(r"tr(\d+)$", tag)
    return int(m.group(1)) if m else -1


def profile(tag: str, g: torch.Generator, args) -> dict:
    """Task loss plus the per-layer residual RMS profile for one run, both
    measured under the run's own training noise/input distribution."""
    model, ck = load_model(tag, args.ckpt, DEVICE)
    adv = resolve_adv_config(ck)
    x_p_outer = adv.x_p_outer if adv is not None else None
    x_threshold = adv.x_threshold if adv is not None else 1.0
    noise_std = adv.resid_noise_std if adv is not None else 0.0

    loss = eval_task_loss(
        model,
        g,
        DEVICE,
        n=args.n_eval,
        x_p_outer=x_p_outer,
        x_threshold=x_threshold,
        noise_std=noise_std,
    )
    x_full, _ = sample_batch(
        args.n_rms,
        model.num_x,
        generator=g,
        device=DEVICE,
        x_p_outer=x_p_outer,
        x_threshold=x_threshold,
    )
    with torch.no_grad():
        _y, caches = model(x_full, return_cache=True, noise=noise_std, generator=g)
    rms = [c.pow(2).mean().sqrt().item() for c in caches]
    return dict(tag=tag, loss=loss, rms=rms, peak=max(rms), final=rms[-1])


def main():
    g = torch.Generator(device=DEVICE).manual_seed(args.seed)
    tags = sorted(
        {
            os.path.basename(p)
            for pattern in args.run_glob
            for p in glob.glob(os.path.join("runs", pattern))
        },
        key=_trial_index,
    )
    if not tags:
        raise SystemExit(f"[error] no runs match {args.run_glob}")

    rows = []
    for tag in tags:
        try:
            rows.append(profile(tag, g, args))
        except FileNotFoundError as e:
            print(f"[skip] {tag}: {e}")
    if not rows:
        raise SystemExit("[error] no run had a loadable checkpoint.")
    for r in rows:
        r["decay"] = r["peak"] / r["final"] if r["final"] else float("nan")

    rows.sort(key=lambda r: r["loss"])
    print(
        f"\n{'tag':28s} {'task_loss':>10s} {'peak':>8s} {'final':>8s} "
        f"{'peak/final':>10s}       per-layer RMS"
    )
    for r in rows:
        prof = " ".join(f"{v:5.1f}" for v in r["rms"])
        flag = "FAIL" if r["loss"] >= args.threshold else "pass"
        print(
            f"{r['tag']:28s} {r['loss']:10.3e} {r['peak']:8.1f} {r['final']:8.1f} "
            f"{r['decay']:10.2f}  {flag}  {prof}"
        )

    losses = [r["loss"] for r in rows]
    print(f"\n=== Spearman vs task loss (n={len(rows)}) ===")
    for key in ("peak", "final", "decay"):
        rho, p = spearmanr(losses, [r[key] for r in rows])
        print(f"  {key:6s} rho={rho:+.3f}  p={p:.4f}{'  *' if p < 0.05 else ''}")


main()
