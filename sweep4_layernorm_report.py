"""Compare task-loss distributions between two sets of training runs, to judge
whether an architecture/hyperparameter change improved training reliability.

Built for the LayerNorm question (does enabling LN shift the task-loss
distribution down at fixed lam?), but takes arbitrary run-tag globs, so it
serves any A/B over runs/ directories.

Reliability is reported three ways, because at the sample sizes a sweep can
afford (~15 runs/arm) no single number settles it:
  - fraction of runs below a task-loss threshold -- the headline number, but
    badly underpowered: 6/15 vs 11/15 is only p~0.14 by Fisher exact.
  - the full sorted per-run losses and their quantiles -- what actually shows
    a distribution shift.
  - Mann-Whitney U on the raw losses -- a rank test on the continuous values,
    which is the question "did the distribution shift down" asked directly,
    and keeps the power that binarizing throws away. Reported two-sided; halve
    it for the one-sided "LN is better" reading.

Loss is `data.eval_task_loss` freshly recomputed from each run's checkpoint,
matching sweep_analysis.py / sweep_threshold_report.py rather than reading
history.jsonl (some runs' final checkpoint predates their last history
record). Each run is evaluated under its own checkpoint's
x_p_outer/x_threshold/resid_noise_std, so arms that differ in training noise
are still each scored on the distribution they trained under.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=GLOB",
        help="one arm of the comparison, e.g. "
        "'no-LN=sweep3_lam0.1_tr*'. Repeatable; the first arm is the "
        "reference the others are tested against.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=7e-3,
        help="task loss below which a run counts as having learned the task.",
    )
    p.add_argument("--ckpt", default="last", help="checkpoint name under runs/<tag>/.")
    p.add_argument("--n-eval", type=int, default=50_000, help="fresh examples per run.")
    p.add_argument(
        "--noise-mult",
        type=float,
        default=1.0,
        help="multiplier on each checkpoint's own resid_noise_std at eval.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None, help="also write raw losses here.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import glob
import json
import os
import re

import torch
from scipy.stats import fisher_exact, mannwhitneyu

from data import eval_task_loss
from probe_lib import load_model, resolve_adv_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _trial_index(tag: str) -> int:
    """Sort key so tr2 precedes tr10 (plain string sort would not)."""
    m = re.search(r"tr(\d+)$", tag)
    return int(m.group(1)) if m else -1


def final_task_loss(tag: str, g: torch.Generator, args) -> float:
    model, ck = load_model(tag, args.ckpt, DEVICE)
    adv = resolve_adv_config(ck)
    return eval_task_loss(
        model,
        g,
        DEVICE,
        n=args.n_eval,
        x_p_outer=adv.x_p_outer if adv is not None else None,
        x_threshold=adv.x_threshold if adv is not None else 1.0,
        noise_std=adv.resid_noise_std * args.noise_mult if adv is not None else 0.0,
    )


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def main():
    g = torch.Generator(device=DEVICE).manual_seed(args.seed)
    arms: dict[str, dict[str, float]] = {}
    for spec in args.arm:
        label, _, pattern = spec.partition("=")
        if not pattern:
            raise SystemExit(f"[error] --arm must be LABEL=GLOB, got {spec!r}")
        tags = sorted(
            (os.path.basename(p) for p in glob.glob(os.path.join("runs", pattern))),
            key=_trial_index,
        )
        if not tags:
            raise SystemExit(f"[error] arm {label!r}: no runs match {pattern!r}")
        print(f"[eval] arm {label!r}: {len(tags)} runs matching {pattern!r}")
        losses = {}
        for tag in tags:
            try:
                losses[tag] = final_task_loss(tag, g, args)
            except FileNotFoundError as e:
                print(f"  [skip] {tag}: {e}")
                continue
            print(f"  {tag:32s} {losses[tag]:.4e}")
        if not losses:
            # Distinct from the no-match case above: the tags exist but none
            # had a loadable checkpoint (e.g. a run dir that synced as an
            # empty shell), which would otherwise surface as a ZeroDivisionError
            # in the summary table.
            raise SystemExit(
                f"[error] arm {label!r}: {len(tags)} run(s) matched {pattern!r} "
                f"but none had a loadable {args.ckpt!r} checkpoint."
            )
        arms[label] = losses

    thr = args.threshold
    print(f"\n=== summary (task loss, threshold {thr:.1e}) ===")
    header = f"{'arm':16s} {'n':>3s} {'<thr':>7s} {'frac':>6s} {'median':>10s} {'p25':>10s} {'p75':>10s} {'min':>10s} {'max':>10s}"
    print(header)
    print("-" * len(header))
    for label, losses in arms.items():
        v = sorted(losses.values())
        n_pass = sum(1 for x in v if x < thr)
        print(
            f"{label:16s} {len(v):3d} {n_pass:7d} {n_pass / len(v):6.2f} "
            f"{quantile(v, 0.5):10.3e} {quantile(v, 0.25):10.3e} "
            f"{quantile(v, 0.75):10.3e} {v[0]:10.3e} {v[-1]:10.3e}"
        )

    labels = list(arms)
    ref = labels[0]
    ref_v = list(arms[ref].values())
    for label in labels[1:]:
        v = list(arms[label].values())
        # method="exact" rather than the normal approximation: at n~15/arm the
        # two differ materially (n=5 vs 5, fully separated: 0.0079 vs 0.0122).
        u, p_u = mannwhitneyu(v, ref_v, alternative="two-sided", method="exact")
        a = sum(1 for x in v if x < thr)
        c = sum(1 for x in ref_v if x < thr)
        _odds, p_f = fisher_exact([[a, len(v) - a], [c, len(ref_v) - c]])
        print(f"\n=== {label!r} vs reference {ref!r} ===")
        print(f"  Mann-Whitney U (raw losses, two-sided): U={u:.1f}  p={p_u:.4f}")
        print(f"  Fisher exact on {thr:.1e} threshold counts: p={p_f:.4f}")
        print(
            f"  median {quantile(sorted(v), 0.5):.3e} vs "
            f"{quantile(sorted(ref_v), 0.5):.3e}"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"threshold": thr, "arms": arms}, f, indent=2)
        print(f"\n[write] {args.json_out}")


main()
