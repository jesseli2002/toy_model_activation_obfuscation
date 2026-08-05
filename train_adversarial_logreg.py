"""Adversarial training: model vs. a stateful LogisticRegression probe,
trained *simultaneously* with the model.

The probe is a single LogisticRegression instance over the concatenation of
one or more penalized hidden layers, advanced a few solver iterations per
training step from the previous step's coefficients rather than refit from
scratch (see the three run modes below for how the probe's initial
coefficients are established). The probe backend is
`sklearn.linear_model.LogisticRegression` (CPU) or one of two GPU-resident
torch reimplementations solving the same objective, selected via
`--probe-backend` (see probe_backend.py).

The training objective combines a probe-adversarial penalty with the task
loss (see LogregAdversarialConfig for the weighting and probe hyperparameters).
The probe (both its own fit and the differentiable penalty) is always
evaluated against one fixed input batch sampled before the training loop
starts -- the task loss, by contrast, resamples a fresh batch every step.

Each run is a standalone experiment with no automated pass/fail check. The
deliverable is the trained checkpoint + diagnostics; run once, then stop and
review.

Three run modes (see plans/rare_flags_config_plan.md for the full design):
  - fresh run: hyperparameters are freshly resolved from `--config`'s JSON
    file. Writes `runs/<tag>/config.json` once.
  - `--resume <tag>`: strictly continues the same experiment. All
    hyperparameters are restored from the checkpoint, never re-read from
    `--config`. `runs/<tag>/config.json` is read once for a sanity check
    against the restored config (a mismatch warns but never rewrites the
    file) -- not for resolving hyperparameters.
  - `--fork-from <source_tag>` (with `--tag <new_tag>`): branches a new
    experiment off `source_tag`'s checkpoint -- architecture, weights, and
    iteration count come from there, same as `--resume` -- but the optimizer
    (including its state, not just its hyperparameters) and the rest of the
    adversarial-objective hyperparameters are freshly resolved exactly like a
    fresh run. `runs/<new_tag>/config.json` additionally records `forked_from`.

Two run-directory artifacts are written once, at tag-creation time, and are
read-only thereafter for the lifetime of that tag (see `LogregRunConfig` for
`config.json`'s schema): `runs/<tag>/input_config.json` is a verbatim copy of
the `--config` file, kept reusable as a later `--config` argument (e.g. for
`--fork-from`); `runs/<tag>/config.json` is the fully-resolved run config.
Hand-editing either after creation has no effect except tripping the
mismatch warning on a later `--resume`.

Progress is written as numbered `runs/<tag>/checkpoints/iter_<n>.pt` files
with `last.pt` a symlink to the newest, plus `logs/history.jsonl`. Two
invariants hold between them for a run at rest -- one that finished, or that
stopped on a Ctrl-C:

  - the last line of `history.jsonl` describes the iteration `last.pt`
    resolves to;
  - every `iter_<n>.pt` on disk has a history entry.

Both come from `--ckpt-interval` being a multiple of `--log-interval` (so
periodic checkpoints only land on logged iterations) plus the loop logging
and checkpointing whichever iteration it actually stopped at. `best.pt` is
outside this: it is written whenever the loss improves, which is generally
not a logged iteration.

"At rest" is the limit of the guarantee. Mid-run, history legitimately runs
ahead of `last.pt` between checkpoints, and a run killed without a SIGINT
(SIGKILL, OOM, power) can be left that way -- a later `--resume` then logs
those iterations a second time.

Configuration is split two ways, and the split is a near-invariant worth
naming: a setting lives in the `--config` JSON file (i.e. on
`LogregAdversarialConfig`) if and only if it can change across a
`--fork-from`. Concretely:
  - Architecture (`ResidualMLPConfig`: `--num-x`/`--d-model`/`--d-mlp`/
    `--num-blocks`) is pinned once, at from-scratch init, and is never in the
    config file -- `--resume`/`--fork-from` always inherit it from the source
    checkpoint rather than letting it be re-specified.
  - Everything else that affects training is config-file-only, with no CLI
    flag at all, so a `--fork-from`'s exact hyperparameters live in one
    reviewable/diffable JSON file rather than split across CLI history and a
    file. `lam` used to be the sole exception (a CLI flag with a Python
    default, since it's the most frequently tuned value) -- that carve-out
    caused enough confusion to not be worth it, so `lam` moved into the
    config file too.
  - Bookkeeping flags (`--tag`, `--resume`, `--fork-from`, `--tag-force`,
    `--probe-backend`, `--log-interval`, `--ckpt-interval`,
    `--max-iters`) sit outside this split entirely: they're run control that
    never varies within one tag's lineage and is never persisted to
    `config.json`.
"""

import argparse
import collections
import copy
import dataclasses
import json
import math
import os
import shutil
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass

import config
from config import (
    ForkedFrom,
    LogregAdversarialConfig,
    LogregRunConfig,
    ResidualMLPConfig,
)
from paths import ckpt_dir, log_dir, run_dir
from rate_meter import EMARateMeter

# Per-step warm-started solver iterations for the probe update (small: the
# solver resumes from last step's coefficients, so a handful of lbfgs steps
# is enough to track the model). The init fit (before the training loop) uses
# --probe-init-iters instead, since it starts from scratch.
PROBE_STEP_MAX_ITER = 100


def parse_args():
    p = argparse.ArgumentParser(
        description="Adversarial training: model vs. a simultaneous, "
        "stateful LogisticRegression probe (sklearn or GPU-resident torch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_init = p.add_argument_group(
        "model initialization",
        "Required for a fresh run (no default -- see ResidualMLPConfig); "
        "errors if passed alongside --resume/--fork-from, which instead "
        "take the architecture from the checkpoint being restored.",
    )
    g_init.add_argument("--num-x", type=int, default=None)
    g_init.add_argument("--d-model", type=int, default=None)
    g_init.add_argument("--d-mlp", type=int, default=None)
    g_init.add_argument("--num-blocks", type=int, default=None)

    g_book = p.add_argument_group("bookkeeping")
    g_book.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON file with the config-file-only hyperparameters (see "
        "LogregAdversarialConfig's fields without a default). Required "
        "for a fresh run or --fork-from; under --resume it's never read -- "
        "hyperparameters instead come from the checkpoint.",
    )
    g_book.add_argument("--tag", type=str, default="adv-logreg")
    g_book.add_argument("--resume", action="store_true")
    g_book.add_argument(
        "--seed",
        type=int,
        default=None,
        help="required for a fresh run or --fork-from; forbidden under "
        "--resume, which reseeds from the checkpoint's own recorded seed "
        "instead of this invocation's.",
    )
    g_book.add_argument(
        "--fork-from",
        type=str,
        default=None,
        metavar="SOURCE_TAG",
        help="branch a new --tag off SOURCE_TAG's latest checkpoint (architecture, "
        "weights, optimizer state, iteration count, and history -- continued for "
        "continuous loss curves). Unlike --resume, the adversarial-objective "
        "hyperparameters are freshly resolved from this invocation's "
        "--config, not inherited from SOURCE_TAG. Mutually exclusive "
        "with --resume.",
    )
    g_book.add_argument(
        "--tag-force",
        action="store_true",
        help="delete an existing runs/<tag> directory before a fresh run.",
    )
    g_book.add_argument(
        "--probe-backend",
        choices=config.PROBE_BACKEND_CHOICES,
        default="auto",
        help="'auto' (default): torch (GPU-resident) probe iff CUDA is "
        "available, else sklearn. The others force a backend regardless of "
        "device -- e.g. to smoke-test a GPU backend on a CPU-only machine. "
        "'newton' solves the same probe objective by damped Newton instead "
        "of L-BFGS: far cheaper per fit on launch-latency-bound hardware, at "
        "equal-or-better accuracy (see probe_newton.py).",
    )
    g_book.add_argument("--log-interval", type=int, default=100)
    g_book.add_argument("--ckpt-interval", type=int, default=200)
    g_book.add_argument("--max-iters", type=int, default=config.MAX_ITERS)
    g_book.add_argument("--rate-meter-window", type=float, default=1000.0)

    args = p.parse_args()
    if args.resume and args.fork_from is not None:
        p.error("--resume and --fork-from are mutually exclusive.")
    if not args.resume and args.config is None:
        p.error("--config PATH is required (unless --resume).")
    if args.resume and args.seed is not None:
        p.error(
            "--seed cannot be combined with --resume -- resume reseeds from "
            "the checkpoint's own recorded seed instead."
        )
    if not args.resume and args.seed is None:
        p.error("--seed required for a fresh run or --fork-from.")

    # Every checkpoint iter must also be a log iter, so each checkpoint on
    # disk has a matching history.jsonl entry.
    if args.log_interval <= 0:
        p.error(f"--log-interval must be positive, got {args.log_interval}.")
    if args.ckpt_interval <= 0:
        p.error(f"--ckpt-interval must be positive, got {args.ckpt_interval}.")
    if args.ckpt_interval % args.log_interval != 0:
        p.error(
            f"--ckpt-interval ({args.ckpt_interval}) must be a multiple of "
            f"--log-interval ({args.log_interval})."
        )

    arch_flags = {
        "num_x": "--num-x",
        "d_model": "--d-model",
        "d_mlp": "--d-mlp",
        "num_blocks": "--num-blocks",
    }
    if args.resume or args.fork_from is not None:
        offending = [
            flag for dest, flag in arch_flags.items() if getattr(args, dest) is not None
        ]
        if offending:
            mode = "--resume" if args.resume else "--fork-from"
            p.error(
                f"{', '.join(offending)} cannot be combined with {mode} -- "
                f"architecture comes from the checkpoint being restored."
            )
    else:
        missing = [
            flag for dest, flag in arch_flags.items() if getattr(args, dest) is None
        ]
        if missing:
            p.error(f"{', '.join(missing)} required for a fresh run.")
    return args


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()
    # Cheap existence checks, fired before run-dir setup (in main()) can delete
    # an existing runs/<tag> or start restoring from a nonexistent checkpoint.
    if args.fork_from is not None:
        source_ckpt = os.path.join(ckpt_dir(args.fork_from), "last.pt")
        if not os.path.exists(source_ckpt):
            raise SystemExit(
                f"[error] --fork-from source checkpoint not found: {source_ckpt}"
            )
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir(args.tag), "last.pt")
        if not os.path.exists(resume_ckpt):
            raise SystemExit(
                f"[error] --resume: no checkpoint at {resume_ckpt} to resume from."
            )

import warnings

import torch
from sklearn.exceptions import ConvergenceWarning

from data import sample_batch
from model import ResidualMLP, ResidualMLPConfig
from data import eval_max_err
from probe_backend import build_probe_pipeline, fit_probe, resolve_probe_backend
from stableadamw import StableAdamW


def _resolve_hidden_layers(penalty_layers, num_blocks: int) -> list[int]:
    """Hidden residual layers = 1 .. num_blocks-1 (see module docstring)."""
    all_hidden = list(range(1, num_blocks))
    if penalty_layers == "all":
        return all_hidden
    layers = sorted(set(penalty_layers))
    for lyr in layers:
        if lyr == 0:
            raise SystemExit(
                "[error] layer 0 is the embedding: c sits in a fixed coordinate, "
                "so a probe there is trivially perfect. Penalizing it fights an "
                "unwinnable battle for no reason; drop it."
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


def _read_config_file(path: str) -> dict:
    """Read --config's JSON file. Required-key validation happens later, in
    load_run_config."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[error] --config file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[error] --config {path} is not valid JSON: {e}")


def load_run_config(
    file_fields: dict, *, num_blocks: int, config_path: str
) -> tuple[LogregAdversarialConfig, list[int]]:
    """Build a LogregAdversarialConfig from --config's file_fields, resolving
    `penalty_layers` (a config-file value of "all" or an explicit list)
    against num_blocks first -- so a missing/invalid `penalty_layers`
    surfaces the same as any other bad --config key. A missing required key
    in the file surfaces as the dataclass constructor's TypeError; re-raised
    here as a SystemExit naming the specific key(s), per this module's
    [error] convention. Returns (adv_config, hidden_layers) -- the latter is
    what callers actually index caches with."""
    fields = dict(file_fields)
    if "penalty_layers" in fields:
        fields["penalty_layers"] = _resolve_hidden_layers(
            fields["penalty_layers"], num_blocks
        )
    try:
        adv_config = LogregAdversarialConfig(**fields)
    except TypeError:
        required = {
            f.name
            for f in dataclasses.fields(LogregAdversarialConfig)
            if f.default is dataclasses.MISSING
        }
        missing = sorted(required - fields.keys())
        if missing:
            raise SystemExit(
                f"[error] --config {config_path} missing required key(s): {missing}"
            )
        raise
    return adv_config, adv_config.penalty_layers


def _config_json_path(tag: str) -> str:
    return os.path.join(run_dir(tag), "config.json")


def _input_config_json_path(tag: str) -> str:
    return os.path.join(run_dir(tag), "input_config.json")


def _write_run_config(
    tag: str,
    model_config: ResidualMLPConfig,
    adv_config: LogregAdversarialConfig,
    input_config_path: str,
    seed: int,
    forked_from: ForkedFrom | None = None,
) -> None:
    """Write runs/<tag>/config.json (fully resolved) and
    runs/<tag>/input_config.json (verbatim copy of --config) once, at
    tag-creation time (fresh run or --fork-from). Write-once, read-only
    thereafter -- see module docstring; --resume never calls this."""
    run_config = LogregRunConfig(
        model=model_config, adversarial=adv_config, seed=seed, forked_from=forked_from
    )
    path = _config_json_path(tag)
    with open(path, "w") as f:
        json.dump(run_config.to_dict(), f, indent=2)
    shutil.copyfile(input_config_path, _input_config_json_path(tag))
    print(
        f"[config] wrote {path} (write-once -- hand-edits after this point "
        f"are only detected, as a warning, on a later --resume; never applied)"
    )


def _check_config_json(
    tag: str, model_config: ResidualMLPConfig, adv_config: LogregAdversarialConfig
) -> None:
    """--resume sanity check: compare runs/<tag>/config.json (written once at
    tag-creation) against the checkpoint-restored config. A mismatch means
    someone hand-edited the file since -- warn and proceed on the
    checkpoint's values; the file itself is left untouched either way."""
    path = _config_json_path(tag)
    if not os.path.exists(path):
        print(f"[warn] {path} not found -- nothing to sanity-check against.")
        return
    with open(path) as f:
        on_disk = LogregRunConfig.from_dict(json.load(f))
    if on_disk.adversarial != adv_config or on_disk.model != model_config:
        print(
            f"[warn] {path} does not match the checkpoint's config -- was it "
            f"hand-edited after {tag} was created? Proceeding with the "
            f"checkpoint's values; {path} is left as-is."
        )


def _lr_at(it: int, max_iters: int, adv_config: LogregAdversarialConfig) -> float:
    """Linear warmup over lr_warmup_iters, then cosine decay from lr to
    lr * lr_min_frac over the rest of the run -- see
    LogregAdversarialConfig.lr_warmup_iters."""
    if it < adv_config.lr_warmup_iters:
        return adv_config.lr * (it + 1) / adv_config.lr_warmup_iters
    decay_span = max(1, max_iters - adv_config.lr_warmup_iters)
    progress = min(1.0, (it - adv_config.lr_warmup_iters) / decay_span)
    cos_frac = 0.5 * (1 + math.cos(math.pi * progress))
    frac = adv_config.lr_min_frac + (1 - adv_config.lr_min_frac) * cos_frac
    return adv_config.lr * frac


def _lam_at(it: int, adv_config: LogregAdversarialConfig) -> float:
    """Three phases, run in order: lam=0 for lam0_warmup_iters, then lam
    ramping linearly 0 -> lam over lam_warmup_iters, then lam held constant
    -- see LogregAdversarialConfig.lam0_warmup_iters/lam_warmup_iters."""
    if it < adv_config.lam0_warmup_iters:
        return 0.0
    ramp_it = it - adv_config.lam0_warmup_iters
    if ramp_it < adv_config.lam_warmup_iters:
        return adv_config.lam * ramp_it / adv_config.lam_warmup_iters
    return adv_config.lam


def _make_optimizer(
    params, adv_config: LogregAdversarialConfig
) -> torch.optim.Optimizer:
    """Construct the optimizer named by adv_config.optimizer_kind."""
    betas = (adv_config.adam_beta1, adv_config.adam_beta2)
    if adv_config.optimizer_kind == "adamw":
        return torch.optim.AdamW(
            params, lr=adv_config.lr, eps=adv_config.adam_eps, betas=betas
        )
    elif adv_config.optimizer_kind == "stableadamw":
        return StableAdamW(
            params,
            lr=adv_config.lr,
            eps=adv_config.adam_eps,
            betas=betas,
            d=adv_config.stableadamw_d,
        )
    else:
        raise ValueError(f"unknown optimizer_kind {adv_config.optimizer_kind!r}")


def _restore_checkpoint(ckpt_path: str, device, *, restore_optimizer: bool = True):
    """Load a full checkpoint -- architecture, weights, and iteration count,
    plus (if `restore_optimizer`) optimizer state -- shared by --resume
    (source: args.tag) and --fork-from (source: args.fork_from). Returns
    (model, opt, last_iter, best_loss, rck) so callers can pull any other
    checkpoint field they need (e.g. --resume rebuilds adv_config from rck).
    `last_iter` is the last iteration the checkpoint actually completed
    (0-indexed) -- a caller resuming training must pass `last_iter + 1` as
    the new loop's start, or it repeats that iteration.

    `restore_optimizer=False` for --fork-from: like every other adversarial
    hyperparameter (lr, adam_eps, adam_beta1, adam_beta2, ...), optimizer_kind is freshly
    resolved from the new --config rather than inherited from the source tag,
    so there's no single optimizer state (shape, momentum) that's guaranteed
    to still make sense -- the caller builds a fresh optimizer from that
    freshly-resolved adv_config instead. --resume, by contrast, keeps the same
    adv_config as the checkpoint, so restoring optimizer state is exact. This
    also means only restore_optimizer=True requires rck to have an
    `adv_config` key -- --fork-from can still restore an (architecture,
    weights, iter count) checkpoint that predates that field."""
    model, rck = ResidualMLP.load(ckpt_path, map_location=device)
    model = model.to(device)
    if restore_optimizer:
        if "adv_config" not in rck:
            raise SystemExit(
                f"[error] {ckpt_path} predates the nested adv_config checkpoint "
                f"layout and cannot be --resume'd. Start a fresh run, or "
                f"--fork-from it instead."
            )
        # The checkpoint's OWN historical config -- not the caller's
        # freshly-resolved adv_config, which for --fork-from may differ.
        hist_adv_config = LogregAdversarialConfig.from_dict(rck["adv_config"])
        opt = _make_optimizer(model.parameters(), hist_adv_config)
        opt.load_state_dict(rck["opt"])
    else:
        opt = None
    last_iter = rck["iter"]
    best_loss = rck.get("best_loss", float("inf"))
    return model, opt, last_iter, best_loss, rck


def _restore_rng_state(rck: dict, device) -> torch.Generator:
    """--resume counterpart to save_checkpoint's RNG snapshot: restores
    torch's global default generator (and CUDA's, if present) in place, and
    rebuilds `gen` -- the training loop's own generator -- from its saved
    state, so the interrupted run's RNG stream continues rather than
    reseeding a fresh one."""
    if "rng_state" not in rck:
        raise SystemExit(
            "[error] checkpoint predates RNG-state checkpointing and cannot "
            "be exactly --resume'd. Use --fork-from instead (reseeds fresh)."
        )
    # RNG state tensors must be CPU ByteTensors regardless of which device
    # they manage -- but ResidualMLP.load's map_location=device moves every
    # tensor in the checkpoint (these included) onto that device, so undo it.
    torch.set_rng_state(rck["rng_state"].cpu())
    if rck["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(rck["cuda_rng_state"].cpu())
    gen = torch.Generator(device=device)
    gen.set_state(rck["gen_state"].cpu())
    return gen


def _history_path(tag: str) -> str:
    return os.path.join(log_dir(tag), "history.jsonl")


def _read_history(path: str) -> list[dict]:
    """Read a history.jsonl file (one JSON object per line) into a list."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_history(path: str, entry: dict) -> None:
    """Append one entry to history.jsonl -- O(1) per call, unlike a full-file
    rewrite (see PR #98)."""
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _forked_history(source_tag: str, fork_iter: int) -> list[dict]:
    """The new tag's history.jsonl seed: source_tag's own history entries up
    to (inclusive of) the fork point, so loss curves stay continuous across
    the fork boundary for plotting -- new entries append after this as the
    new tag trains."""
    source_history = _read_history(_history_path(source_tag))
    return [h for h in source_history if h["iter"] <= fork_iter]


def concat_caches_torch(caches: list[torch.Tensor], layers: list[int]) -> torch.Tensor:
    return torch.cat([caches[lyr] for lyr in layers], dim=1)


def _trimmed_mean(s: torch.Tensor, trim_frac: float) -> torch.Tensor:
    """Mean of `s` after dropping its bottom/top `trim_frac` quantile tails
    (each), so a handful of extreme projected scores can't dominate the
    adversarial penalty. Quantile thresholds are computed without gradient
    (only which elements survive should be a discrete, non-differentiable
    choice); the mean over the survivors stays differentiable."""
    if trim_frac <= 0:
        return s.mean()
    with torch.no_grad():
        lo = torch.quantile(s, trim_frac)
        hi = torch.quantile(s, 1 - trim_frac)
        mask = (s >= lo) & (s <= hi)
    return s[mask].mean()


def score_penalty(
    cat_live: torch.Tensor,
    affine: tuple[torch.Tensor, torch.Tensor],
    label: torch.Tensor,
    kind: str,
    trim_frac: float = 0.0,
) -> torch.Tensor:
    """Differentiable adversarial penalty: project the live (grad-carrying),
    concatenated-across-layers activations onto the probe's current learned
    direction.

    "meandiff"/"meandiff-relu" push the two classes' (optionally
    trimmed-mean, see `_trimmed_mean`) scores together -- this needs the true
    label to separate the classes. "probe-report" instead mimics what
    training against a deployed probe actually looks like: the probe just
    reports "flagged" for any score past its decision threshold (0, since
    `affine` already folds the threshold into `b_eff`), regardless of which
    class it came from, so the penalty is every score's excess over that
    threshold, label-agnostic."""
    w_eff, b_eff = affine
    s = cat_live @ w_eff + b_eff
    if kind == "probe-report":
        return _trimmed_mean(torch.relu(s), trim_frac)
    gap = _trimmed_mean(s[label], trim_frac) - _trimmed_mean(s[~label], trim_frac)
    if kind == "meandiff-relu":
        return torch.relu(gap)
    elif kind == "meandiff":
        return gap
    else:
        raise ValueError(f"unknown probe_loss_kind: {kind!r}")


def clip_grad_norm_per_block_(blocks, max_norm: float) -> None:
    """Clip each block's gradients to `max_norm` by their own norm, fused
    across every block.

    Numerically the same as one `torch.nn.utils.clip_grad_norm_` per block
    (per-block, not whole-model -- see LogregAdversarialConfig.grad_clip), but
    every block's per-tensor norms come from a single multi-tensor kernel and
    every block's scale is applied in one more. The per-block loop it replaces
    was a separate launch sequence per block, which on launch-latency-bound
    hardware cost more than the clipping arithmetic itself. See PR #146.
    """
    per_block = [
        [p.grad for p in block.parameters() if p.grad is not None] for block in blocks
    ]
    widths = {len(g) for g in per_block}
    if not per_block or widths == {0}:
        return
    if len(widths) != 1:
        # Blocks disagree on how many gradients they have (some parameter
        # didn't get one), so the norms can't be reshaped into a rectangle.
        for block in blocks:
            torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm)
        return

    grads = [g for block_grads in per_block for g in block_grads]
    width = widths.pop()
    norms = torch.stack(torch._foreach_norm(grads)).view(len(per_block), width)
    block_norm = norms.square().sum(dim=1).sqrt()
    # Matching clip_grad_norm_'s own epsilon and clamp.
    coef = (max_norm / (block_norm + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, list(coef.repeat_interleave(width).unbind()))


@dataclass
class TrainRecord:
    """One completed training step, everything a caller needs to checkpoint,
    log, or resume from it."""

    iter: int
    loss: float
    l_task: float | None
    l_probe: float | None
    lam_eff: float | None
    lr: float
    affine: tuple[torch.Tensor, torch.Tensor]
    # Cumulative count of --explode-detected-and-corrected steps so far THIS
    # process invocation (see adv_config.explode_factor) -- resets to 0 on
    # --resume/--fork-from rather than continuing the source run's count,
    # same as e.g. the it/s rate below.
    n_exploded: int = 0


def _history_entry(record: TrainRecord, **extra) -> dict:
    """Build one `history.jsonl` entry from a `TrainRecord`, overridden/extended
    by `**extra` -- the single schema shared by the log-interval and final
    sites, rather than two hand-built dicts drifting independently."""
    d = dataclasses.asdict(record)
    del d["affine"]  # tensors aren't JSON-serializable, not needed in history
    d.update(extra)
    return d


def train_steps(
    model,
    opt,
    gen,
    probe,
    adv_config: LogregAdversarialConfig,
    max_iters: int,
    hidden_layers: list[int],
    start_iter: int,
    affine: tuple[torch.Tensor, torch.Tensor],
    probe_x: torch.Tensor,
    probe_label: torch.Tensor,
    device,
):
    """Generator over training iterations, yielding one `TrainRecord` per
    completed step (forward, probe update, backward, optimizer step). No
    checkpointing/logging here -- that's the caller's job, done between
    yields. This also means a KeyboardInterrupt while the caller is
    consuming this generator always leaves the caller's for-loop variable
    holding the last *fully completed* step, never a half-updated one.

    probe_x/probe_label: the probe dataset (sampled once by the caller) --
    reused for every probe fit and penalty until/unless
    adv_config.probe_resample_interval periodically redraws it, unlike the
    task batch below which resamples fresh each iteration."""
    num_x = model.config.num_x
    n_exploded = 0
    # Losses from the explode_window_iters-1 completed iterations before the
    # current one (post revert-and-retry if one happened), oldest first --
    # combined with the current iteration's own pre-step loss below, this
    # gives a window of explode_window_iters iterations total. See
    # adv_config.explode_window_iters.
    recent_losses: collections.deque[float] = collections.deque(
        maxlen=max(0, adv_config.explode_window_iters - 1)
    )
    # lam=0 means the probe penalty never enters the loss (see lam_eff below,
    # which is always 0 too regardless of --lam-warmup-iters) -- skip probe
    # resampling/refitting/scoring entirely rather than paying for a value
    # that gets multiplied by zero.
    skip_probe = adv_config.lam == 0

    def forward_loss(
        x_task: torch.Tensor,
        y: torch.Tensor,
        lam_eff: float,
        probe_x: torch.Tensor,
        probe_label: torch.Tensor,
        noise: torch.Tensor,
        probe_noise: torch.Tensor | None,
        *,
        retrain_probe: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One full forward at the model's current weights, returning
        (loss, l_task, l_probe).

        task: noisy pass -- this is what forbids shrinking c's encoding below
        the noise floor (see plans/resid_stream_noise_plan.md); `noise` is a
        pre-drawn blob (see `model.generate_noise`) rather than a generator,
        so callers can replay the identical noise across multiple calls
        within one iteration (see the explode-check/redo passes below). probe:
        one pass over the probe set feeds both the fit (when `retrain_probe`)
        and the penalty -- `probe_noise` (None unless adv_config.probe_noise,
        see there) puts that pass on the same noise floor the task pass uses,
        so a probe fit on noisy activations is also scored on noisy
        activations rather than mixing a noisy fit with a clean readout or
        vice versa (see adv_config.probe_noise's docstring for why that
        mismatch matters). Under lam=0 the probe pass is skipped entirely
        (l_probe is nan) rather than paying for a value that gets multiplied
        by zero.
        """
        nonlocal affine
        y_pred_full = model.forward(x_task, noise=noise)
        l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)
        if skip_probe:
            return l_task, l_task, torch.tensor(float("nan"))

        _, caches = model.forward(probe_x, return_cache=True, noise=probe_noise)
        cat_live = concat_caches_torch(caches, hidden_layers)
        if retrain_probe:
            X_fit = cat_live.detach()[:: adv_config.probe_subsample]  # no-op at 1
            label_fit = probe_label[:: adv_config.probe_subsample]
            assert label_fit.any() and (~label_fit).any(), (
                "subsampled probe batch has only one class present -- lower "
                "--probe-subsample or raise --batch-size."
            )
            fit_probe(probe, X_fit, label_fit, PROBE_STEP_MAX_ITER)
            affine = probe.get_affine(device)

        l_probe = score_penalty(
            cat_live,
            affine,
            probe_label,
            adv_config.probe_loss_kind,
            adv_config.probe_loss_trim_frac,
        )
        return lam_eff * l_probe + (1 - lam_eff) * l_task, l_task, l_probe

    def optimizer_step(loss: torch.Tensor, grad_clip: float) -> None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_per_block_(model.blocks, grad_clip)
        # grad_clip/adam_eps/adam_beta2 are band-aids for instability;
        # optimizer_kind="stableadamw" (update clipping) is a cleaner fix --
        # see PR #77 for the original discussion. Whether grad_clip/adam_eps
        # /adam_beta2 can now be relaxed under stableadamw is still an open,
        # empirical question (not addressed by adding the option itself).
        opt.step()

    for it in range(start_iter, max_iters):
        x_task, y = sample_batch(
            adv_config.batch_size,
            num_x,
            generator=gen,
            device=device,
            x_p_outer=adv_config.x_p_outer,
            x_threshold=adv_config.x_threshold,
        )
        if (
            not skip_probe
            and adv_config.probe_resample_interval > 0
            and it % adv_config.probe_resample_interval == 0
        ):
            probe_x, _ = sample_batch(
                adv_config.batch_size, num_x, generator=gen, device=device
            )
            probe_label = probe_x[:, num_x] >= adv_config.class_threshold
            assert probe_label.any() and (~probe_label).any(), (
                "resampled probe batch has only one class present -- check "
                "--class-threshold against c's range."
            )
        lam_eff = _lam_at(it, adv_config)

        lr_eff = _lr_at(it, max_iters, adv_config)
        for group in opt.param_groups:
            group["lr"] = lr_eff

        # Drawn once per iteration and reused verbatim across every
        # forward_loss call below (initial, explode-check, explode-redo) --
        # an explicit, replayable blob instead of snapshotting/resetting
        # `gen`'s RNG state around the draw (see plans/model_noise_blob_plan.md).
        # probe_noise is a separate draw, same std, same reuse-across-calls
        # reasoning -- it's for the probe's forward, not the task pass `noise`
        # covers, and there's no reason to correlate the two. Only drawn when
        # adv_config.probe_noise is set, so a disabled probe_noise consumes
        # no extra RNG state and stays bit-identical to a pre-probe_noise run.
        noise = model.generate_noise(
            adv_config.batch_size, adv_config.resid_noise_std, gen
        )
        probe_noise = (
            model.generate_noise(adv_config.batch_size, adv_config.resid_noise_std, gen)
            if adv_config.probe_noise
            else None
        )
        loss, l_task, l_probe = forward_loss(
            x_task,
            y,
            lam_eff,
            probe_x,
            probe_label,
            noise,
            probe_noise,
            retrain_probe=it % adv_config.probe_retrain_interval == 0,
        )
        # Snapshot BEFORE the step, so a detected explosion (below) can
        # revert to it. TODO(perf): every-iteration deepcopy + re-forward
        # just to catch a rare event -- a cheaper retroactive alternative
        # exists, see PR #77's revert-and-retry discussion (not the
        # StableAdamW note in optimizer_step).
        #
        # loss.item() is read here rather than unconditionally: it syncs the
        # host against the GPU, and nothing outside the explode path wants
        # the pre-step loss.
        if adv_config.explode_factor > 0:
            loss_before_step = loss.item()
            pre_model_state = copy.deepcopy(model.state_dict())
            pre_opt_state = copy.deepcopy(opt.state_dict())

        optimizer_step(loss, adv_config.grad_clip)

        if adv_config.explode_factor > 0:
            with torch.no_grad():
                loss_after, _, _ = forward_loss(
                    x_task,
                    y,
                    lam_eff,
                    probe_x,
                    probe_label,
                    noise,
                    probe_noise,
                    retrain_probe=False,
                )
                loss_after_step = loss_after.item()

            # The smallest loss over the last explode_window_iters completed
            # iterations, plus this iteration's own pre-step loss -- catches
            # both a single-step spike and gradual creep across several
            # steps (see adv_config.explode_window_iters).
            baseline_loss = min([loss_before_step] + list(recent_losses))

            # Also require the step to have made this iteration's own loss
            # worse, not just left it above the historical window baseline --
            # otherwise a step that's recovering from an earlier arrested
            # spike (elevated relative to baseline_loss but still improving)
            # gets needlessly re-clipped.
            if (
                loss_after_step > adv_config.explode_factor * baseline_loss
                and loss_after_step > loss_before_step
            ):
                n_exploded += 1
                print(
                    f"[explode] iter={it} loss {baseline_loss:.3e} -> "
                    f"{loss_after_step:.3e} ({loss_after_step / baseline_loss:.1f}x)"
                    f" -- reverting and redoing with a tighter clip"
                )
                model.load_state_dict(pre_model_state)
                opt.load_state_dict(pre_opt_state)

                # Fresh forward: the previous graph was freed by backward()
                # above, and this is otherwise numerically the same
                # pre-step state already used for l_task/l_probe/loss.
                loss, l_task, l_probe = forward_loss(
                    x_task,
                    y,
                    lam_eff,
                    probe_x,
                    probe_label,
                    noise,
                    probe_noise,
                    retrain_probe=False,
                )
                optimizer_step(
                    loss, adv_config.grad_clip / adv_config.explode_clip_divisor
                )

            recent_losses.append(loss.item())

        yield TrainRecord(
            iter=it,
            loss=loss.item(),
            l_task=float(l_task.item()),
            l_probe=float(l_probe.item()),
            lam_eff=lam_eff,
            lr=lr_eff,
            affine=affine,
            n_exploded=n_exploded,
        )


class _Interruptible:
    """Records whether a SIGINT arrived while it was installed."""

    def __init__(self):
        self.interrupted = False


@contextmanager
def _defer_keyboard_interrupt():
    """Ignore SIGINT for the duration of the wrapped block, then re-raise it
    (as KeyboardInterrupt) on the way out. Yields an object whose
    `.interrupted` flag lets the block stop at a point of its own choosing --
    a Ctrl-C therefore never lands mid-write, however hard it is spammed.

    The flip side: nothing inside the block can be interrupted promptly, so
    wrap only code that reaches a safe stopping point quickly. Genuinely
    wedged code needs SIGQUIT/SIGKILL instead."""
    state = _Interruptible()
    old_handler = signal.getsignal(signal.SIGINT)

    def _handler(signum, frame):
        state.interrupted = True

    signal.signal(signal.SIGINT, _handler)
    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, old_handler)
    if state.interrupted:
        raise KeyboardInterrupt


def _atomic_write(write_fn, path: str) -> None:
    """Run `write_fn(tmp_path)` and rename the result over `path`, so `path`
    is always either the previous complete file or the new one."""
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def _point_symlink(link_path: str, target_name: str) -> None:
    """Atomically (re)point `link_path` at a sibling file. Replaces whatever
    is already there, including the plain file older runs wrote. The target is
    stored relative, so the run directory stays relocatable."""
    tmp = link_path + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target_name, tmp)
    os.replace(tmp, link_path)


def save_checkpoint(
    path, record: TrainRecord, model, opt, best_loss, hidden_layers, adv_config, gen
):
    """Save all training state needed to resume from disk. Includes every
    RNG's state (`gen` -- the training loop's own generator -- plus torch's
    global default generator, which model-init and any generator-less
    randomness still draw from) so --resume can continue the interrupted
    run's RNG stream in place, rather than reseeding a fresh one.

    Callers are responsible for crash-safety: wrap in `_atomic_write` (so a
    kill can't leave a half-written file at `path`) and in
    `_defer_keyboard_interrupt` (so a Ctrl-C can't split the save from the
    bookkeeping that goes with it)."""
    w_eff, b_eff = record.affine
    model.save(
        path,
        iter=record.iter,
        opt=opt.state_dict(),
        best_loss=best_loss,
        probe_w=w_eff.cpu(),
        probe_b=b_eff.cpu(),
        probe_layers=hidden_layers,
        adv_config=adv_config.to_dict(),
        rng_state=torch.get_rng_state(),
        cuda_rng_state=(
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
        gen_state=gen.get_state(),
    )


def checkpoint_path(run_ckpt_dir: str, iter: int) -> str:
    return os.path.join(run_ckpt_dir, f"iter_{iter}.pt")


def write_checkpoint(run_ckpt_dir: str, iter: int, save_fn) -> str:
    """Write `iter_<iter>.pt` and repoint `last.pt` at it. Every numbered
    checkpoint is kept permanently.

    Callers are responsible for deferring SIGINT around this (see
    `_defer_keyboard_interrupt`)."""
    last_path = os.path.join(run_ckpt_dir, "last.pt")
    path = checkpoint_path(run_ckpt_dir, iter)
    _atomic_write(save_fn, path)
    _point_symlink(last_path, os.path.basename(path))
    return path


def _start_resume(args, last_path: str, device):
    """--resume's full run-state restore: everything main() needs is either
    inherited from the checkpoint (model/opt/adv_config/hidden_layers) or
    continued in place (gen's RNG state) -- none of the shared fresh-run
    setup (seeding, load_run_config, _make_optimizer, _write_run_config)
    applies here. Returns (model, opt, start_iter, best_loss, adv_config,
    hidden_layers, gen)."""
    # (raises if last_path predates the nested adv_config checkpoint layout
    # -- see _restore_checkpoint)
    model, opt, last_iter, best_loss, rck = _restore_checkpoint(last_path, device)
    adv_config = LogregAdversarialConfig.from_dict(rck["adv_config"])
    hidden_layers = adv_config.penalty_layers
    gen = _restore_rng_state(rck, device)
    # history.jsonl is append-only: this run's earlier entries are already on
    # disk, so resuming needs no read -- new entries just keep appending to
    # the same file.
    start_iter = last_iter + 1
    print(f"[resume] from iter {start_iter}, best_loss={best_loss:.3e}")
    _check_config_json(args.tag, model.config, adv_config)
    return model, opt, start_iter, best_loss, adv_config, hidden_layers, gen


def _start_fork_from(args, hist_path: str, device):
    """Restore (architecture, weights, iteration count) from
    args.fork_from's checkpoint for a new tag -- the adversarial-objective
    hyperparameters/optimizer are NOT restored here; the caller resolves
    those fresh from --config like a scratch run. Returns (model,
    start_iter, best_loss, forked_from)."""
    source_ckpt_path = os.path.join(ckpt_dir(args.fork_from), "last.pt")
    model, _, last_iter, best_loss, rck = _restore_checkpoint(
        source_ckpt_path, device, restore_optimizer=False
    )
    # One-time seed write (not per-iteration, so no quadratic cost): copy
    # source_tag's pre-fork entries into the new tag's own history.jsonl so
    # loss curves stay continuous across the fork.
    for h in _forked_history(args.fork_from, last_iter):
        _append_history(hist_path, h)
    forked_from = ForkedFrom(tag=args.fork_from, iter=last_iter)
    start_iter = last_iter + 1
    print(f"[fork] from {args.fork_from} @ iter {last_iter}, best_loss={best_loss:.3e}")
    return model, start_iter, best_loss, forked_from


def main(args):
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_backend = resolve_probe_backend(args.probe_backend, device)

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

    last_path = os.path.join(run_ckpt_dir, "last.pt")
    best_path = os.path.join(run_ckpt_dir, "best.pt")
    hist_path = _history_path(args.tag)

    if args.resume:
        model, opt, start_iter, best_loss, adv_config, hidden_layers, gen = (
            _start_resume(args, last_path, device)
        )
    elif args.fork_from is not None:
        model, start_iter, best_loss, forked_from = _start_fork_from(
            args, hist_path, device
        )
    else:
        model_config = ResidualMLPConfig(
            num_x=args.num_x,
            d_model=args.d_model,
            d_mlp=args.d_mlp,
            num_blocks=args.num_blocks,
        )
        model = ResidualMLP(model_config).to(device)
        start_iter = 0
        best_loss = float("inf")
        forked_from = None

    if not args.resume:
        torch.manual_seed(args.seed)
        gen = torch.Generator(device=device).manual_seed(args.seed + 1)
        if args.fork_from is None:
            # Re-seeded, deliberate re-init -- the ctor's own (unseeded) init
            # above is discarded, same as --resume/--fork-from's own ctor-time
            # init is discarded by their (already-run) load_state_dict.
            model.reset_parameters()
            print(f"[init] scratch model cfg={model_config}")

        # Hyperparameters are freshly resolved from --config (never read from
        # the checkpoint, unlike --resume above).
        adv_config, hidden_layers = load_run_config(
            _read_config_file(args.config),
            num_blocks=model.config.num_blocks,
            config_path=args.config,
        )
        # Always freshly built (never inherited from a --fork-from source),
        # same as every other adversarial hyperparameter above -- see
        # `_restore_checkpoint`'s restore_optimizer=False note.
        opt = _make_optimizer(model.parameters(), adv_config)
        _write_run_config(
            args.tag,
            model.config,
            adv_config,
            input_config_path=args.config,
            forked_from=forked_from,
            seed=args.seed,
        )

    num_x, num_blocks = model.config.num_x, model.config.num_blocks
    if not hidden_layers:
        raise SystemExit(
            f"[error] no penalty layers (num_blocks={num_blocks} has no hidden "
            f"layers). Nothing to hide against."
        )

    # --- sample the fixed probe dataset (reused for every fit + penalty for
    # the rest of the run -- see module docstring) and init-fit a probe over
    # the concatenation of penalized layers ---
    probe_x, _ = sample_batch(
        adv_config.batch_size, num_x, generator=gen, device=device
    )
    probe_label = probe_x[:, num_x] >= adv_config.class_threshold
    assert probe_label.any() and (~probe_label).any(), (
        "fixed probe batch has only one class present -- check --class-threshold "
        "against c's range."
    )
    with torch.no_grad():
        _, init_caches = model.forward(probe_x, return_cache=True)
    cat_init = concat_caches_torch(init_caches, hidden_layers)
    probe = build_probe_pipeline(
        adv_config.probe_C, adv_config.probe_init_iters, probe_backend
    )
    probe.fit(cat_init.detach(), probe_label)
    affine = probe.get_affine(device)
    print(
        f"[init] fit concatenated probe (backend={probe_backend}) over layers "
        f"{hidden_layers}, init_iters={adv_config.probe_init_iters}, "
        f"C={adv_config.probe_C}"
    )

    print(
        f"[adv] tag={args.tag} lam={adv_config.lam} penalty_layers={hidden_layers} "
        f"num_blocks={num_blocks} bs={adv_config.batch_size} "
        f"class_threshold={adv_config.class_threshold} "
        f"probe_loss_kind={adv_config.probe_loss_kind} "
        f"probe_backend={probe_backend} probe_subsample={adv_config.probe_subsample} "
        f"probe_retrain_interval={adv_config.probe_retrain_interval} "
        f"probe_resample_interval={adv_config.probe_resample_interval} "
        f"probe_loss_trim_frac={adv_config.probe_loss_trim_frac} "
        f"resid_noise_std={adv_config.resid_noise_std} probe_noise={adv_config.probe_noise} "
        f"grad_clip={adv_config.grad_clip} "
        f"lr={adv_config.lr} lr_warmup_iters={adv_config.lr_warmup_iters} "
        f"lr_min_frac={adv_config.lr_min_frac} adam_eps={adv_config.adam_eps} "
        f"adam_beta1={adv_config.adam_beta1} adam_beta2={adv_config.adam_beta2} "
        f"optimizer_kind={adv_config.optimizer_kind} "
        f"stableadamw_d={adv_config.stableadamw_d} "
        f"explode_factor={adv_config.explode_factor} "
        f"explode_clip_divisor={adv_config.explode_clip_divisor} "
        f"device={device} iters {start_iter}->{args.max_iters}"
    )

    # Placeholder record for the (edge-case) zero-iteration run, e.g.
    # --resume past --max-iters: train_steps() then yields nothing and there
    # is nothing to save (the run's existing last.pt already describes iter
    # start_iter - 1), but the closing summary still reports off a record.
    record = TrainRecord(
        iter=start_iter - 1,
        loss=best_loss,
        l_task=None,
        l_probe=None,
        lam_eff=None,
        lr=adv_config.lr,
        affine=affine,
    )

    def save(path):
        save_checkpoint(
            path, record, model, opt, best_loss, hidden_layers, adv_config, gen
        )

    t0 = time.time()
    rate_meter = EMARateMeter(start_iter, window=args.rate_meter_window)
    max_err = None
    last_logged_iter = None

    def log_iter():
        """Append a history entry for `record`'s iteration and print it."""
        nonlocal max_err, last_logged_iter
        max_err = eval_max_err(model, gen, device=device)
        _append_history(hist_path, _history_entry(record, max_err=max_err))
        last_logged_iter = record.iter
        rate = rate_meter.update(record.iter)
        print(
            f"iter {record.iter:>6d}  loss {record.loss:.3e}  task {record.l_task:.3e}  "
            f"probe {record.l_probe:.3e}  max_err {max_err:.3e}  "
            f"n_exploded {record.n_exploded}  {rate:.1f} it/s"
        )

    def checkpoint():
        return write_checkpoint(run_ckpt_dir, record.iter, save)

    ran_any_iters = False
    # SIGINT is deferred for the whole loop, so a Ctrl-C (however hard it is
    # spammed) only takes effect at the end of an iteration, after the saves
    # below have run to completion. Every write in here is covered by this one
    # block -- nothing inside needs its own.
    with _defer_keyboard_interrupt() as sigint:
        for record in train_steps(
            model,
            opt,
            gen,
            probe,
            adv_config,
            args.max_iters,
            hidden_layers,
            start_iter,
            affine,
            probe_x,
            probe_label,
            device,
        ):
            ran_any_iters = True
            if record.loss < best_loss:
                best_loss = record.loss
                _atomic_write(save, best_path)

            if record.iter % args.log_interval == 0:
                log_iter()

            if record.iter % args.ckpt_interval == 0 and record.iter > start_iter:
                checkpoint()

            if sigint.interrupted:
                print(
                    f"\n[interrupt] Ctrl-C, stopping after iter {record.iter}...",
                    flush=True,
                )
                break

        # Checkpoint wherever the run actually stopped -- the end of
        # --max-iters, or the interrupted iteration. That iteration is
        # generally not a --log-interval multiple, so log it first; this is
        # what establishes the two history/checkpoint invariants described at
        # the top of this module.
        if ran_any_iters:
            if last_logged_iter != record.iter:
                log_iter()
            print(f"[save] {last_path} -> {os.path.basename(checkpoint())}")
    # _defer_keyboard_interrupt re-raises here if a Ctrl-C arrived above.

    max_err_str = "n/a" if max_err is None else f"{max_err:.3e}"
    print(
        f"[done] iter {record.iter}  best_loss {best_loss:.3e}  final max_err {max_err_str}  "
        f"elapsed {time.time()-t0:.1f}s"
    )
    print(f"[done] checkpoints in {run_ckpt_dir}, history in {hist_path}")
    print(f"[next] python adversarial_report.py --tag {args.tag}")


if __name__ == "__main__":
    main(args)
