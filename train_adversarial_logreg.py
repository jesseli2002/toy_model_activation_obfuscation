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
    `--probe-backend`, `--log-interval`, `--ckpt-interval`, `--save-every-n`,
    `--max-iters`) sit outside this split entirely: they're run control that
    never varies within one tag's lineage and is never persisted to
    `config.json`.
"""

import argparse
import collections
import copy
import dataclasses
import json
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
    g_book.add_argument(
        "--save-every-n",
        type=int,
        nargs="?",
        const=-1,
        default=-1,
        help=(
            "also save a numbered snapshot checkpoint every N iters "
            "(-1 = --ckpt-interval, 0 = disable)"
        ),
    )
    g_book.add_argument("--max-iters", type=int, default=config.MAX_ITERS)

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
    (model, opt, start_iter, best_loss, rck) so callers can pull any other
    checkpoint field they need (e.g. --resume rebuilds adv_config from rck).

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
    start_iter = rck["iter"]
    best_loss = rck.get("best_loss", float("inf"))
    return model, opt, start_iter, best_loss, rck


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
    torch.set_rng_state(rck["rng_state"])
    if rck["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(rck["cuda_rng_state"])
    gen = torch.Generator(device=device)
    gen.set_state(rck["gen_state"])
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


@dataclass
class TrainRecord:
    """One completed training step, everything a caller needs to checkpoint,
    log, or resume from it."""

    iter: int
    loss: float
    l_task: float | None
    l_probe: float | None
    lam_eff: float | None
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
        clean pass over the probe set, full resolution, so the probe can
        still out-resolve the model; `retrain_probe` first advances the probe
        on those activations and updates `affine`. Under lam=0 the probe pass
        is skipped entirely (l_probe is nan) rather than paying for a value
        that gets multiplied by zero.
        """
        nonlocal affine
        y_pred_full = model.forward(x_task, noise=noise)
        l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)
        if skip_probe:
            return l_task, l_task, torch.tensor(float("nan"))

        _, caches = model.forward(probe_x, return_cache=True)
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
            # Per-block, not whole-model -- see LogregAdversarialConfig.grad_clip.
            for block in model.blocks:
                torch.nn.utils.clip_grad_norm_(block.parameters(), grad_clip)
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
        if adv_config.lam_warmup_iters > 0:
            lam_eff = adv_config.lam * min(1.0, it / adv_config.lam_warmup_iters)
        else:
            lam_eff = adv_config.lam

        # Drawn once per iteration and reused verbatim across every
        # forward_loss call below (initial, explode-check, explode-redo) --
        # an explicit, replayable blob instead of snapshotting/resetting
        # `gen`'s RNG state around the draw (see plans/model_noise_blob_plan.md).
        noise = model.generate_noise(
            adv_config.batch_size, adv_config.resid_noise_std, gen
        )
        loss, l_task, l_probe = forward_loss(
            x_task,
            y,
            lam_eff,
            probe_x,
            probe_label,
            noise,
            retrain_probe=it % adv_config.probe_retrain_interval == 0,
        )
        loss_before_step = loss.item()

        # Snapshot BEFORE the step, so a detected explosion (below) can
        # revert to it. TODO(perf): every-iteration deepcopy + re-forward
        # just to catch a rare event -- a cheaper retroactive alternative
        # exists, see PR #77's revert-and-retry discussion (not the
        # StableAdamW note in optimizer_step).
        if adv_config.explode_factor > 0:
            pre_model_state = copy.deepcopy(model.state_dict())
            pre_opt_state = copy.deepcopy(opt.state_dict())

        optimizer_step(loss, adv_config.grad_clip)

        if adv_config.explode_factor > 0:
            with torch.no_grad():
                loss_after, _, _ = forward_loss(
                    x_task, y, lam_eff, probe_x, probe_label, noise, retrain_probe=False
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
                    x_task, y, lam_eff, probe_x, probe_label, noise, retrain_probe=False
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
            affine=affine,
            n_exploded=n_exploded,
        )


@contextmanager
def _defer_keyboard_interrupt():
    """Ignore SIGINT for the duration of the wrapped block, then re-raise it
    (as KeyboardInterrupt) immediately after -- so a Ctrl-C during the block
    can't leave a half-written checkpoint on disk."""
    interrupted = False
    old_handler = signal.getsignal(signal.SIGINT)

    def _handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)
    if interrupted:
        raise KeyboardInterrupt


def save_checkpoint(
    path, record: TrainRecord, model, opt, best_loss, hidden_layers, adv_config, gen
):
    """Save all training state needed to resume from disk, atomically (a
    SIGINT can't corrupt the file). Includes every RNG's state (`gen` -- the
    training loop's own generator -- plus torch's global default generator,
    which model-init and any generator-less randomness still draw from) so
    --resume can continue the interrupted run's RNG stream in place, rather
    than reseeding a fresh one."""
    w_eff, b_eff = record.affine
    with _defer_keyboard_interrupt():
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


def main(args):
    if args.save_every_n == -1:
        args.save_every_n = args.ckpt_interval
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
        # (raises if last_path predates the nested adv_config checkpoint
        # layout -- see _restore_checkpoint)
        model, opt, start_iter, best_loss, rck = _restore_checkpoint(last_path, device)
        adv_config = LogregAdversarialConfig.from_dict(rck["adv_config"])
        hidden_layers = adv_config.penalty_layers
        gen = _restore_rng_state(rck, device)
        # history.jsonl is append-only: this run's earlier entries are
        # already on disk, so resuming needs no read -- new entries just
        # keep appending to the same file.
        print(f"[resume] from iter {start_iter}, best_loss={best_loss:.3e}")
        _check_config_json(args.tag, model.config, adv_config)
    elif args.fork_from is not None:
        source_ckpt_path = os.path.join(ckpt_dir(args.fork_from), "last.pt")
        model, _, start_iter, best_loss, rck = _restore_checkpoint(
            source_ckpt_path, device, restore_optimizer=False
        )
        # One-time seed write (not per-iteration, so no quadratic cost):
        # copy source_tag's pre-fork entries into the new tag's own
        # history.jsonl so loss curves stay continuous across the fork.
        for h in _forked_history(args.fork_from, start_iter):
            _append_history(hist_path, h)
        forked_from = ForkedFrom(tag=args.fork_from, iter=start_iter)
        print(
            f"[fork] from {args.fork_from} @ iter {start_iter}, "
            f"best_loss={best_loss:.3e}"
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
        f"resid_noise_std={adv_config.resid_noise_std} grad_clip={adv_config.grad_clip} "
        f"lr={adv_config.lr} adam_eps={adv_config.adam_eps} "
        f"adam_beta1={adv_config.adam_beta1} adam_beta2={adv_config.adam_beta2} "
        f"optimizer_kind={adv_config.optimizer_kind} "
        f"stableadamw_d={adv_config.stableadamw_d} "
        f"explode_factor={adv_config.explode_factor} "
        f"explode_clip_divisor={adv_config.explode_clip_divisor} "
        f"device={device} iters {start_iter}->{args.max_iters}"
    )

    # Placeholder record for the (edge-case) zero-iteration run, e.g.
    # --resume past --max-iters: train_steps() then yields nothing, and the
    # final save/log below still needs a record to work with.
    record = TrainRecord(
        iter=start_iter,
        loss=best_loss,
        l_task=None,
        l_probe=None,
        lam_eff=None,
        affine=affine,
    )

    def save(path):
        save_checkpoint(
            path, record, model, opt, best_loss, hidden_layers, adv_config, gen
        )

    t0 = time.time()
    rate_meter = EMARateMeter(start_iter)
    try:
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
            if record.loss < best_loss:
                best_loss = record.loss
                save(best_path)

            if record.iter % args.log_interval == 0:
                me = eval_max_err(model, gen, device=device)
                _append_history(hist_path, _history_entry(record, max_err=me))
                rate = rate_meter.update(record.iter)
                print(
                    f"iter {record.iter:>6d}  loss {record.loss:.3e}  task {record.l_task:.3e}  "
                    f"probe {record.l_probe:.3e}  max_err {me:.3e}  "
                    f"n_exploded {record.n_exploded}  {rate:.1f} it/s"
                )

            if record.iter % args.ckpt_interval == 0 and record.iter > start_iter:
                save(last_path)

            if (
                args.save_every_n != 0  # i.e. not disabled
                and record.iter % args.save_every_n == 0
                and record.iter > start_iter
            ):
                save(os.path.join(run_ckpt_dir, f"iter_{record.iter}.pt"))
    except KeyboardInterrupt:
        print(
            f"\n[interrupt] KeyboardInterrupt caught, saving checkpoint at iter {record.iter}..."
        )
        save(last_path)
        print(f"[interrupt] saved to {last_path}")
        raise

    # final logging + save
    save(last_path)
    me = eval_max_err(model, gen, device=device)
    _append_history(
        hist_path,
        _history_entry(
            record, loss=best_loss, l_task=None, l_probe=None, max_err=me, final=True
        ),
    )
    print(
        f"[done] iter {record.iter}  best_loss {best_loss:.3e}  final max_err {me:.3e}  "
        f"elapsed {time.time()-t0:.1f}s"
    )
    print(f"[done] checkpoints in {run_ckpt_dir}, history in {hist_path}")
    print(f"[next] python adversarial_report.py --tag {args.tag}")


if __name__ == "__main__":
    main(args)
