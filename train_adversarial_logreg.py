"""Step 3 (variant) — adversarial training: model vs. a stateful
LogisticRegression probe, trained *simultaneously* with the model.

Unlike train_adversarial.py's closed-form DoM/LDA penalty (recomputed from
scratch each step, no inner probe optimizer), this script keeps a single
LogisticRegression probe over the concatenation of all penalized hidden
layers, warm-started once against the initial model and then advanced a few
solver iterations per training step -- an actual adversary that tracks the
model as it moves, rather than a static closed-form proxy for one. The probe
backend is `sklearn.linear_model.LogisticRegression` (CPU) or a GPU-resident
torch reimplementation, selected via `--probe-backend` (see probe_backend.py).

The training objective combines a probe-adversarial penalty with the task
loss (see LogregAdversarialConfig for the weighting and probe hyperparameters).
The probe (both its own fit and the differentiable penalty) is always
evaluated against one fixed input batch sampled before the training loop
starts -- the task loss, by contrast, resamples a fresh batch every step.

The interesting science is the same as train_adversarial.py: not "can it hide
c" but HOW -- does it hide c only at the probed threshold ("hidden"), or
genuinely erase linear c-information across the range ("erased")? A moving,
adaptive probe is a strictly harder adversary than the closed-form penalty,
so this is a stress test of whatever hiding the DoM/LDA runs found.

This is NOT gated. The deliverable is the trained checkpoint + diagnostics;
run once, then stop and review.

Three run modes (see plans/rare_flags_config_plan.md for the full design):
  - fresh run: hyperparameters are freshly resolved from `--config`'s JSON
    file plus CLI-common flags. Writes `runs/<tag>/config.json` once.
  - `--resume <tag>`: strictly continues the same experiment. All
    hyperparameters (CLI-common and config-file) are restored from the
    checkpoint, never re-read from `--config` or CLI. `runs/<tag>/config.json`
    is read once for a sanity check against the restored config (a mismatch
    warns but never rewrites the file) -- not for resolving hyperparameters.
  - `--fork-from <source_tag>` (with `--tag <new_tag>`): branches a new
    experiment off `source_tag`'s checkpoint -- architecture, weights,
    optimizer, and iteration count all come from there, same as `--resume` --
    but the adversarial-objective hyperparameters are freshly resolved
    exactly like a fresh run. `runs/<new_tag>/config.json` additionally
    records `forked_from`.

Two run-directory artifacts are written once, at tag-creation time, and are
read-only thereafter for the lifetime of that tag (see `LogregRunConfig` for
`config.json`'s schema): `runs/<tag>/input_config.json` is a verbatim copy of
the `--config` file, kept reusable as a later `--config` argument (e.g. for
`--fork-from`); `runs/<tag>/config.json` is the fully-resolved run config.
Hand-editing either after creation has no effect except tripping the
mismatch warning on a later `--resume`.
"""

import argparse
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

    g_adv = p.add_argument_group(
        "adversarial objective (CLI-common)",
        "Touched often enough to stay CLI flags; every other hyperparameter "
        "lives in --config's JSON file instead -- see LogregAdversarialConfig.",
    )
    g_adv.add_argument(
        "--lam",
        type=float,
        default=LogregAdversarialConfig.lam,
        help="convex-combination weight: loss = lam * L_probe + (1-lam) * L_task. "
        "lam=1 optimizes purely for hiding c (task loss ignored). lam=0 is "
        "plain task training.",
    )

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
        "--fork-from",
        type=str,
        default=None,
        metavar="SOURCE_TAG",
        help="branch a new --tag off SOURCE_TAG's latest checkpoint (architecture, "
        "weights, optimizer state, iteration count, and history -- continued for "
        "continuous loss curves). Unlike --resume, the adversarial-objective "
        "hyperparameters are freshly resolved from this invocation's "
        "--config/--lam, not inherited from SOURCE_TAG. Mutually exclusive "
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
        "available, else sklearn. 'sklearn'/'torch' force a backend "
        "regardless of device -- e.g. to smoke-test the torch backend on a "
        "CPU-only machine.",
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
    load_run_config, once lam is known too."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[error] --config file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[error] --config {path} is not valid JSON: {e}")


def load_run_config(
    file_fields: dict, *, lam: float, num_blocks: int, config_path: str
) -> tuple[LogregAdversarialConfig, list[int]]:
    """Merge --config's file_fields with this invocation's --lam into one
    LogregAdversarialConfig, resolving `penalty_layers` (a config-file value
    of "all" or an explicit list) against num_blocks first -- so a missing/
    invalid `penalty_layers` surfaces the same as any other bad --config key.
    A missing required key in the file surfaces as the dataclass
    constructor's TypeError; re-raised here as a SystemExit naming the
    specific key(s), per this module's [error] convention. Returns
    (adv_config, hidden_layers) -- the latter is what callers actually index
    caches with."""
    fields = dict(file_fields)
    if "penalty_layers" in fields:
        fields["penalty_layers"] = _resolve_hidden_layers(
            fields["penalty_layers"], num_blocks
        )
    try:
        adv_config = LogregAdversarialConfig(lam=lam, **fields)
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
    forked_from: ForkedFrom | None = None,
) -> None:
    """Write runs/<tag>/config.json (fully resolved) and
    runs/<tag>/input_config.json (verbatim copy of --config) once, at
    tag-creation time (fresh run or --fork-from). Write-once, read-only
    thereafter -- see module docstring; --resume never calls this."""
    run_config = LogregRunConfig(
        model=model_config, adversarial=adv_config, forked_from=forked_from
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


def _restore_checkpoint(ckpt_path: str, device):
    """Load a full checkpoint -- architecture, weights, optimizer state, and
    iteration count -- shared by --resume (source: args.tag) and --fork-from
    (source: args.fork_from). The two differ only in WHICH tag's checkpoint
    feeds this, not in what gets restored here. Returns (model, opt,
    start_iter, best_loss, rck) so callers can pull any other checkpoint
    field they need (e.g. --resume rebuilds adv_config from rck)."""
    model, rck = ResidualMLP.load(ckpt_path, map_location=device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)  # lr/eps/betas set by caller
    opt.load_state_dict(rck["opt"])
    start_iter = rck["iter"]
    best_loss = rck.get("best_loss", float("inf"))
    return model, opt, start_iter, best_loss, rck


def _forked_history(source_tag: str, fork_iter: int) -> list[dict]:
    """The new tag's history.json seed: source_tag's own history entries up
    to (inclusive of) the fork point, so loss curves stay continuous across
    the fork boundary for plotting -- new entries append after this as the
    new tag trains."""
    source_hist_path = os.path.join(log_dir(source_tag), "history.json")
    if not os.path.exists(source_hist_path):
        return []
    with open(source_hist_path) as f:
        source_history = json.load(f)
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
    direction and push the two classes' (optionally trimmed-mean, see
    `_trimmed_mean`) scores together."""
    w_eff, b_eff = affine
    s = cat_live @ w_eff + b_eff
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
    probe_dt: float
    model_dt: float
    # Cumulative count of --explode-detected-and-corrected steps so far THIS
    # process invocation (see adv_config.explode_factor) -- resets to 0 on
    # --resume/--fork-from rather than continuing the source run's count,
    # same as e.g. the it/s rate below.
    n_exploded: int = 0


def _history_entry(record: TrainRecord, **extra) -> dict:
    """Build one `history.json` entry from a `TrainRecord`, overridden/extended
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
    for it in range(start_iter, max_iters):
        t_fwd0 = time.time()
        x_task, y = sample_batch(
            adv_config.batch_size,
            num_x,
            generator=gen,
            device=device,
            x_p_outer=adv_config.x_p_outer,
            x_threshold=adv_config.x_threshold,
        )

        # Snapshotted right before the noise draw below, so a detected
        # explosion (below) can replay the IDENTICAL noise on the
        # check/redo forwards by resetting to this state first, instead of
        # consuming extra draws from `gen` that would desync every later
        # iteration's noise from a run with --explode-factor disabled.
        noise_gen_state = gen.get_state()
        # task: noisy pass -- this is what forbids shrinking c's encoding
        # below the noise floor (see plans/resid_stream_noise_plan.md).
        y_pred_full = model.forward(
            x_task, noise_std=adv_config.resid_noise_std, generator=gen
        )
        l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)

        # probe fit + penalty: clean pass over the probe set (fixed unless
        # --probe-resample-interval periodically redraws it, see below),
        # full resolution -- the probe stays exempt from the noise so it can
        # still out-resolve the model.
        if (
            adv_config.probe_resample_interval > 0
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
        _, caches = model.forward(probe_x, return_cache=True)
        fwd_dt = time.time() - t_fwd0

        cat_live = concat_caches_torch(caches, hidden_layers)

        t_probe0 = time.time()
        if it % adv_config.probe_retrain_interval == 0:
            X = cat_live.detach()
            X_fit = X[:: adv_config.probe_subsample]  # no-op slice at 1
            label_fit = probe_label[:: adv_config.probe_subsample]
            assert label_fit.any() and (~label_fit).any(), (
                "subsampled probe batch has only one class present -- lower "
                "--probe-subsample or raise --batch-size."
            )
            fit_probe(probe, X_fit, label_fit, PROBE_STEP_MAX_ITER)
            affine = probe.get_affine(device)
        probe_dt = time.time() - t_probe0

        l_probe = score_penalty(
            cat_live,
            affine,
            probe_label,
            adv_config.probe_loss_kind,
            adv_config.probe_loss_trim_frac,
        )

        if adv_config.lam_warmup_iters > 0:
            lam_eff = adv_config.lam * min(1.0, it / adv_config.lam_warmup_iters)
        else:
            lam_eff = adv_config.lam
        loss = lam_eff * l_probe + (1 - lam_eff) * l_task

        loss_before_step = loss.item()
        t_bwd0 = time.time()

        # Snapshot BEFORE the step, so a detected explosion (below) can
        # revert to it. TODO(perf): every-iteration deepcopy + re-forward
        # just to catch a rare event -- a cheaper retroactive alternative
        # exists, see PR #77's revert-and-retry discussion (not the
        # StableAdamW note below).
        if adv_config.explode_factor > 0:
            pre_model_state = copy.deepcopy(model.state_dict())
            pre_opt_state = copy.deepcopy(opt.state_dict())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if adv_config.grad_clip > 0:
            # Per-block, not whole-model -- see LogregAdversarialConfig.grad_clip.
            for block in model.blocks:
                torch.nn.utils.clip_grad_norm_(block.parameters(), adv_config.grad_clip)
        # TODO(perf/quality): grad_clip/adam_eps/adam_beta2 are band-aids
        # for instability; StableAdamW (update clipping) might be a
        # cleaner fix -- see PR #77.
        opt.step()

        if adv_config.explode_factor > 0:
            gen.set_state(noise_gen_state)  # replay the same noise as above
            with torch.no_grad():
                y_pred_after = model.forward(
                    x_task, noise_std=adv_config.resid_noise_std, generator=gen
                )
                l_task_after = torch.mean((y_pred_after[:, :num_x] - y) ** 2)
                _, caches_after = model.forward(probe_x, return_cache=True)
                cat_after = concat_caches_torch(caches_after, hidden_layers)
                l_probe_after = score_penalty(
                    cat_after,
                    affine,
                    probe_label,
                    adv_config.probe_loss_kind,
                    adv_config.probe_loss_trim_frac,
                )
                loss_after_step = (
                    lam_eff * l_probe_after + (1 - lam_eff) * l_task_after
                ).item()

            if loss_after_step > adv_config.explode_factor * loss_before_step:
                n_exploded += 1
                print(
                    f"[explode] iter={it} loss {loss_before_step:.3e} -> "
                    f"{loss_after_step:.3e} ({loss_after_step / loss_before_step:.1f}x)"
                    f" -- reverting and redoing with a tighter clip"
                )
                model.load_state_dict(pre_model_state)
                opt.load_state_dict(pre_opt_state)
                gen.set_state(noise_gen_state)  # redo must see the same noise too

                # Fresh forward: the previous graph was freed by backward()
                # above, and this is otherwise numerically the same
                # pre-step state already used for l_task/l_probe/loss.
                y_pred_full = model.forward(
                    x_task, noise_std=adv_config.resid_noise_std, generator=gen
                )
                l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)
                _, caches = model.forward(probe_x, return_cache=True)
                cat_live = concat_caches_torch(caches, hidden_layers)
                l_probe = score_penalty(
                    cat_live,
                    affine,
                    probe_label,
                    adv_config.probe_loss_kind,
                    adv_config.probe_loss_trim_frac,
                )
                loss = lam_eff * l_probe + (1 - lam_eff) * l_task

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if adv_config.grad_clip > 0:
                    tight_clip = adv_config.grad_clip / adv_config.explode_clip_divisor
                    for block in model.blocks:
                        torch.nn.utils.clip_grad_norm_(block.parameters(), tight_clip)
                opt.step()

        model_dt = fwd_dt + (time.time() - t_bwd0)

        yield TrainRecord(
            iter=it,
            loss=loss.item(),
            l_task=float(l_task.item()),
            l_probe=float(l_probe.item()),
            lam_eff=lam_eff,
            affine=affine,
            probe_dt=probe_dt,
            model_dt=model_dt,
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
    path, record: TrainRecord, model, opt, best_loss, hidden_layers, adv_config
):
    """Save all training state needed to resume from disk, atomically (a
    SIGINT can't corrupt the file)."""
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
    hist_path = os.path.join(run_log_dir, "history.json")
    history = []  # list of dicts

    if args.resume:
        model, opt, start_iter, best_loss, rck = _restore_checkpoint(last_path, device)
        if "adv_config" not in rck:
            raise SystemExit(
                f"[error] {last_path} predates the nested adv_config checkpoint "
                f"layout and cannot be --resume'd. Start a fresh run, or "
                f"--fork-from it instead."
            )
        adv_config = LogregAdversarialConfig.from_dict(rck["adv_config"])
        hidden_layers = adv_config.penalty_layers
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        print(f"[resume] from iter {start_iter}, best_loss={best_loss:.3e}")
        _check_config_json(args.tag, model.config, adv_config)
    else:
        # Hyperparameters are freshly resolved from --config + --lam (never
        # read from the checkpoint, unlike --resume above).
        file_fields = _read_config_file(args.config)
        if "seed" in file_fields:
            # Needed before the scratch init below is reproducible. (Under
            # --fork-from the model built here is discarded in favor of the
            # restored checkpoint, so the seed in effect at this point
            # doesn't matter there.)
            torch.manual_seed(file_fields["seed"])

        if args.fork_from is not None:
            source_ckpt_path = os.path.join(ckpt_dir(args.fork_from), "last.pt")
            model, opt, start_iter, best_loss, rck = _restore_checkpoint(
                source_ckpt_path, device
            )
            history = _forked_history(args.fork_from, start_iter)
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
            print(f"[init] scratch model cfg={model_config}")
            start_iter = 0
            best_loss = float("inf")
            forked_from = None

        adv_config, hidden_layers = load_run_config(
            file_fields,
            lam=args.lam,
            num_blocks=model.config.num_blocks,
            config_path=args.config,
        )
        if args.fork_from is None:
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=adv_config.lr,
                eps=adv_config.adam_eps,
                betas=(0.9, adv_config.adam_beta2),
            )
        _write_run_config(
            args.tag,
            model.config,
            adv_config,
            input_config_path=args.config,
            forked_from=forked_from,
        )

    num_x, num_blocks = model.config.num_x, model.config.num_blocks
    if not hidden_layers:
        raise SystemExit(
            f"[error] no penalty layers (num_blocks={num_blocks} has no hidden "
            f"layers). Nothing to hide against."
        )

    # --resume/--fork-from restore the optimizer's own state_dict, which
    # includes the SOURCE run's lr/eps/betas -- override with this
    # invocation's adv_config, the only case where these differ (--fork-from
    # with a new --config). A no-op for a fresh run (opt was already built
    # with them) and for --resume (adv_config comes from this same
    # checkpoint, so the values are identical either way).
    for pg in opt.param_groups:
        pg["lr"] = adv_config.lr
        pg["eps"] = adv_config.adam_eps
        pg["betas"] = (0.9, adv_config.adam_beta2)
    torch.manual_seed(adv_config.seed)

    gen = torch.Generator(device=device).manual_seed(adv_config.seed + 1)

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
        f"adam_beta2={adv_config.adam_beta2} explode_factor={adv_config.explode_factor} "
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
        probe_dt=0.0,
        model_dt=0.0,
    )

    def save(path):
        save_checkpoint(path, record, model, opt, best_loss, hidden_layers, adv_config)

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
                me = eval_max_err(model, num_x, gen, device=device)
                history.append(_history_entry(record, max_err=me))
                with open(hist_path, "w") as f:
                    json.dump(history, f)
                rate = rate_meter.update(record.iter)
                print(
                    f"iter {record.iter:>6d}  loss {record.loss:.3e}  task {record.l_task:.3e}  "
                    f"probe {record.l_probe:.3e}  λ {record.lam_eff:.1e}  max_err {me:.3e}  "
                    f"probe_dt {record.probe_dt*1e3:.1f}ms  model_dt {record.model_dt*1e3:.1f}ms  "
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
    me = eval_max_err(model, num_x, gen, device=device)
    history.append(
        _history_entry(
            record, loss=best_loss, l_task=None, l_probe=None, max_err=me, final=True
        )
    )
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(
        f"[done] iter {record.iter}  best_loss {best_loss:.3e}  final max_err {me:.3e}  "
        f"elapsed {time.time()-t0:.1f}s"
    )
    print(f"[done] checkpoints in {run_ckpt_dir}, history in {hist_path}")
    print(f"[next] python adversarial_report.py --tag {args.tag}")


if __name__ == "__main__":
    main(args)
