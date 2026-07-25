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

Normally warm-starts from an existing train_adversarial.py-produced
checkpoint (`--warmstart PATH`). `--no-warmstart` inits a fresh model from
`--num-x`/`--d-model`/`--d-mlp`/`--num-blocks` instead, conflating "learn the
task" with "hide c from a probe that's learning simultaneously" -- a
confound, so only use it to intentionally study that confound.

Three run modes (see plans/rare_flags_config_plan.md for the full design):
  - fresh run: hyperparameters are freshly resolved from `--config`'s JSON
    file plus `--lam`/`--penalty-layers`. Writes `runs/<tag>/config.json`
    once.
  - `--resume <tag>`: strictly continues the same experiment. All
    hyperparameters (CLI-common and config-file) are restored from the
    checkpoint, never re-read from `--config` or CLI. `runs/<tag>/config.json`
    is read once for a sanity check against the restored config (a mismatch
    warns but never rewrites the file) -- not for resolving hyperparameters.
  - `--fork-from <source_tag>` (with `--tag <new_tag>`): branches a new
    experiment off `source_tag`'s checkpoint (weights/optimizer/iter), but
    hyperparameters are freshly resolved exactly like a fresh run --
    `runs/<new_tag>/config.json` additionally records `forked_from`.

`runs/<tag>/config.json` is write-once and read-only thereafter for the
lifetime of that tag: hand-editing it after creation has no effect except
tripping the mismatch warning on a later `--resume`.
"""

import argparse
import dataclasses
import json
import os
import shutil
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass

import config
from config import LogregAdversarialConfig, ResidualMLPConfig
from paths import ckpt_dir, log_dir, run_dir

# Per-step warm-started solver iterations for the probe update (small: the
# solver resumes from last step's coefficients, so a handful of lbfgs steps
# is enough to track the model). The init fit (before the training loop) uses
# --probe-init-iters instead, since it starts from scratch.
PROBE_STEP_MAX_ITER = 100


def _parse_penalty_layers(s: str) -> str | list[int]:
    if s.strip().lower() == "all":
        return "all"
    return [int(v) for v in s.split(",") if v.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(
        description="Adversarial training: model vs. a simultaneous, "
        "stateful LogisticRegression probe (sklearn or GPU-resident torch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_init = p.add_argument_group(
        "model initialization",
        "Warm-start from a train_adversarial.py checkpoint (default), or init "
        "from scratch with the given architecture.",
    )
    g_init.add_argument(
        "--warmstart",
        type=str,
        default=None,
        metavar="PATH",
        help="checkpoint to warm-start from (train_adversarial.py-produced, "
        "loaded via ResidualMLP.load). Architecture is taken from this "
        "checkpoint's config. Mutually exclusive with --no-warmstart.",
    )
    g_init.add_argument(
        "--no-warmstart",
        action="store_true",
        help="init the model from scratch instead of warm-starting -- "
        "conflates learning the task with hiding c from a probe that's "
        "learning simultaneously, so only use this to intentionally study "
        "that confound. Requires --num-x/--d-model/--d-mlp/--num-blocks.",
    )
    g_init.add_argument(
        "--num-x",
        type=int,
        default=ResidualMLPConfig.num_x,
        help="(--no-warmstart only)",
    )
    g_init.add_argument(
        "--d-model",
        type=int,
        default=ResidualMLPConfig.d_model,
        help="(--no-warmstart only)",
    )
    g_init.add_argument(
        "--d-mlp",
        type=int,
        default=None,
        help="default: num_x. (--no-warmstart only)",
    )
    g_init.add_argument(
        "--num-blocks",
        type=int,
        default=ResidualMLPConfig.num_blocks,
        help="(--no-warmstart only)",
    )

    g_adv = p.add_argument_group(
        "adversarial objective (CLI-common)",
        "Touched often enough to stay CLI flags; every other hyperparameter "
        "(probe/data/optimizer) lives in --config's JSON file instead -- see "
        "LogregAdversarialConfig.",
    )
    g_adv.add_argument(
        "--lam",
        type=float,
        default=LogregAdversarialConfig.lam,
        help="convex-combination weight: loss = lam * L_probe + (1-lam) * L_task. "
        "lam=1 optimizes purely for hiding c (task loss ignored). lam=0 is "
        "plain task training.",
    )
    g_adv.add_argument(
        "--penalty-layers",
        type=_parse_penalty_layers,
        default="all",
        help="'all' = every hidden layer (1..num_blocks-1), or a comma-separated "
        "subset e.g. '1,2,3'.",
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
        help="branch a new --tag off SOURCE_TAG's latest checkpoint (weights, "
        "optimizer state, iteration count, and history -- continued for "
        "continuous loss curves). Unlike --resume, hyperparameters are "
        "freshly resolved from this invocation's --config/--lam/"
        "--penalty-layers, not inherited from SOURCE_TAG. Mutually exclusive "
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
    if args.warmstart is not None and args.no_warmstart:
        p.error("--warmstart and --no-warmstart are mutually exclusive.")
    if args.warmstart is None and not args.no_warmstart:
        p.error("specify --warmstart PATH or --no-warmstart.")
    if args.resume and args.fork_from is not None:
        p.error("--resume and --fork-from are mutually exclusive.")
    if not args.resume and args.config is None:
        p.error("--config PATH is required (unless --resume).")
    return args


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()
    # Cheap existence checks, fired before run-dir setup (in main()) can delete
    # an existing runs/<tag> or start restoring from a nonexistent checkpoint.
    if not args.no_warmstart and not os.path.exists(args.warmstart):
        raise SystemExit(f"[error] --warmstart checkpoint not found: {args.warmstart}")
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


def resolve_model(args, device):
    """Warm-start from a train_adversarial.py checkpoint (default) or init
    from scratch, per --warmstart/--no-warmstart. Assumes --warmstart's
    existence has already been checked (Tier 2, in the __main__ guard)."""
    if not args.no_warmstart:
        model, _ = ResidualMLP.load(args.warmstart, map_location=device)
        model = model.to(device)
        model_config = model.config
        print(f"[init] warm-started from {args.warmstart} (cfg={model_config})")
    else:
        model_config = ResidualMLPConfig(
            num_x=args.num_x,
            d_model=args.d_model,
            d_mlp=args.d_mlp,
            num_blocks=args.num_blocks,
        )
        model = ResidualMLP(model_config).to(device)
        print(f"[init] scratch model cfg={model_config}")
    return model, model_config


def _read_config_file(path: str) -> dict:
    """Read --config's JSON file. Required-key validation happens later, in
    load_run_config, once lam/penalty_layers are known too."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[error] --config file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[error] --config {path} is not valid JSON: {e}")


def load_run_config(
    file_fields: dict, *, lam: float, penalty_layers: list[int], config_path: str
) -> LogregAdversarialConfig:
    """Merge --config's file_fields with this invocation's CLI-common
    hyperparameters (lam, resolved penalty_layers) into one
    LogregAdversarialConfig. A missing required key in the file surfaces as
    the dataclass constructor's TypeError; re-raised here as a SystemExit
    naming the specific key(s), per this module's [error] convention."""
    try:
        return LogregAdversarialConfig(
            lam=lam, penalty_layers=penalty_layers, **file_fields
        )
    except TypeError:
        required = {
            f.name
            for f in dataclasses.fields(LogregAdversarialConfig)
            if f.default is dataclasses.MISSING
        }
        missing = sorted(required - file_fields.keys())
        if missing:
            raise SystemExit(
                f"[error] --config {config_path} missing required key(s): {missing}"
            )
        raise


def _config_json_path(tag: str) -> str:
    return os.path.join(run_dir(tag), "config.json")


def _write_config_json(
    tag: str, adv_config: LogregAdversarialConfig, forked_from: dict | None = None
) -> None:
    """Write runs/<tag>/config.json once, at tag-creation time (fresh run or
    --fork-from). Write-once, read-only thereafter -- see module docstring;
    --resume never calls this."""
    d = adv_config.to_dict()
    if forked_from is not None:
        d["forked_from"] = forked_from
    path = _config_json_path(tag)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(
        f"[config] wrote {path} (write-once -- hand-edits after this point "
        f"are only detected, as a warning, on a later --resume; never applied)"
    )


def _check_config_json(tag: str, adv_config: LogregAdversarialConfig) -> None:
    """--resume sanity check: compare runs/<tag>/config.json (written once at
    tag-creation) against the checkpoint-restored config. A mismatch means
    someone hand-edited the file since -- warn and proceed on the
    checkpoint's values; the file itself is left untouched either way."""
    path = _config_json_path(tag)
    if not os.path.exists(path):
        return  # predates this feature -- nothing to check against
    with open(path) as f:
        on_disk = json.load(f)
    on_disk.pop("forked_from", None)
    if LogregAdversarialConfig(**on_disk) != adv_config:
        print(
            f"[warn] {path} does not match the checkpoint's config -- was it "
            f"hand-edited after {tag} was created? Proceeding with the "
            f"checkpoint's values; {path} is left as-is."
        )


def _restore_checkpoint(ckpt_path: str, model, opt, device):
    """Load weights + optimizer state + iteration count from `ckpt_path` in
    place onto `model`/`opt` -- shared by --resume (source: args.tag) and
    --fork-from (source: args.fork_from). The two differ only in WHICH tag's
    checkpoint/config feeds this, not in what gets restored here. Returns
    (start_iter, best_loss, rck) so callers can pull any other checkpoint
    field they need (e.g. --resume rebuilds adv_config from rck)."""
    rck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(rck["model"])
    opt.load_state_dict(rck["opt"])
    start_iter = rck["iter"]
    best_loss = rck.get("best_loss", float("inf"))
    return start_iter, best_loss, rck


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


def score_penalty(
    cat_live: torch.Tensor,
    affine: tuple[torch.Tensor, torch.Tensor],
    label: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Differentiable adversarial penalty: project the live (grad-carrying),
    concatenated-across-layers activations onto the probe's current learned
    direction and push the two classes' mean scores together."""
    w_eff, b_eff = affine
    s = cat_live @ w_eff + b_eff
    gap = s[label].mean() - s[~label].mean()
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

    probe_x/probe_label: the fixed probe dataset (sampled once by the
    caller) -- reused for every probe fit and penalty this run, unlike the
    task batch below which resamples fresh each iteration."""
    num_x = model.config.num_x
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

        # task: noisy pass -- this is what forbids shrinking c's encoding
        # below the noise floor (see plans/resid_stream_noise_plan.md).
        y_pred_full = model.forward(
            x_task, noise_std=adv_config.resid_noise_std, generator=gen
        )
        l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)

        # probe fit + penalty: clean pass over the FIXED probe set, full
        # resolution -- the probe stays exempt from the noise so it can
        # still out-resolve the model.
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
            cat_live, affine, probe_label, adv_config.probe_loss_kind
        )

        if adv_config.lam_warmup_iters > 0:
            lam_eff = adv_config.lam * min(1.0, it / adv_config.lam_warmup_iters)
        else:
            lam_eff = adv_config.lam
        loss = lam_eff * l_probe + (1 - lam_eff) * l_task

        t_bwd0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if adv_config.grad_clip > 0:
            # Per-block, not whole-model -- see LogregAdversarialConfig.grad_clip.
            for block in model.blocks:
                torch.nn.utils.clip_grad_norm_(block.parameters(), adv_config.grad_clip)
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
            **adv_config.to_dict(),
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

    # Fresh/fork: hyperparameters are freshly resolved from --config + CLI.
    # Resume: hyperparameters instead come from the checkpoint below, so
    # --config is never read for them here.
    file_fields = None if args.resume else _read_config_file(args.config)
    if file_fields is not None and "seed" in file_fields:
        # Needed before resolve_model() below for --no-warmstart's scratch
        # init to be reproducible. (Under --resume/--fork-from the model
        # built here is fully overwritten by the restored checkpoint, so the
        # seed in effect at this point doesn't matter there.)
        torch.manual_seed(file_fields["seed"])

    model, model_config = resolve_model(args, device)
    num_x, num_blocks = model_config.num_x, model_config.num_blocks
    hidden_layers = _resolve_hidden_layers(args.penalty_layers, num_blocks)
    if not hidden_layers:
        raise SystemExit(
            f"[error] no penalty layers (num_blocks={num_blocks} has no hidden "
            f"layers). Nothing to hide against."
        )

    last_path = os.path.join(run_ckpt_dir, "last.pt")
    best_path = os.path.join(run_ckpt_dir, "best.pt")
    hist_path = os.path.join(run_log_dir, "history.json")
    history = []  # list of dicts

    if args.resume:
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)  # lr restored below
        start_iter, best_loss, rck = _restore_checkpoint(last_path, model, opt, device)
        adv_config = LogregAdversarialConfig.from_dict(rck)
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        print(f"[resume] from iter {start_iter}, best_loss={best_loss:.3e}")
        _check_config_json(args.tag, adv_config)
    else:
        adv_config = load_run_config(
            file_fields,
            lam=args.lam,
            penalty_layers=hidden_layers,
            config_path=args.config,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=adv_config.lr)
        if args.fork_from is not None:
            source_ckpt_path = os.path.join(ckpt_dir(args.fork_from), "last.pt")
            start_iter, best_loss, rck = _restore_checkpoint(
                source_ckpt_path, model, opt, device
            )
            history = _forked_history(args.fork_from, start_iter)
            forked_from = {"tag": args.fork_from, "iter": start_iter}
            print(
                f"[fork] from {args.fork_from} @ iter {start_iter}, "
                f"best_loss={best_loss:.3e}"
            )
        else:
            start_iter = 0
            best_loss = float("inf")
            forked_from = None
        _write_config_json(args.tag, adv_config, forked_from=forked_from)

    # --resume/--fork-from restore the optimizer's own state_dict, which
    # includes the SOURCE run's lr -- override with this invocation's
    # adv_config.lr, the only case where that differs (--fork-from with a new
    # --config). A no-op for a fresh run (opt was already built with it).
    for pg in opt.param_groups:
        pg["lr"] = adv_config.lr
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
        f"resid_noise_std={adv_config.resid_noise_std} grad_clip={adv_config.grad_clip} "
        f"lr={adv_config.lr} device={device} iters {start_iter}->{args.max_iters}"
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
                rate = (record.iter - start_iter + 1) / (time.time() - t0 + 1e-9)
                print(
                    f"iter {record.iter:>6d}  loss {record.loss:.3e}  task {record.l_task:.3e}  "
                    f"probe {record.l_probe:.3e}  λ {record.lam_eff:.1e}  max_err {me:.3e}  "
                    f"probe_dt {record.probe_dt*1e3:.1f}ms  model_dt {record.model_dt*1e3:.1f}ms  "
                    f"{rate:.1f} it/s"
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
