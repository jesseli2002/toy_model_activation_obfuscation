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
"""

import argparse
import json
import os
import shutil
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass

import config
from config import LogregAdversarialConfig, ResidualMLPConfig

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
        "adversarial objective",
        "How much weight to put on hiding c from the probe, and where in the "
        "model that penalty is applied.",
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
        "--lam-warmup-iters",
        type=int,
        default=LogregAdversarialConfig.lam_warmup_iters,
        help="linearly ramp the penalty weight 0 -> lam over this many iters "
        "(task weight ramps 1 -> 1-lam correspondingly). 0 = no ramp.",
    )
    g_adv.add_argument(
        "--penalty-layers",
        type=_parse_penalty_layers,
        default="all",
        help="'all' = every hidden layer (1..num_blocks-1), or a comma-separated "
        "subset e.g. '1,2,3'.",
    )
    g_adv.add_argument(
        "--class-threshold",
        type=float,
        default=LogregAdversarialConfig.class_threshold,
        help="probe class split: label = (c >= threshold). c ~ U[1,2], so 1.5 "
        "gives an ~even split.",
    )
    g_adv.add_argument(
        "--resid-noise-std",
        type=float,
        default=LogregAdversarialConfig.resid_noise_std,
        help="absolute Gaussian noise std added to the residual stream after "
        "every hidden layer (caches 1..num_blocks-1) on the task-loss forward "
        "pass only. 0 = no noise (pre-noise behavior).",
    )

    g_probe = p.add_argument_group(
        "probe (adversary) configuration",
        "The LogisticRegression probe's own hyperparameters and how "
        "aggressively it's refit each step.",
    )
    g_probe.add_argument(
        "--probe-C",
        type=float,
        default=LogregAdversarialConfig.probe_C,
        help="inverse L2 regularization strength for each layer's "
        "LogisticRegression probe (sklearn's C; smaller = more regularization).",
    )
    g_probe.add_argument(
        "--probe-init-iters",
        type=int,
        default=LogregAdversarialConfig.probe_init_iters,
        help="max_iter for the one-time init fit (before the training loop), "
        "which starts each probe from scratch. Per-step updates during "
        f"training instead use a fixed max_iter={PROBE_STEP_MAX_ITER} "
        "(warm-started, so a few iters is enough).",
    )
    g_probe.add_argument(
        "--probe-backend",
        choices=config.PROBE_BACKEND_CHOICES,
        default="auto",
        help="'auto' (default): torch (GPU-resident) probe iff CUDA is "
        "available, else sklearn. 'sklearn'/'torch' force a backend "
        "regardless of device -- e.g. to smoke-test the torch backend on a "
        "CPU-only machine.",
    )
    g_probe.add_argument(
        "--probe-loss-kind",
        choices=config.PROBE_LOGREG_LOSS_CHOICES,
        default=LogregAdversarialConfig.probe_loss_kind,
        help="'meandiff-relu' (default): relu(mean(s|label=1) - mean(s|label=0)) "
        "along the probe's current learned direction s. 'meandiff': same but "
        "without the relu cap.",
    )
    g_probe.add_argument(
        "--probe-subsample",
        type=int,
        default=LogregAdversarialConfig.probe_subsample,
        help="fit each per-step probe update on every Nth row of the batch "
        "instead of the full batch (e.g. 8 = 1/8 of the rows). Cuts sklearn "
        "fit cost roughly linearly; the model's own forward/backward still "
        "uses the full batch. 1 = no subsampling.",
    )
    g_probe.add_argument(
        "--probe-retrain-interval",
        type=int,
        default=LogregAdversarialConfig.probe_retrain_interval,
        help="refit the probe (and re-extract its affine) only once every N "
        "training iterations; other iterations reuse the last extracted "
        "affine for the penalty. 1 = refit every iteration (default "
        "behavior before this option existed).",
    )

    g_opt = p.add_argument_group("optimization")
    g_opt.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    g_opt.add_argument("--lr", type=float, default=config.LR)
    g_opt.add_argument("--max-iters", type=int, default=config.MAX_ITERS)
    g_opt.add_argument("--seed", type=int, default=LogregAdversarialConfig.seed)

    g_book = p.add_argument_group("bookkeeping")
    g_book.add_argument("--tag", type=str, default="adv-logreg")
    g_book.add_argument("--resume", action="store_true")
    g_book.add_argument(
        "--tag-force",
        action="store_true",
        help="delete an existing runs/<tag> directory before a fresh run.",
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

    args = p.parse_args()
    if args.warmstart is not None and args.no_warmstart:
        p.error("--warmstart and --no-warmstart are mutually exclusive.")
    if args.warmstart is None and not args.no_warmstart:
        p.error("specify --warmstart PATH or --no-warmstart.")
    return args


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import warnings

import torch
from sklearn.exceptions import ConvergenceWarning

from data import sample_batch
from model import ResidualMLP, ResidualMLPConfig
from paths import ckpt_dir, log_dir, run_dir
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

    it: int
    loss: float
    l_task: float | None
    l_probe: float | None
    lam_eff: float | None
    affine: tuple[torch.Tensor, torch.Tensor]
    probe_dt: float
    model_dt: float


def train_steps(
    model,
    opt,
    gen,
    probe,
    args,
    hidden_layers: list[int],
    start_iter: int,
    affine: tuple[torch.Tensor, torch.Tensor],
    device,
):
    """Generator over training iterations, yielding one `TrainRecord` per
    completed step (forward, probe update, backward, optimizer step). No
    checkpointing/logging here -- that's the caller's job, done between
    yields. This also means a KeyboardInterrupt while the caller is
    consuming this generator always leaves the caller's for-loop variable
    holding the last *fully completed* step, never a half-updated one."""
    num_x = model.config.num_x
    for it in range(start_iter, args.max_iters):
        t_fwd0 = time.time()
        x_full, y = sample_batch(args.batch_size, num_x, generator=gen, device=device)

        # task: noisy pass -- this is what forbids shrinking c's encoding
        # below the noise floor (see plans/resid_stream_noise_plan.md).
        y_pred_full = model.forward(
            x_full, noise_std=args.resid_noise_std, generator=gen
        )
        l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)

        # probe fit + penalty: clean pass, full resolution -- the probe stays
        # exempt from the noise so it can still out-resolve the model.
        _, caches = model.forward(x_full, return_cache=True)
        fwd_dt = time.time() - t_fwd0

        label = x_full[:, num_x] >= args.class_threshold
        assert label.any() and (~label).any(), (
            "batch has only one probe class present -- check --class-threshold "
            "against c's range."
        )

        cat_live = concat_caches_torch(caches, hidden_layers)

        t_probe0 = time.time()
        if it % args.probe_retrain_interval == 0:
            X = cat_live.detach()
            if args.probe_subsample > 1:
                X_fit = X[:: args.probe_subsample]
                label_fit = label[:: args.probe_subsample]
                assert label_fit.any() and (~label_fit).any(), (
                    "subsampled probe batch has only one class present -- lower "
                    "--probe-subsample or raise --batch-size."
                )
            else:
                X_fit, label_fit = X, label
            fit_probe(probe, X_fit, label_fit, PROBE_STEP_MAX_ITER)
            affine = probe.get_affine(device)
        probe_dt = time.time() - t_probe0

        l_probe = score_penalty(cat_live, affine, label, args.probe_loss_kind)

        if args.lam_warmup_iters > 0:
            lam_eff = args.lam * min(1.0, it / args.lam_warmup_iters)
        else:
            lam_eff = args.lam
        loss = lam_eff * l_probe + (1 - lam_eff) * l_task

        t_bwd0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model_dt = fwd_dt + (time.time() - t_bwd0)

        yield TrainRecord(
            it=it,
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
    """Persist model + optimizer state, the probe's affine boundary at
    `record`, and enough config to resume. A SIGINT arriving mid-write is
    deferred until the write completes (see `_defer_keyboard_interrupt`)."""
    w_eff, b_eff = record.affine
    with _defer_keyboard_interrupt():
        model.save(
            path,
            iter=record.it,
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
    # ad-hoc band-aid for this script's own (sklearn-backend) probe fits;
    # doesn't belong in probe_backend.py, which is meant to be reusable.
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

    torch.manual_seed(args.seed)

    # --- init the model: warm-start (default) or from scratch ---
    if not args.no_warmstart:
        if not os.path.exists(args.warmstart):
            raise SystemExit(
                f"[error] --warmstart checkpoint not found: {args.warmstart}"
            )
        model, _ = ResidualMLP.load(args.warmstart, map_location=device)
        model = model.to(device)
        model_config = model.config
        num_x, num_blocks = model_config.num_x, model_config.num_blocks
        print(f"[init] warm-started from {args.warmstart} (cfg={model_config})")
    else:
        num_x, num_blocks = args.num_x, args.num_blocks
        model_config = ResidualMLPConfig(
            num_x=num_x, d_model=args.d_model, d_mlp=args.d_mlp, num_blocks=num_blocks
        )
        model = ResidualMLP(model_config).to(device)
        print(f"[init] scratch model cfg={model_config}")

    hidden_layers = _resolve_hidden_layers(args.penalty_layers, num_blocks)
    if not hidden_layers:
        raise SystemExit(
            f"[error] no penalty layers (num_blocks={num_blocks} has no hidden "
            f"layers). Nothing to hide against."
        )

    adv_config = LogregAdversarialConfig(
        lam=args.lam,
        lam_warmup_iters=args.lam_warmup_iters,
        penalty_layers=hidden_layers,
        warmstart_path=args.warmstart if not args.no_warmstart else None,
        seed=args.seed,
        probe_C=args.probe_C,
        probe_init_iters=args.probe_init_iters,
        class_threshold=args.class_threshold,
        probe_loss_kind=args.probe_loss_kind,
        probe_subsample=args.probe_subsample,
        probe_retrain_interval=args.probe_retrain_interval,
        resid_noise_std=args.resid_noise_std,
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
        # Note: probes are stateful sklearn objects, not part of the
        # checkpoint -- resume re-inits them from the resumed model below,
        # same as a fresh run.

    # --- init-fit a single probe over the concatenation of penalized layers ---
    x_full, _ = sample_batch(args.batch_size, num_x, generator=gen, device=device)
    with torch.no_grad():
        _, init_caches = model.forward(x_full, return_cache=True)
    init_label = x_full[:, num_x] >= args.class_threshold
    assert init_label.any() and (~init_label).any(), (
        "init batch has only one probe class present -- check --class-threshold "
        "against c's range."
    )
    cat_init = concat_caches_torch(init_caches, hidden_layers)
    probe = build_probe_pipeline(args.probe_C, args.probe_init_iters, probe_backend)
    probe.fit(cat_init.detach(), init_label)
    affine = probe.get_affine(device)
    print(
        f"[init] fit concatenated probe (backend={probe_backend}) over layers "
        f"{hidden_layers}, init_iters={args.probe_init_iters}, C={args.probe_C}"
    )

    print(
        f"[adv] tag={args.tag} lam={args.lam} penalty_layers={hidden_layers} "
        f"num_blocks={num_blocks} bs={args.batch_size} "
        f"class_threshold={args.class_threshold} probe_loss_kind={args.probe_loss_kind} "
        f"probe_backend={probe_backend} probe_subsample={args.probe_subsample} "
        f"probe_retrain_interval={args.probe_retrain_interval} "
        f"resid_noise_std={args.resid_noise_std} "
        f"lr={args.lr} device={device} iters {start_iter}->{args.max_iters}"
    )

    for pg in opt.param_groups:
        pg["lr"] = args.lr

    # Placeholder record for the (edge-case) zero-iteration run, e.g.
    # --resume past --max-iters: train_steps() then yields nothing, and the
    # final save/log below still needs a record to work with.
    record = TrainRecord(
        it=start_iter,
        loss=best_loss,
        l_task=None,
        l_probe=None,
        lam_eff=None,
        affine=affine,
        probe_dt=0.0,
        model_dt=0.0,
    )

    t0 = time.time()
    try:
        for record in train_steps(
            model,
            opt,
            gen,
            probe,
            args,
            hidden_layers,
            start_iter,
            affine,
            device,
        ):
            if record.loss < best_loss:
                best_loss = record.loss
                save_checkpoint(
                    best_path, record, model, opt, best_loss, hidden_layers, adv_config
                )

            if record.it % args.log_interval == 0:
                me = eval_max_err(model, num_x, gen, device=device)
                history.append(
                    {
                        "iter": record.it,
                        "loss": record.loss,
                        "l_task": record.l_task,
                        "l_probe": record.l_probe,
                        "lam_eff": record.lam_eff,
                        "max_err": me,
                        "probe_dt": record.probe_dt,
                        "model_dt": record.model_dt,
                    }
                )
                with open(hist_path, "w") as f:
                    json.dump(history, f)
                rate = (record.it - start_iter + 1) / (time.time() - t0 + 1e-9)
                print(
                    f"iter {record.it:>6d}  loss {record.loss:.3e}  task {record.l_task:.3e}  "
                    f"probe {record.l_probe:.3e}  λ {record.lam_eff:.1e}  max_err {me:.3e}  "
                    f"probe_dt {record.probe_dt*1e3:.1f}ms  model_dt {record.model_dt*1e3:.1f}ms  "
                    f"{rate:.1f} it/s"
                )

            if record.it % args.ckpt_interval == 0 and record.it > start_iter:
                save_checkpoint(
                    last_path, record, model, opt, best_loss, hidden_layers, adv_config
                )

            if (
                args.save_every_n != 0  # i.e. not disabled
                and record.it % args.save_every_n == 0
                and record.it > start_iter
            ):
                save_checkpoint(
                    os.path.join(run_ckpt_dir, f"iter_{record.it}.pt"),
                    record,
                    model,
                    opt,
                    best_loss,
                    hidden_layers,
                    adv_config,
                )
    except KeyboardInterrupt:
        print(
            f"\n[interrupt] KeyboardInterrupt caught, saving checkpoint at iter {record.it}..."
        )
        save_checkpoint(
            last_path, record, model, opt, best_loss, hidden_layers, adv_config
        )
        print(f"[interrupt] saved to {last_path}")
        raise

    # final logging + save
    save_checkpoint(last_path, record, model, opt, best_loss, hidden_layers, adv_config)
    me = eval_max_err(model, num_x, gen, device=device)
    history.append(
        {
            "iter": record.it,
            "loss": best_loss,
            "l_task": None,
            "l_probe": None,
            "max_err": me,
            "final": True,
        }
    )
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(
        f"[done] iter {record.it}  best_loss {best_loss:.3e}  final max_err {me:.3e}  "
        f"elapsed {time.time()-t0:.1f}s"
    )
    print(f"[done] checkpoints in {run_ckpt_dir}, history in {hist_path}")
    print(f"[next] python adversarial_report.py --tag {args.tag}")


if __name__ == "__main__":
    main(args)
