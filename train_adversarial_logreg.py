"""Step 3 (variant) — adversarial training: model vs. a stateful sklearn
LogisticRegression probe, trained *simultaneously* with the model.

Unlike train_adversarial.py's closed-form DoM/LDA penalty (recomputed from
scratch each step, no inner probe optimizer), this script keeps a single
`sklearn.linear_model.LogisticRegression` probe over the *concatenation of
all penalized hidden layers*, warm-started once against the initial model
and then advanced by a handful of solver iterations per training step -- an
actual adversary that tracks the model as it moves, rather than a static
closed-form proxy for one.

    L = lam * L_probe(live activations vs. probe's current boundary)
      + (1 - lam) * L_task(full-range c ~ U[1,2])

Per iteration:
  1. One forward pass on a fresh batch (c ~ U[1,2]) gives both the task
     prediction and the hidden-layer activation caches -- reused for
     everything below, no separate probe sub-batch.
  2. Probe labels come from the same batch: label = (c >= --class-threshold),
     splitting the c in [1,2] range into two classes (~50/50 at the default
     threshold 1.5).
  3. The probe is updated in place: `pipeline.fit(X, label)` on the
     *detached*, concatenated (across penalized layers) activations, with
     `LogisticRegression(warm_start=True)` and `max_iter` fixed to a small
     constant (PROBE_STEP_MAX_ITER) so the lbfgs solver resumes from last
     step's coefficients and only takes a few more steps -- cheap relative to
     the model's forward+backward. The StandardScaler is refit every step
     too, so the model can't dodge the probe by uniformly shrinking its own
     representation.
  4. The probe's current affine decision function (scaler mean_/scale_ +
     logreg coef_/intercept_) is extracted as detached torch constants
     `w_eff, b_eff` such that `s(r) = r @ w_eff + b_eff` reproduces the
     probe's score on the raw (unscaled), concatenated activation vector `r`.
     `s` is then recomputed on the *live* (grad-carrying) concatenated
     caches, giving a differentiable adversarial penalty without backprop
     through sklearn's solver:
         gap   = mean(s[label==1]) - mean(s[label==0])
         l_probe = relu(gap)
     relu caps the reward once the classes overlap at the boundary, so the
     model isn't pushed to over-invert (the probe would just relearn the
     flipped direction). True labels (from c), not the probe's *predicted*
     labels, are used for the split -- conditioning on predicted labels would
     make the loss discontinuous around the boundary.

     Note: a single probe over the concatenation is a strictly weaker
     adversary than one probe per layer (it can only learn one global linear
     combination across all penalized layers, not the best direction within
     each layer independently) -- but it costs one sklearn fit per step
     instead of len(penalty_layers), which is what makes simultaneous
     training tractable at all.

"Hidden" layers are the residual-stream caches strictly between the embedding
(cache 0 = input; c sits verbatim in a fixed coordinate, nothing to hide/train)
and the final residual (cache num_blocks -> y; the task *requires* it to encode
c, since sat(x,-1,1) != sat(x,-2,2)). So hidden = caches 1 .. num_blocks-1.

This is NOT gated. The deliverable is the trained checkpoint + diagnostics; run
once, then stop and review. The interesting science is the same as
train_adversarial.py: not "can it hide c" but HOW -- does it hide c only at the
probed threshold (recoverable elsewhere in [1,2] -> "hidden"), or genuinely
erase linear c-information across the range ("erased")? A moving, adaptive
probe is a strictly harder adversary than the closed-form penalty, so this is
a stress test of whatever hiding the DoM/LDA runs found.

Requires warm-starting from an existing train_adversarial.py-produced
checkpoint (no from-scratch path -- conflating "learn the task" with "hide c
from a probe that's learning simultaneously" adds a confound and isn't
supported here).

Usage:
    python train_adversarial_logreg.py --tag adv-logreg1 --lam 0.5 \
        --warmstart-path runs/nx32/checkpoints/best.pt --max-iters 6000
    python train_adversarial_logreg.py --resume --tag adv-logreg1 --max-iters 12000
    # feasibility check: can c be hidden at layer 1 at all, ignoring task loss
    python train_adversarial_logreg.py --tag adv-logreg-feas-l1 --lam 1.0 \
        --penalty-layers 1 --warmstart-path runs/nx32/checkpoints/best.pt \
        --max-iters 6000
"""

import argparse
import json
import os
import shutil
import time

import config
from config import LogregAdversarialConfig, ResidualMLPConfig

# Per-step warm-started solver iterations for the probe update (small: the
# solver resumes from last step's coefficients, so a handful of lbfgs steps
# is enough to track the model). The init fit (before the training loop) uses
# --probe-init-iters instead, since it starts from scratch.
PROBE_STEP_MAX_ITER = 10


def _parse_penalty_layers(s: str) -> str | list[int]:
    if s.strip().lower() == "all":
        return "all"
    return [int(v) for v in s.split(",") if v.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(
        description="Adversarial training: model vs. a simultaneous, "
        "stateful sklearn LogisticRegression probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tag", type=str, default="adv-logreg")
    p.add_argument(
        "--warmstart-path",
        type=str,
        required=True,
        help="checkpoint to warm-start from (train_adversarial.py-produced, "
        "loaded via ResidualMLP.load). Architecture is taken from this "
        "checkpoint's config. Required -- this script has no from-scratch path.",
    )
    p.add_argument(
        "--lam",
        type=float,
        default=LogregAdversarialConfig.lam,
        help="convex-combination weight: loss = lam * L_probe + (1-lam) * L_task. "
        "lam=1 optimizes purely for hiding c (task loss ignored). lam=0 is "
        "plain task training.",
    )
    p.add_argument(
        "--lam-warmup-iters",
        type=int,
        default=LogregAdversarialConfig.lam_warmup_iters,
        help="linearly ramp the penalty weight 0 -> lam over this many iters "
        "(task weight ramps 1 -> 1-lam correspondingly). 0 = no ramp.",
    )
    p.add_argument(
        "--penalty-layers",
        type=_parse_penalty_layers,
        default="all",
        help="'all' = every hidden layer (1..num_blocks-1), or a comma-separated "
        "subset e.g. '1,2,3'.",
    )
    p.add_argument(
        "--class-threshold",
        type=float,
        default=LogregAdversarialConfig.class_threshold,
        help="probe class split: label = (c >= threshold). c ~ U[1,2], so 1.5 "
        "gives an ~even split.",
    )
    p.add_argument(
        "--probe-C",
        type=float,
        default=LogregAdversarialConfig.probe_C,
        help="inverse L2 regularization strength for each layer's "
        "LogisticRegression probe (sklearn's C; smaller = more regularization).",
    )
    p.add_argument(
        "--probe-init-iters",
        type=int,
        default=LogregAdversarialConfig.probe_init_iters,
        help="max_iter for the one-time init fit (before the training loop), "
        "which starts each probe from scratch. Per-step updates during "
        f"training instead use a fixed max_iter={PROBE_STEP_MAX_ITER} "
        "(warm-started, so a few iters is enough).",
    )
    p.add_argument(
        "--probe-loss-kind",
        choices=config.PROBE_LOGREG_LOSS_CHOICES,
        default=LogregAdversarialConfig.probe_loss_kind,
        help="'meandiff-relu' (default): relu(mean(s|label=1) - mean(s|label=0)) "
        "along the probe's current learned direction s. 'meandiff': same but "
        "without the relu cap.",
    )
    p.add_argument(
        "--probe-subsample",
        type=int,
        default=LogregAdversarialConfig.probe_subsample,
        help="fit each per-step probe update on every Nth row of the batch "
        "instead of the full batch (e.g. 8 = 1/8 of the rows). Cuts sklearn "
        "fit cost roughly linearly; the model's own forward/backward still "
        "uses the full batch. 1 = no subsampling.",
    )
    p.add_argument(
        "--probe-retrain-interval",
        type=int,
        default=LogregAdversarialConfig.probe_retrain_interval,
        help="refit the probe (and re-extract its affine) only once every N "
        "training iterations; other iterations reuse the last extracted "
        "affine for the penalty. 1 = refit every iteration (default "
        "behavior before this option existed).",
    )
    # Optimization
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--max-iters", type=int, default=config.MAX_ITERS)
    p.add_argument("--seed", type=int, default=LogregAdversarialConfig.seed)
    # Bookkeeping
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--tag-force",
        action="store_true",
        help="delete an existing runs/<tag> directory before a fresh run.",
    )
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    args = parse_args()

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from data import sample_batch
from model import ResidualMLP, ResidualMLPConfig
from paths import ckpt_dir, log_dir, run_dir
from train_model import eval_max_err


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


def build_probe_pipeline(C: float, max_iter: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(warm_start=True, C=C, max_iter=max_iter),
    )


def fit_probe(pipeline: Pipeline, X: np.ndarray, label: np.ndarray, max_iter: int):
    """Update `pipeline` in place: refit the StandardScaler (so the model
    can't dodge the probe by uniformly shrinking its own activations) and
    advance the warm-started LogisticRegression solver by `max_iter` more
    iterations."""
    pipeline.named_steps["logisticregression"].max_iter = max_iter
    pipeline.fit(X, label)


def extract_affine(
    pipeline: Pipeline, device, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """The probe's current decision score is affine in the raw (unscaled)
    activation r: s(r) = w_eff . r + b_eff, folding the StandardScaler into
    the LogisticRegression coefficients. Returns detached torch tensors."""
    scaler = pipeline.named_steps["standardscaler"]
    logreg = pipeline.named_steps["logisticregression"]
    mu = torch.as_tensor(scaler.mean_, device=device, dtype=dtype)
    sigma = torch.as_tensor(scaler.scale_, device=device, dtype=dtype)
    w = torch.as_tensor(logreg.coef_[0], device=device, dtype=dtype)
    b = float(logreg.intercept_[0])
    w_eff = w / sigma
    b_eff = b - (w * mu / sigma).sum()
    return w_eff.detach(), b_eff.detach()


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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # --- warm-start the model (only supported init path) ---
    if not os.path.exists(args.warmstart_path):
        raise SystemExit(
            f"[error] --warmstart-path checkpoint not found: {args.warmstart_path}"
        )
    model, _ = ResidualMLP.load(args.warmstart_path, map_location=device)
    model = model.to(device)
    model_config = model.config
    num_x, num_blocks = model_config.num_x, model_config.num_blocks
    print(f"[init] warm-started from {args.warmstart_path} (cfg={model_config})")

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
        warmstart_path=args.warmstart_path,
        seed=args.seed,
        probe_C=args.probe_C,
        probe_init_iters=args.probe_init_iters,
        class_threshold=args.class_threshold,
        probe_loss_kind=args.probe_loss_kind,
        probe_subsample=args.probe_subsample,
        probe_retrain_interval=args.probe_retrain_interval,
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

    def save(path, it):
        model.save(
            path,
            iter=it,
            opt=opt.state_dict(),
            best_loss=best_loss,
            **adv_config.to_dict(),
        )

    # --- init-fit a single probe over the concatenation of penalized layers ---
    x_full, _ = sample_batch(args.batch_size, num_x, generator=gen, device=device)
    with torch.no_grad():
        _, init_caches = model.forward(x_full, return_cache=True)
    init_label_np = (x_full[:, num_x] >= args.class_threshold).cpu().numpy()
    assert init_label_np.any() and (~init_label_np).any(), (
        "init batch has only one probe class present -- check --class-threshold "
        "against c's range."
    )
    cat_init = concat_caches_torch(init_caches, hidden_layers)
    probe: Pipeline = build_probe_pipeline(args.probe_C, args.probe_init_iters)
    probe.fit(cat_init.cpu().numpy(), init_label_np)
    affine = extract_affine(probe, device)
    print(
        f"[init] fit concatenated probe over layers {hidden_layers}, "
        f"init_iters={args.probe_init_iters}, C={args.probe_C}"
    )

    print(
        f"[adv] tag={args.tag} lam={args.lam} penalty_layers={hidden_layers} "
        f"num_blocks={num_blocks} bs={args.batch_size} "
        f"class_threshold={args.class_threshold} probe_loss_kind={args.probe_loss_kind} "
        f"probe_subsample={args.probe_subsample} "
        f"probe_retrain_interval={args.probe_retrain_interval} "
        f"lr={args.lr} device={device} iters {start_iter}->{args.max_iters}"
    )

    for pg in opt.param_groups:
        pg["lr"] = args.lr

    t0 = time.time()
    it = start_iter
    for it in range(start_iter, args.max_iters):
        t_fwd0 = time.time()
        x_full, y = sample_batch(args.batch_size, num_x, generator=gen, device=device)
        y_pred_full, caches = model.forward(x_full, return_cache=True)
        l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)
        fwd_dt = time.time() - t_fwd0

        label = x_full[:, num_x] >= args.class_threshold
        label_np = label.detach().cpu().numpy()
        assert label_np.any() and (~label_np).any(), (
            "batch has only one probe class present -- check --class-threshold "
            "against c's range."
        )

        cat_live = concat_caches_torch(caches, hidden_layers)

        t_probe0 = time.time()
        if it % args.probe_retrain_interval == 0:
            X = cat_live.detach().cpu().numpy()
            if args.probe_subsample > 1:
                X_fit = X[:: args.probe_subsample]
                label_fit = label_np[:: args.probe_subsample]
                assert label_fit.any() and (~label_fit).any(), (
                    "subsampled probe batch has only one class present -- lower "
                    "--probe-subsample or raise --batch-size."
                )
            else:
                X_fit, label_fit = X, label_np
            fit_probe(probe, X_fit, label_fit, PROBE_STEP_MAX_ITER)
            affine = extract_affine(probe, device)
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

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            save(best_path, it)

        if it % args.log_interval == 0:
            me = eval_max_err(model, num_x, gen, device=device)
            history.append(
                {
                    "iter": it,
                    "loss": lv,
                    "l_task": float(l_task.item()),
                    "l_probe": float(l_probe.item()),
                    "lam_eff": lam_eff,
                    "max_err": me,
                    "probe_dt": probe_dt,
                    "model_dt": model_dt,
                }
            )
            with open(hist_path, "w") as f:
                json.dump(history, f)
            rate = (it - start_iter + 1) / (time.time() - t0 + 1e-9)
            print(
                f"iter {it:>6d}  loss {lv:.3e}  task {l_task.item():.3e}  "
                f"probe {l_probe.item():.3e}  λ {lam_eff:.2f}  max_err {me:.3e}  "
                f"probe_dt {probe_dt*1e3:.1f}ms  model_dt {model_dt*1e3:.1f}ms  "
                f"{rate:.1f} it/s"
            )

        if it % args.ckpt_interval == 0 and it > start_iter:
            save(last_path, it)

    # final logging + save
    save(last_path, it)
    me = eval_max_err(model, num_x, gen, device=device)
    history.append(
        {
            "iter": it,
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
        f"[done] iter {it}  best_loss {best_loss:.3e}  final max_err {me:.3e}  "
        f"elapsed {time.time()-t0:.1f}s"
    )
    print(f"[done] checkpoints in {run_ckpt_dir}, history in {hist_path}")
    print(f"[next] python adversarial_report.py --tag {args.tag}")


if __name__ == "__main__":
    main(args)
