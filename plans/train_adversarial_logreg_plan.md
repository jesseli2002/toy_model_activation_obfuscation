# Plan: `train_adversarial_logreg.py` — adversarial simultaneous LogisticRegression probe

Build off the existing `train_adversarial_logreg.py` (verbatim copy of
`train_adversarial.py`). Goal: instead of a closed-form differentiable DoM/LDA
penalty recomputed each step, keep a **stateful sklearn `LogisticRegression`
probe** that is trained *simultaneously* with the model, one solver iteration per
model step, and pit the model against that probe's current decision boundary.

Do the work as the small, self-contained commits listed at the bottom. Each
commit should leave the script importable and `--help` fast (light-imports-only
before `parse_args`, per CLAUDE.md).

---

## Core design

### 1. Single forward pass, reused for task + probe
Current script draws a task batch (`sample_batch`, `c ~ U[1,2]`) *and* a separate
pinned `c∈{1,2}` sub-batch for the probe. New design: **one** forward pass per
step.

- `x_full, y = sample_batch(batch_size, num_x, gen, device)` — `c ~ U[1,2]`.
- `y_pred_full, caches = model.forward(x_full, return_cache=True)`.
- Task loss: `l_task = mean((y_pred_full[:, :num_x] - y)**2)`. (Equivalent to
  `model.task_output`, but we need `caches`, so call `forward(..., return_cache=True)`
  once and slice.)
- Probe **labels** from the same batch's `c`: `label = (c >= 1.5)` (the `c` column
  is `x_full[:, num_x]`). Split classes at `c < 1.5` vs `c >= 1.5`. Compute once;
  reused for every layer.
- Reuse `caches[l]` (for each penalized hidden layer `l`) as the probe's features.

Drop `--probe-batch-size`, `_probe_caches`, `_delta_means_from_x`, and the pinned
`sample_fixed_c` eval batch machinery — all DoM-specific.

### 2. Stateful, warm-started probe (one per hidden layer)
Keep one `sklearn` pipeline **per penalized hidden layer** (a dict
`probes: {layer: Pipeline}`), because each layer has its own activation space.

- Pipeline: `make_pipeline(StandardScaler(), LogisticRegression(warm_start=True, max_iter=...))`.
  Matches the convention already in `adversarial_report.py:184` /
  `train_probe.py:343`. `LogisticRegression` is L2-regularized by default (`C=1.0`);
  expose `--probe-C` (default `1.0`) so it can be tuned.
- **Init fit** (before the training loop): do one forward pass on a fresh batch,
  and fit each layer's probe with a *large* `max_iter` (add `--probe-init-iters`,
  default e.g. `1000`) so the probe starts well-trained against the warm-started
  model. Store `INIT_ITERS` as the probe's current `max_iter`.
- **Per-step update**: after the init fit, on each training iteration, increment
  `logreg.max_iter += 1` and call `pipeline.fit(X, label)` again. With
  `warm_start=True` on the `LogisticRegression`, the lbfgs solver resumes from the
  previous coefficients and runs ~one more iteration — cheap. Reach into the
  pipeline's `LogisticRegression` step to bump `max_iter` (e.g.
  `pipe.named_steps["logisticregression"].max_iter += 1`).
- `X` for fit = `caches[l].detach().cpu().numpy()`; `label` = the boolean array
  above. sklearn is CPU/numpy and non-differentiable — the probe *fit* never needs
  gradient.
- Guard: both classes must be present in the batch (with `c~U[1,2]` and threshold
  1.5 the split is ~50/50 over a 16k batch, so this is safe; add a cheap assert).

> **StandardScaler / warm_start interaction — decision point.** `StandardScaler`
> refits its mean/std on every `fit` call, so warm-started `LogisticRegression`
> coefficients are technically re-interpreted in a shifted feature space each step.
> Two options:
> - **(A, recommended) Freeze the scaler after the init fit**: fit `StandardScaler`
>   once at init, then only `transform`. This keeps warm-start coherent so "one
>   solver iteration/step" is genuinely meaningful, and the LR can absorb slow
>   activation drift into its coef/intercept. Implement by fitting the scaler on the
>   init batch, then, per step, `X_scaled = scaler.transform(X)` and calling
>   `logreg.fit(X_scaled, label)` directly (skip the pipeline's scaler refit), or by
>   setting the scaler's params and not refitting.
> - **(B) Keep the plain pipeline** (`pipeline.fit` refits the scaler each step) —
>   simplest, matches the exact "StandardScaler pipeline" wording, warm-start is then
>   only an approximate init hint. Fine early; degrades as the model deliberately
>   shifts activations to fool the probe.
>
> Default to **(A)**; leave a comment pointing at this tradeoff. Flag to the user if
> anything about (A) turns out awkward.

### 3. Differentiable adversarial probe loss
The probe is non-differentiable, so we can't backprop through `pipeline.fit`.
Instead extract the probe's current **affine decision function** as torch
constants and build a differentiable score on the live (grad-carrying) `caches[l]`:

Scaler gives `mu = scaler.mean_`, `sigma = scaler.scale_`. LR gives `w = coef_[0]`,
`b = intercept_[0]` in *scaled* space. The decision score is affine in the raw
activation `r`:

```
s(r) = w · ((r - mu) / sigma) + b
     = (w / sigma) · r + (b - Σ w·mu/sigma)
     =  w_eff · r + b_eff
```

Build `w_eff = w / sigma` and `b_eff = b - (w * mu / sigma).sum()` as **detached**
torch tensors on-device (recomputed each step from the just-updated probe), then
`s = caches[l] @ w_eff + b_eff` is differentiable w.r.t. the model.

**Loss on the score (user's idea, refined) — decision point.** The probe learns
`s>0` for class 1 (`c≥1.5`), `s<0` for class 0. To fool it, drive the two classes'
scores toward each other / toward the boundary. Recommended default:

```
gap_l   = mean(s[label==1]) - mean(s[label==0])       # >0 when probe separates
l_probe = mean_l  relu(gap_l)                          # mean over penalized layers
```

- This is a difference-of-means **along the probe's learned direction** (vs. the
  raw mean-difference direction of the DoM penalty) — that's the adversarial
  coupling: the probe adapts its direction, the model responds.
- `relu(gap)` caps the reward once the classes overlap at the boundary, so the
  model isn't pushed to pointlessly *over-invert* (the probe would just relearn the
  flipped direction anyway).
- The user's original phrasing ("minimize mean score for elements *classified* >0,
  maximize for those *classified* <0") conditions on the probe's *predicted* sign;
  that's discontinuous/unstable. Using the **true labels** (from `c`) instead is
  smooth and is what the recommendation above does. Mention this substitution to
  the user.
- Make the loss kind selectable via a small `--probe-loss-kind` if cheap
  (`meandiff-relu` default, plain `meandiff` as an alt), but a single default is
  fine for a first cut — don't over-build. Confirm the default with the user if in
  doubt.

Total loss keeps the existing convex combination + λ-warmup:
`loss = lam_eff * l_probe + (1 - lam_eff) * l_task`.

### 4. Step ordering (per iteration)
1. Forward pass (with grad) → `y_pred_full`, `caches`; compute `l_task`.
2. `label = (c >= 1.5)`.
3. **Update probe(s)** on `detach().cpu().numpy()` activations (time this).
4. Extract `w_eff, b_eff` per layer from the freshly-updated probe; compute
   differentiable `l_probe` from the live `caches`.
5. `loss = lam_eff*l_probe + (1-lam_eff)*l_task`; `backward()`; `opt.step()`
   (time the model fwd+bwd+step).

Reusing the same `caches` for both the probe fit (detached) and the differentiable
penalty (live) is exactly the "reuse the forward pass" requirement.

### 5. Timing / sanity log
Time (a) the probe update over all layers and (b) the model
forward+backward+step, per iteration. Add both to the console log line (e.g.
`probe_dt 2.1ms  model_dt 18.4ms`) — sanity check that probe training is much
faster than model training. Also stash into `history` for the run log. Report
per-layer or total probe time; total is enough.

### 6. Warm-start required, no default path
- `--init` default → `"warmstart"`.
- `--warmstart-path` default → `None`; **error out** if `--init warmstart` and no
  path given ("must provide --warmstart-path"). No default checkpoint.
- Keep `--init scratch` working for parity (it already builds a fresh model). The
  checkpoint arch is still read from the warm-start checkpoint as today. Confirm the
  warm-started model is a `train_adversarial.py`-produced checkpoint (same
  `ResidualMLP.load` path — already compatible).

### 7. Drop cosine LR
Remove `_cosine_lr`, the `--lr-final` flag, and the per-step LR rescheduling loop.
Use a constant `args.lr` (still set once on the optimizer; no per-step
`pg["lr"]` update needed).

### 8. Config / checkpoint metadata
`AdversarialConfig` carries DoM-specific fields (`probe_loss`,
`probe_loss_eps`, `lda_shrinkage`, `probe_loss_detach_denom`) that don't apply.
Add a small sibling dataclass in `config.py`, e.g. `LogregAdversarialConfig`, with
the fields this script actually uses: `lam`, `lam_warmup_iters`, `penalty_layers`,
`init`, `warmstart_path`, `seed`, `probe_C`, `probe_init_iters`,
`class_threshold` (=1.5), `probe_loss_kind`, plus the freeze-scaler choice. Follow
the same `to_dict` / `from_dict` / `_LEGACY_DEFAULTS` idiom as the existing
dataclasses. Save it into the checkpoint via `save(...)` as today. Drop the
DoM-only CLI flags (`--probe-loss`, `--probe-loss-eps`, `--lda-shrinkage`,
`--probe-loss-detach-denom`).

### 9. Docstring / usage
Rewrite the module docstring to describe the simultaneous-logreg design (stateful
warm-started probe, one solver step/iter, `c<1.5` vs `c≥1.5` split, differentiable
score penalty along the probe's learned direction). Update the `Usage:` examples to
require `--warmstart-path`. Keep the "hidden vs. erased" framing and the hidden-layer
definition (caches `1..num_blocks-1`).

---

## Open questions to surface to the user (don't block; pick the recommended default and note the assumption)
1. **Scaler**: freeze after init (recommended, §2A) vs. refit each step (§2B).
2. **Probe loss**: `relu(meandiff)` along the learned direction (recommended) vs.
   plain `meandiff`; and confirm true-label (not predicted-label) conditioning is OK.
3. Whether `--init scratch` should be kept at all, or warmstart-only.

---

## Commit breakdown (small, reviewable, each leaves the script runnable)
1. **Drop cosine LR.** Remove `_cosine_lr`, `--lr-final`, and per-step LR
   rescheduling; constant `args.lr`. Mechanical.
2. **Require explicit warm-start.** `--init` default `warmstart`;
   `--warmstart-path` default `None` + validation error when missing. Update the
   `Usage:` examples. (Script still runs the old DoM penalty at this point.)
3. **Add the logreg-probe module/helpers (not yet wired).** Pure additions:
   build-pipeline, init-fit, per-step update (`max_iter += 1`, warm_start),
   `extract_affine(pipeline) -> (w_eff, b_eff)`, and the differentiable
   `score_penalty(caches, w_eff, b_eff, label, kind)`. Small, unit-testable.
   Include the freeze-scaler helper for §2A.
4. **Wire the training loop to the logreg probe.** Single forward pass →
   task loss + reused caches; per-layer probe update + differentiable penalty;
   `label = c >= 1.5`; timing (`probe_dt` / `model_dt`) in the log + history.
   Remove the now-dead DoM helpers (`_probe_penalty`, `_probe_caches`,
   `_delta_means_from_x`, pinned eval batch) and DoM-only CLI flags. Rewrite the
   module docstring.
5. **Config + metadata.** Add `LogregAdversarialConfig` to `config.py`; save it in
   the checkpoint; add `--probe-C`, `--probe-init-iters`, and (if built)
   `--probe-loss-kind`.
6. **Smoke test + cleanup.** Tiny run (`--max-iters ~50`, small
   `--penalty-layers 1`) against `runs/nx32/checkpoints/best.pt` to confirm: probe
   loss trends down, `probe_dt << model_dt`, checkpoint + history written,
   `--help` still fast. `/home/jesse/v/bin/black --check` the file. (Agent may fold
   this into the prior commits.)

Commits 3–4 could merge if the agent prefers, but keeping the pure-helper commit
(3) separate makes review of the sklearn plumbing easier.
