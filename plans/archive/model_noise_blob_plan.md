# Model-owned noise blob, replacing RNG-state replay in train_steps

## Context

`train_adversarial_logreg.py`'s `train_steps` needs the **same** residual-stream
noise to appear in up to three forward passes per iteration: the initial noisy
task pass, the post-step explode-check pass, and (if an explosion is detected)
the redo pass after reverting. Today this is achieved by snapshotting `gen`'s
RNG state right before the noisy forward and resetting it (`gen.set_state(...)`)
before each pass that must replay it — see the "Snapshotted right before the
first noise draw..." comment in `train_steps`. This works but is fragile: any
draw from `gen` accidentally placed inside the snapshot/reset window (or a
future refactor moving code around it) silently desyncs the replay.

This plan moves noise generation into `model.py` as an explicit, replayable
value instead of an implicit RNG-state window. `ResidualMLP` already owns the
knowledge of how many blocks get noise and at what shape (`model.py:140-170`);
today that knowledge is *used* internally but the caller still has to reason
about `gen`'s state externally to get replay. Exposing a `generate_noise()`
method keeps that shape/block-count knowledge inside `model.py` while giving
`train_steps` an opaque, replayable handle.

Followed on from the `forward_loss` closure-purity cleanup (see commit
`797a075`: `probe_x`/`probe_label` made explicit args) — this plan addresses
the two remaining sources of implicit closure state flagged in that
discussion: `gen`'s RNG-state replay dance is the one item that's genuinely
worth restructuring; `affine`/`probe` mutation was judged inevitable and is
out of scope here.

## Design

**`model.py`**

- New method `ResidualMLP.generate_noise(batch_size, generator) -> list[Tensor]`:
  draws one `torch.randn(...)` tensor per injection point (mirrors the current
  in-`forward` loop over `caches[1:num_blocks]`), matching device/dtype of the
  model's parameters. Unscaled (unit variance) — `noise_std` is applied at
  the point of use in `forward`, not baked into the blob (see Out of scope).
- `ResidualMLP.forward` and `task_output` replace their `generator` param with
  `noise: torch.Generator | list[Tensor] | None = None`:
  - `None` → no noise, regardless of `noise_std` (today's default path).
  - `isinstance(noise, torch.Generator)` → internally calls
    `self.generate_noise(x_full.shape[0], noise)` and proceeds as below. This
    preserves every existing non-training call site (`train_probe`,
    `data.eval_max_err`, `analytic.py`, `adversarial_report.py`,
    `train_adversarial.py`) with zero changes — they keep passing a generator.
  - otherwise (a `list[Tensor]`) → use directly as the per-block noise,
    scaled by `noise_std` at each injection point.
- This is an explicit type-based overload (`isinstance` check on one param).
  Accepted as a minor smell for this scope; see Alternatives.

**`train_adversarial_logreg.py`**

- In `train_steps`, once per iteration — at the same point `noise_gen_state`
  is captured today — draw `noise_blob = model.generate_noise(adv_config.batch_size, gen)`.
- `forward_loss` gains a `noise: list[Tensor]` param, threaded straight into
  its two `model.forward(...)` calls (task pass and probe pass — note only
  the task pass actually uses noise today; probe pass stays clean per
  `resid_stream_noise_plan.md`).
- Thread the same `noise_blob` into all `forward_loss` calls within one
  iteration (initial, explode-check, explode-redo) instead of resetting `gen`.
- Delete: `noise_gen_state = gen.get_state()`, both
  `gen.set_state(noise_gen_state)` calls in the explode block, and the
  "Snapshotted right before..." comment describing the replay window.

## Out of scope

- **Folding `noise_std` into `generate_noise`'s output** (pre-scaled blob).
  Judged a bigger surface change than warranted this session — keep
  `noise_std` as `forward`'s own knob, applied at the injection point.
- **Bit-identical RNG draw order vs. pre-refactor runs/checkpoints.** The
  user has explicitly accepted breaking reproducibility against existing
  runs this one time — no `_LEGACY_DEFAULTS`-style migration or draw-order
  preservation effort needed.
- **`train_adversarial.py`** (LDA path) — not touched; user has largely
  moved off it.
- **`affine`/`probe` closure mutation in `forward_loss`** — separately
  discussed and judged inevitable, not part of this plan.

## Verification

1. `black --check`, `python -m pytest test_train_adversarial_logreg.py`.
2. `noise=None` forward is bit-identical to a call with `noise` omitted
   entirely (guards every existing non-training call site).
3. Passing a `Generator` directly vs. manually calling `generate_noise` first
   and passing the resulting blob (same incoming generator state) produce
   identical output — confirms the overload's two paths agree.
4. New property test: within one simulated iteration, the explode-check and
   explode-redo forward passes see bit-identical noise to the initial pass
   (assert tensor equality across the three `forward_loss` calls) — this is
   the actual property the refactor is buying, replacing the old
   snapshot/reset discipline with something assertable.
5. Short smoke run (`--max-iters` ~200) with `--explode-factor` set low
   enough to trigger occasionally, confirming training still proceeds and
   reverts sanely.

## Alternatives considered

- **Two explicit params** (`noise: list[Tensor] | None`, `generator:
  Generator | None`) instead of one overloaded param — avoids type-based
  dispatch, more explicit at call sites, but adds a second always-present
  kwarg everywhere. Deferred; revisit if the overload proves confusing to
  read at call sites.
- **Passing pre-drawn noise tensors from `train_steps` directly**, without a
  `model.generate_noise` method — rejected: leaks block-count/shape knowledge
  out of `model.py` into the training loop, the opposite of what this plan
  is for.
