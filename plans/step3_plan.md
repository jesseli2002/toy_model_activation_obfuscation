# Step 3 — Adversarial training (model vs. difference-of-means probe)

> Focused plan for Step 3, written 2026-07-18 after Steps 1 & 2 landed. Supersedes
> the "Step 3" and "Step 3 pitfalls" sections of `detailed_plan.md`, which were
> drafted at project start against assumptions that no longer hold (`d_mlp = num_x`,
> 4 blocks — see **Corrections**). Read this file, not those sections, for Step 3.

## The question we're actually asking

We add a loss term that pressures the model to make `c=1` and `c=2` activations share
the *same mean* along the residual stream (i.e. defeat a difference-of-means probe),
while still computing `sat(x,-c,c)` correctly. The interesting science is **not**
"can it hide c" (we expect yes) but **how it hides it**:

- Does it hide `c` only at the two *probed* values `{1, 2}`, leaving `c` fully
  recoverable everywhere else in `[1,2]`? ("**hidden**")
- Or does it genuinely remove linear `c`-information across the whole range?
  ("**erased**")
- Does it defeat the *specific* probe (DoM) but stay readable by a stronger one
  (logreg/LDA exploiting second-moment structure — means match, covariances differ)?

Every one of these is an interesting result, not a failure. The job of the Step 3
script is to have enough logging to **tell these outcomes apart**. There is no gate:
per the original spec, run once, then stop and review together.

## What's already in place (reuse, don't rebuild)

- `model.py` — `ResidualMLP(num_x, d_model, d_mlp, num_blocks=...)`, fixed
  `W_E=[I;0]`/`W_U=W_E^T`, returns `y` and per-layer residual cache via
  `forward(x, return_cache=True)` → `caches[0..num_blocks]` (0 = embedding,
  `i` = after block `i-1`; the hidden layers here are 1..3).
- `data.py` — `sample_batch` (c~U[1,2]) and `sample_fixed_c` (c pinned).
- `train_probe.py` — DoM + logreg probes at any layer, plus causal steering; its
  `load_model`, `capture_layers`, `binary_dataset` are directly reusable as the
  Step 3 evaluation harness.
- `train_model.py` — `eval_max_err` for the task-fidelity metric.
- Canonical trained checkpoint: `runs/nx32/checkpoints/best.pt`
  (`num_x=32, d_model=256, d_mlp=32, num_blocks=4`) — the Step-1 model Step 3
  warm-starts from.

## Corrections to the original plan's assumptions

- **`d_mlp = num_x`, not `num_x+1`.** There is *no* single spare neuron. The old
  pitfall #2 ("the one spare neuron emits an affine shift") is void as written.
- **The model is 4 blocks deep, not 2.** The exact `sat` construction needs `num_x`
  ReLUs across 2 blocks (~64 units); the trained model has 4×32 = 128 ReLU units. So
  the spare capacity for obfuscation lives in **depth**, not one extra width neuron.
  An affine/quadratic `c`-shift is still cheaply available — more so — just not via
  the mechanism the old plan named.
- Old "pitfalls" 1–3 are **hypotheses about model behavior**, reclassified below as
  *outcomes to measure*. Old pitfall #4 (logging) is *useful* for distinguishing
  outcomes but not mission-critical — we build the logging because it's cheap and it's
  the whole point, not because the run fails without it.

## The script — `train_adversarial.py`

Mirror `train_model.py`'s structure (argparse-before-heavy-imports, per-`--tag` run
dir, checkpoint/resume, cosine LR). New pieces:

### Loss
```
L = L_task(full-range c)  +  λ · L_probe(pinned c ∈ {1,2}, all hidden layers)
```
- **`L_task`** — MSE over the first `num_x` outputs on a batch with `c ~ U[1,2]` (the
  *full* range). Keeping the task trained across all `c` is essential: it prevents
  "c not recoverable at held-out c" from being confounded with "the model was never
  trained there." This is the single most important design choice.
- **`L_probe`** — sum over the hidden layers of the squared L2 norm of the difference
  of class means, on a *separate* pinned sub-batch (half `c=1`, half `c=2`, `x`
  resampled):
  `Σ_{ℓ∈hidden} ‖ mean(r_ℓ | c=2) − mean(r_ℓ | c=1) ‖²`. This *is* the DoM probe's
  separability; driving it to 0 makes the two class means coincide at every hidden
  layer → DoM accuracy → chance. Closed-form, no inner probe-training loop — keeps the
  first experiment clean. (A co-adapting logreg adversary would need alternating
  optimization; explicitly out of scope for the first run.)

### Key knobs (argparse)
- `--init warmstart|scratch`, default **warmstart** from `runs/nx32/.../best.pt`.
  Warm-start most directly models the safety scenario — a *capable, non-obfuscated*
  model then put under probe pressure — and lets us watch obfuscation *emerge* from a
  known-good solution. `scratch` conflates learning the task with hiding; kept for
  contrast.
- **Penalize all hidden layers** by default (`--penalty-layers all`; an explicit
  subset stays available for follow-up runs). Penalizing every hidden layer closes the
  "relocate `c` to an unpenalized layer" escape route from the start.
- `--lam` penalty weight. Primary run: one value (start modest, ~1.0; tune so the
  penalty actually bites without wrecking the task). Support a small sweep for
  follow-up, but per spec the deliverable is one run → review.
- Standard `--tag`, `--resume`, `--max-iters`, `--seed`.

## Diagnostics — the actual deliverable

Log during training and dump a final report + plots:

1. **Task fidelity** — max-abs elementwise error (reuse `eval_max_err`). Tells us the
   *price* of hiding. Expectation: near-zero, i.e. hiding is nearly free.
2. **Penalized DoM per layer** — `‖Δmean‖` at each hidden layer over training. Should
   fall toward 0 at all penalized layers; per-layer traces also confirm no relocation.
3. **Held-out `c` recovery (hidden vs. erased)** — reuse `train_probe.py`'s harness:
   - **Continuous ridge probe** on `c ~ U[1,2]` (R² of recovered `c`): the cleanest
     "is c linearly present across the range" measure.
   - **Binary logreg/LDA at held-out `c` pairs** *not* in `{1,2}`.
4. **Probe-strength gap** — DoM accuracy vs logreg/LDA accuracy at `{1,2}` *after*
   training. If DoM → chance but logreg still separates, the model defeated the
   *specific* probe via first moments while leaving second-moment structure readable —
   an interesting, reportable outcome.

### Outcomes this logging distinguishes (was "pitfalls" 1–3)
- **Point-hiding / c-dependent shift** — penalized DoM→0, but continuous ridge still
  recovers `c` across `[1,2]`. Model cancels the class means at the pinned points with
  a `c`-dependent shift; `c` remains represented.
- **Full erasure** — no linear probe recovers `c` anywhere in `[1,2]`.
- **Shrink cheat** — `c`-direction scaled to ~ε and read back downstream with large
  weight; DoM→0, task exact. Look for it in the weights if DoM collapses with no
  held-out signal loss.
- *Relocation* is largely foreclosed by penalizing all hidden layers, but the
  per-layer DoM logging still confirms whether it was attempted.

### Measurement-design note (folds in old pitfall #1)
When choosing held-out `c` *pairs*, avoid pairs **symmetric about c = 1.5**
(e.g. 1.25/1.75). The residual after an affine cancellation is a quadratic in `c`
whose vertex sits at exactly `c = 1.5`, so a symmetric pair reads ~0 difference even
when `c` is only *hidden*, not erased — you'd falsely conclude "erased." Prefer an
**asymmetric pair** (e.g. 1.0 vs 1.5) and, better, lean on the **continuous ridge
probe**, which sidesteps the issue entirely. This is a note on *how we measure*, not a
thing to prevent.

## Execution & review
- One training agent at a time (resource constraint). Warm-start from `nx32` keeps
  runs short.
- No gate. Produce: the trained adversarial checkpoint under a new `--tag`, the
  diagnostics report, and plots (DoM-per-layer over training; held-out recovery;
  task-error trace). Then **stop and review together** — Step 3's value is in the
  interpretation, not in passing a threshold.
