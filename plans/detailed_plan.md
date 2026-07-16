# Toy Model of Probe Obfuscation — Project Plan (persistent copy)

> This is the persistent, in-repo copy of the working plan (originally drafted in
> plan-mode scratch space, which can be garbage-collected). It is the source of
> truth going forward. Last updated 2026-07-16.

## Context

**The concern being modeled.** A hypothesized AI-safety failure mode: training a
model against a linear probe causes it to *obfuscate* its activations — hiding the
probed concept rather than eliminating it. This project builds a minimal toy model
of that dynamic, in the spirit of *Toy Models of Superposition* and the
compressed-computation model of arXiv 2501.14926.

**The setup.** A 2-layer residual MLP learns `y = sat(x, -c, +c) = min(c, max(-c, x))`,
elementwise. Inputs are `x_full = [x, c]`: a dense vector `x ~ U[-3,3]^num_x` and a
scalar `c ~ U[1,2]`. `W_E = [I; 0]` and `W_U = W_E^T` are **fixed** (unit-norm
orthogonal rows), so each input coordinate owns a residual direction. Three steps,
gated: (1) train the model, (2) probe `r_1` for `c`, (3) adversarially train
model-vs-probe.

**Two spec issues found and resolved during planning:**
1. *The original task was degenerate.* With the original `x ~ U[-1,1]` and
   `c ~ U[1,2]`, `|x| <= 1 <= c` always, so `sat(x,-c,c) = x` identically —
   saturation never fires, `c` is a no-op, and Step 3 would be vacuous. **Resolved:**
   user widened `x` to `U[-3,3]` (plan.md:6). Saturation now fires ~50% of the time.
2. *The c-slot output was unspecified.* The model emits `num_x+1` values but the
   target is `num_x` wide. **Resolved:** loss covers the first `num_x` outputs only;
   the c-slot is unconstrained, leaving the residual's c-direction free for Step 3's
   pressure to act on.

**Verified during planning (not assumptions):**
- The exact construction is `sat(x,c) = x - ReLU(x-c) + ReLU(-x-c)`. Block 0 computes
  `-ReLU(x-c)` (making `r_1`'s x-part `min(x,c)`); block 1 computes `+ReLU(-x-c)`, and
  since `c > 0`, `ReLU(-min(x,c)-c) = ReLU(-x-c)` exactly. Checked numerically over
  200k samples: **max abs error 0.0**. So `d_mlp = num_x+1` suffices, with one spare
  neuron per block. `analytic.py` builds these exact weights — use it as a
  representability anchor and as a warm-start diagnostic if optimization struggles.
- At `r_1`, `mean(min(x,c)) = -(3-c)^2/12` is **c-dependent**, so **~67% of the
  difference-of-means signal at `r_1` lives in the x-directions**, not the c-direction
  (measured: c-part 1.0, x-part 0.249/dim across 32 dims). Step 3's adversary
  therefore cannot be satisfied by merely zeroing the c-slot — the DoM signal is a
  byproduct of the computation itself. But it *can* be satisfied cheaply another way
  → see **Step 3 pitfalls**.

**Environment.** Greenfield, no git. `/home/jesse/v/bin/python` (3.14), torch 2.12.1,
numpy/matplotlib/scikit-learn/einops/tqdm present. **CUDA unavailable** (no driver);
training runs on CPU (~52 it/s at the full scale). Per CLAUDE.md: long runs use
`run_in_background`, never `&`; `ps` is unreliable; `black` via absolute path; **only
one agent runs training code at a time.**

## Approach

Shared modules imported by all scripts:
- **`config.py`** — hyperparameters as constants (`SEED = 913768`, `D_MODEL = 512`,
  `NUM_X = 32`, `D_MLP = NUM_X + 1`, ranges).
- **`model.py`** — `ResidualMLP` (fixed `W_E`/`W_U` as buffers; two MLP blocks),
  returning `y` and cached `r_0/r_1/r_2`.
- **`data.py`** — `sample_batch(...)` → `(x_full, y)`; `sample_fixed_c(...)` for probes.
- **`analytic.py`** — builds the exact hand-constructed weights; verifies max err ~0.

### Step 1 — `train.py`, `train_validation.py`, `train_plot.py`

Train with MSE over the first `num_x` outputs. Seed 913768. Checkpoint to
`checkpoints/` (best + last), with `--resume` and `--max-iters` so training can be
paused/inspected. Log loss + max-abs-elementwise-error history. Early-stop on
`loss < 1e-12`.

**Known risk — convergence is the real difficulty, not capacity.** An exact zero-loss
solution provably exists (`analytic.py`), so Gate 1 is purely an optimization problem.
Early smoke tests at full scale plateaued far from exact. Escalation ladder documented
inline in `train.py`. **Strategy adopted (user direction 2026-07-16): start at
`num_x=1` — the simplest possible case — nail Gate 1 there, then raise `num_x`
incrementally.** If loss plateaus above the gate across the LR range even at small
`num_x`, that is a genuine concern → stop and report.

**Gate 1** (`train_validation.py`): max abs elementwise error over a large fresh eval
set must be small. If loss plateaus without good eval error, **stop and report** — do
not tune indefinitely.

`train_plot.py` → `plot/`: (a) iteration vs loss and vs max-abs-error; (b) `y` vs `x`
for `c ∈ [1, 1.333, 1.667, 2]`, `num_x` lines each tracing `sat`.

### Step 2 — `probe.py` + validation script

Datasets: N samples at `c=1`, N at `c=2` (x resampled). Forward pass, capture `r_1`.
Probe types: **difference-of-means**, **logistic regression**, **ridge regression on
continuous c**, **LDA** (= variance-whitened DoM).

**Gate 2 thresholds, calibrated by simulating the ideal construction — do NOT use a
naive "all probes near-perfect" bar:**

| probe | expected on a *correct* model | gate |
|---|---|---|
| raw difference-of-means | **~0.765** binary acc | > 0.70 |
| logistic regression | 1.000 binary acc | > 0.99 |
| ridge on continuous `c` | R² = 0.999998 | R² > 0.99 |
| LDA | ~1.000 | > 0.99 |

*Critical:* raw DoM scoring only ~77% is **correct behavior, not a bug** — the DoM
direction is contaminated by high-variance x-components. **Do not fail Gate 2 on the
raw-DoM number, and do not inflate probe training data to "fix" it.** Gate 2 keys on
logreg/ridge/LDA. Use sklearn LDA with `solver='lsqr', shrinkage='auto'` (the default
`svd` solver is a known landmine here).

### Step 3 — `adversarial.py`  **(DEFERRED — do not attempt yet, per user 2026-07-16)**

Loss = task MSE + `λ · |difference of means|` along the DoM direction at `r_1`. One
run, then stop and review. No gate. **Getting through Steps 1–2 cleanly is the
priority; Step 3 has subtle pitfalls (below) that need review before we build it.**

## Step 3 pitfalls — collected for later review, NOT yet resolved

These are the traps that make Step 3 easy to get wrong. Review before implementing.

1. **Blind held-out probe pair `{1.25, 1.75}`.** The DoM penalty pins the mean at two
   `c` values. Cancelling `r_1`'s c-dependent x-mean needs the *quadratic*
   `α(c) = (3-c)^2/12`, but two constraints only fix an *affine* function. The residual
   quadratic `r(c) = c^2/12 - 0.25c + 0.1667` has its vertex at **exactly c = 1.5**, so
   any held-out pair *symmetric about 1.5* (e.g. 1.25/1.75) is **blind** — verified
   `r(1.25) - r(1.75) = -2.8e-17` (machine zero). Using it would falsely show "c erased"
   when c is only hidden. **Use an asymmetric held-out pair, e.g. (1.0, 1.5)** (retains
   ~+0.021/dim of signal).
2. **Affine-shift obfuscation (the predicted, interesting result).** The single spare
   neuron (`d_mlp = num_x+1` leaves one) can emit an affine-in-`c` shift
   `α(c) = -0.25c + 0.5833` that hits `α(1)` and `α(2)` *exactly* — fully defeating the
   DoM probe at **zero task cost, one neuron, c still fully represented everywhere
   else.** Prediction to test: DoM probe fit at `{1,2}` reads ~0, probe fit at held-out
   asymmetric `c` still recovers `c`. `d_mlp` need NOT be raised.
3. **Shrink cheat.** Alternatively the model can leave `r_1[c] = ε·c` tiny and have
   block 1 read it back with weight `1.25/ε` → DoM → 0, task exact, no spare neuron
   needed. Watch for this as an alternative mechanism.
4. **Required logging.** `adversarial.py` must log BOTH the probed-`c` DoM and a
   held-out asymmetric-`c` DoM, plus a logreg/ridge recovery of `c` — otherwise the run
   cannot distinguish "erased" from "hidden," which is the whole question.

## Execution

Originally: three sequential Sonnet subagents, one per step. **Revised 2026-07-16:**
because training is resource-constrained to one agent at a time and we hit unexpected
convergence trouble, the current agent is writing Step 1 code directly and driving the
`num_x=1 → incremental` Gate 1 effort in-process. Stop at Gate 1 once it passes so the
user can review before Step 2. Step 3 deferred.

## Verification

- Step 1: run `train_validation.py` on the final checkpoint; max abs error small.
  Inspect both plots — `y` vs `x` curves must trace `sat` with knees at `±c`.
- Step 2: held-out `c` recovery across probe types; cross-check DoM vs analytic
  (c-part ≈ 1.0, x-part ≈ 0.249/dim).
- Step 3 (later): report task loss vs DoM penalty trade-off and whether `c` remains
  decodable despite matched means at the probed points.
