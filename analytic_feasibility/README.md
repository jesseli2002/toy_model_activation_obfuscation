# Analytic feasibility of exact probe-hiding (session notes, 2026-07-21)

Question studied: can a width-constrained residual MLP (`d_mlp = num_x`) compute
`y = sat(x, -c, +c)` exactly while every probed residual layer is mean-constant
in `c`? Criteria fixed for this analysis:

- **Hiding criterion**: `E_x[r_k | c]` constant in `c` per residual dim. Since
  `x ⊥ c`, this is *exactly* zero covariance with `c`, so it defeats every
  affine regression / LDA / difference-of-means probe identically. It is
  necessary but **not sufficient** for logreg-at-chance (equal means with
  different higher moments can still be threshold-separated).
- **Probe scope**: residual streams `r_1..r_N` (matches `capture_layers`).
  `r_0` contains `c` by construction and is not probed.

## 1. Original problem (probe every layer): OPEN

Neither solved nor proven impossible. Sharp structure found:

### Exactly solvable pieces (all machine-verified, `verify_feasibility.py`)

- **Mean-constant, dead-band-free encoding of c** (the "v-channels"):

      v1 = -2*relu(-x1 - c)   + 2*relu(x1 - 3 + c)   - c + 3/2
      v2 = -4*relu(-x1 - c/2) + 4*relu(x1 + c/2 - 3) - c + 3

  Both have `E_x[v] = 0` for all `c`; `c` is *exactly* recoverable from
  `(x1, v1, v2)` at every point via an x1-slab-gated CPWL decode
  (x1 <= -2: `v1 - 2*x1 - 3/2`; [-2,-1]: `v2 - 4*x1 - 3`; [-1,1]:
  `3/2 - v1`; [1,2]: `3 - v2`; >= 2: `v1 - 2*x1 + 9/2`). Each channel alone
  has unavoidable non-injective x1-bands (mean-zero c-slope forces sign
  flips); two channels with disjoint bands cover everything.
- **Perfect intermediate**: `t_i = x_i - sat(x_i, c)` has mean 0 for all `c`
  and regenerates both kinks exactly: `relu(t_i) = relu(x_i - c)`,
  `relu(-t_i) = relu(-x_i - c)`.
- **Width is exactly the crux**: `d_mlp = 2n+1` solves everything in ONE block
  with perfect hiding (`o0[i] = -relu(x_i - c) + relu(-x_i - c)` + erasure
  neuron). At `d_mlp = n`, a 2-block construction solves `floor((n-1)/2)`
  coordinates exactly with perfect hiding at every layer (2 block-0 neurons
  per seeded coordinate + 1 erasure); the other half of the coordinates is the
  entire open problem.

### No-go theorems

- **T1 (pre-act mean barrier)**: every neuron pre-activation at blocks >= 1 is
  a linear functional of a probed (mean-constant) residual, hence has constant
  conditional mean. So `x_i - c` (mean `-c`) is never available as a pre-act
  after block 0, nor any uniform approximation of it. `c` is linearly readable
  ONLY inside block 0. Kills anchor+erasure hybrids and every
  "reconstruct c-hat then subtract" scheme whose intermediate has nonconstant
  mean (partial maxes, clip gates).
- **T2 (kink-sum theorem)**: if `x_i` enters all probed dims only linearly, any
  channel a single later block writes has x_i-kinks only at positions that are
  values of mean-constant functionals; pointwise
  sum(kink mass * position) then has constant conditional mean, but `t_i`
  needs `2c` and `sat(x_i, .)` needs `-2c`. So no single block can seed a
  fresh coordinate with exact full-manifold c-kinks. Regional (x1-slab) kinks
  at `x_i = c` ARE writable at blocks >= 1 (via v-channels); the open question
  is whether spurious off-slab kinks can be cancelled across multiple blocks.
  Several natural cancellation schemes (swap-triangle, tent-gated partners,
  telescoping compensators) each provably fail on a mean-constancy count.
- **n = 2 is impossible** (argument, not fully formalized): with `d_mlp = 2`,
  block 0's two neurons cannot both erase the c-dim and store any
  mean-constant c-carrying channel (the mean-map kernel is trivial in every
  case split), and any single stored channel has a positive-measure
  non-injective x-band, whose collisions with free `x_2` contradict exactness.
- **epsilon-solutions look achievable** (not fully assembled): shared
  compensator seeds `relu(x_i - c) - relu(x_j - c)` are mean-constant at 1
  neuron/coordinate; chained fallback decodes fail only on exponentially rare
  events.

### Numerical search (n=2/3, `search_exact.py`, results in `search_results.json`)

float64 Adam with penalty ramp to 1e6, Sobol integration, holdout eval:

| config                        | task MSE | mean-viol | note |
|-------------------------------|----------|-----------|------|
| ctrl_nohide_n3 (b4)           | 3.8e-02  | n/a       | **control FAILED** — optimizer can't find the known anchor at width 3 |
| ctrl_wide_n3 (d_mlp=7, hide)  | 2.3e-07  | 7.4e-10   | control OK — constraint handling works |
| hide_n2_b4                    | 2.2e-02  | 2.6e-12   | floor, consistent with impossibility |
| hide_n3_b4 / b6               | 4.3e-02 / 2.3e-02 | ~1e-08 | floor — **uninformative** given failed control |
| hide_n3_warm (v1-encoded b0)  | 5.3e-02  | ~1e-08    | warm start didn't help |

Conclusion: tiny-width optimization is the bottleneck; the n=3 floors are NOT
evidence of impossibility. If revisited: better optimizers / warm starts from
partial constructions / longer anneals.

## 2. Periodic probing (probe every 2 or 4 layers, width n/2 or n/4): SOLVED

**Exact hiding is fully constructible.** Verified end-to-end in
`period2_net.py`: n=24, d_mlp=12, 22 blocks, probes at even layers; task error
1e-14; every probed layer holds exactly
`[sat-finished coords, pending x, 0, v1, v2, 0]` (provably mean-constant).

Key facts:

- **No re-encoding needed**: v-channels are mean-constant and persist in the
  residual forever. The per-period cycle is decode -> use -> **clear** (the
  residual is additive: dirty intermediates written at unprobed layers must be
  actively cancelled — one always-on neuron per dirty dim — before the probe).
- **Period 2 needs a one-hidden-layer exact decode** so that c-hat is a
  *linear read* one layer after the basis: `c === affine(x1, v1, v2) +
  sum_j w_j * relu(P_j(x1, v1, v2))` on the encoding manifold. This exists:
  the cell-wise kink-cancellation system (15 eqs over the 5 diagonal bands cut
  by the four v-kink curves; `period2_decode.py`) is feasible with 13 free
  params, and 8 sign-valid one-sided atoms suffice (max |G - c| = 1e-14).
  Notably the curves `x1 = -c` and `x1 = 3-c` are creases (no one-sided atoms
  exist) but are not needed; all atoms live on `x1 = -c/2` and `x1 = 3-c/2`.
  Min-trees and clip-sums provably do NOT collapse to one layer (boundary
  continuity forces in-range garbage) — the kink-cancellation route is the
  trick.
- **Schedule** (period 2): block 0 writes v1, v2 (r1 unprobed, c still linear
  there); block 1 anchor-finishes a first batch of coords using c directly +
  erases c; then each period: even block computes the 8 decode atoms
  (dirty at odd = unprobed layer), odd block finishes 2 coords via pre-acts
  `x_i - c_hat` (linear read) and clears the 8 atom dims (2*2 + 8 = 12 = n/2,
  exactly the budget). Gotcha: coordinate x1 (the decode's gating variable)
  must be finished LAST — saturating dim 0 early breaks every later decode.
- **Period 4** is subsumed (strictly more unprobed depth per probe); a
  clip(L,1,2) + M*dist(x1, slab) min-tree decode also works with room to
  spare.
- Consequence: the original problem's difficulty is precisely period-1
  probing. At period >= 2, hiding is easy and exact.

### Predictions for the optimization-pressure pivot

If training with period-2/4 probes finds low-loss hidden solutions, look for:
- persistent mean-constant encoding channels (v-like: paired relus with a
  linear `-c` component, kink positions moving with c) written early and
  never cleared;
- decode/use/clear structure phase-locked to the probe period: c-materializing
  directions appearing only at unprobed layers and actively cancelled (not
  decayed) before probed ones;
- gating on a small number of x-coordinates (slab structure in the decode);
- at period 1 (original setup), by T1/T2 any low-loss mean-constant network
  must instead use regional kinks + multi-block cancellation — a concrete
  interp signature to test on adversarially trained checkpoints.

## 3. Simplified period-2 construction (supersedes section 2's decode)

Section 2 is kept above for historical context: it is the derivation that first
established feasibility, and it is correct, but the decode it produces is
larger than necessary. The simplified version below is the one to build on
(demo: `simplified_demo.py`).

Everything about the *encoding* is unchanged — same `v1`, `v2`, same
mean-constancy argument, same 5 bands. What shrinks is the *decode*.

### Only two of the four kink curves are usable

An atom is an affine `P = a0 + a1*x1 + a2*v1 + a3*v2` required to vanish
identically along one kink curve, so that `relu(P)` adds no kink of its own.
Substituting the on-curve values of `v1`, `v2` makes `P` a linear polynomial in
`c`; killing both coefficients pins `(a0, a1)` and leaves `(a2, a3)` free — one
real degree of freedom after scaling, parameterized by
`theta = atan2(a3, a2)`:

    curve x1 = -c/2   : v1 = 3/2 - c, v2 = 3 - c
                        -> a1 = -2*a2 - 2*a3,  a0 = -1.5*a2 - 3*a3
    curve x1 = 3-c/2  : v1 = 3/2,     v2 = 3 - c
                        -> a1 = -2*a3,         a0 = 3*a3 - 1.5*a2

`P` must also be one-sided about its curve (else `relu(P)` is affine and adds
nothing). Scanning `theta` numerically: `x1 = -c` and `x1 = 3-c` are creases and
admit no one-sided atom at any `theta`; all usable atoms live on `x1 = -c/2` and
`x1 = 3-c/2`. (Section 2 found the same thing the long way round.)

### Three atoms suffice, not eight

Counting instead of solving: piecewise-linear functions that are affine in
`(x1, c)` on each of the 5 bands span `5*3 = 15` dimensions, minus `4*2 = 8`
continuity constraints at the four kinks, leaves **7**. Section 2's 15-equation
system is therefore 8 equations redundant. Of those 7 basis directions,
`{1, x1, v1, v2}` are already reachable — the residual stream carries `x1`,
`v1`, `v2` and the next block has a bias — and crucially they need no neurons of
their own, because the *next* block's input matrix can read them straight from
the residual. So only `7 - 4 = 3` ReLU atoms are needed.

Concretely (verified exact to ~4e-15 in `simplified_demo.py`):

    P_A =  2*x1 - 2*v1 + v2      (vanishes on x1 = -c/2)
    P_B = -2*x1 +   v1 - 3/2     (vanishes on x1 = -c/2)
    P_C =           v1 - 3/2     (vanishes on x1 = 3-c/2)

    c = 3 + 4*x1 - 4*v1 + v2 - 2*relu(P_A) + 4*relu(P_B) + 2*relu(P_C)

### Consequences

- **Neuron budget per period drops from `2k + 8` to `2k + 3`.** At `d_mlp = n/2`
  the decode block uses 3 neurons instead of 8, and the use+clear block clears 3
  dims instead of 8. At `n = 24` (section 2's setting) that is 4 coordinates
  finished per period instead of 2 — half the depth for the same task.
- **`d_model` drops** by 5 (three `R` dims instead of eight `Q` dims).
- **No change to the hiding claim.** Probed layers still hold exactly
  `[sat-done x, pending x, 0, v1, v2, 0]`, which is mean-constant in `c`.
- **The hiding is layer-selective, and the demo shows it.** At probed (even)
  layers both probes sit at chance (AUROC ~ 0.50). At unprobed (odd) layers the
  `R_j` dims are live and logistic regression hits AUROC 1.00 — the construction
  hides `c` from the probe schedule, not from an analyst who looks anywhere.
  Note also that difference-of-means only reaches ~0.55 even at those unprobed
  layers: its direction is diluted by the 16 noisy `x` dims, so it badly
  understates what a fitted linear probe can do.

### Status of the section-2 scripts

- `period2_decode.py` — **keep as-is, historical.** It is the sympy feasibility
  proof over all 4 curves x 2 sides x 5 cells that first showed a one-hidden-layer
  decode exists. Nothing in it is wrong; the dimension count above just reaches
  the same conclusion in one line, and the one-sidedness scan reaches the
  "only two curves work" conclusion directly. No update needed.
- `period2_net.py` — **superseded by `simplified_demo.py`.** It still verifies
  end-to-end (n=24, 1e-14), so it is left in place as the record of the original
  8-atom network, but it should not be the starting point for new work: it
  re-derives the 8 atoms by an angular scan at import time, spends 8 neurons per
  decode block, and finishes half as many coordinates per period.
  `simplified_demo.py` is the newer version — same schedule, 3 closed-form
  atoms, any even `num_x`, plus probes and plots.

## Files

- `simplified_demo.py` — **start here.** Self-contained demo of the section-3
  construction. Carries its own trimmed copy of the repo's `model.py`
  (ReLU only, no layernorm/gelu/noise, weights set analytically at float64),
  so it depends on nothing else in the repo: builds the literal network
  (`d_mlp = num_x/2`), verifies task
  exactness and probed-layer content to ~1e-15, then fits difference-of-means
  and sklearn `LogisticRegression` probes at *every* residual layer and plots
  per-layer histograms + AUROCs, the encoding channels, and the learned `y(x)`
  curves (same format as repo-level `adversarial_report.py`'s `*_curves.png`).
  Writes `simplified_*.png` to `$TMPDIR` (or `--out-dir`).
- `verify_feasibility.py` — numeric checks of every claim in section 1.
- `probe_v_channels.py` — is the v1/v2 encoding detectable by a DoM or
  logistic-regression probe (no trained network; closed-form v1/v2 only)?
  DoM is at chance by construction; plain logreg is too, since it's still a
  linear classifier and the c-signal here lives in variance/covariance, not
  the mean — consistent with the "necessary but not sufficient" note above.
  Writes a 2D (v1, v2) scatter + per-probe projection histograms to
  `$TMPDIR` (or `--out-dir`).
- `search_exact.py`, `search_results.json` — n=2/3 optimization search
  harness + outcomes (note the failed no-hide control before trusting floors).
- `period2_decode.py` — sympy feasibility system for the one-layer decode
  (historical; see section 3).
- `period2_net.py` — original 8-atom period-2 exact-obfuscation demo
  (re-derives atoms, builds the literal network, verifies to 1e-14).
  Superseded by `simplified_demo.py`; see section 3.

Scripts were written to run from a scratch dir; `search_exact.py` writes
outputs to `$TMPDIR`.
