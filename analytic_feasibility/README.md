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

## Files

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
- `period2_decode.py` — sympy feasibility system for the one-layer decode.
- `period2_net.py` — self-contained period-2 exact-obfuscation demo
  (re-derives atoms, builds the literal network, verifies to 1e-14).
- `period2_decode_walkthrough.ipynb` — interactive (ipywidgets slider) notebook
  walking through how the period-2 decode formula works, step by step: the
  `v1`/`v2` encoding, the easy-but-ungateable 5-window decode, the
  curve-vanishing ReLU atoms that remove the gating, and the final
  `c = affine + Σ w_j·relu(atom_j)` decomposition.

Scripts were written to run from a scratch dir; `search_exact.py` writes
outputs to `$TMPDIR`.
