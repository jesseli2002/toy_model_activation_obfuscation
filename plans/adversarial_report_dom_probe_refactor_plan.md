# Unify DoM/logreg scoring in `_binary_probe_metrics_all_layers`

## Context

Inside `_binary_probe_metrics_all_layers` (`adversarial_report.py`), the DoM
(difference-of-means) direction's mean vectors are computed twice per layer:
once inside the standalone `_dom_accuracy` helper (to get `dom_acc` /
`delta_norm`), and again a few lines later, by hand, to populate
`plot_inputs[layer]["w_dom"]`/`["midpoint"]`. Same `mu_lo`/`mu_hi`
computation, same layer, same call -- pure duplicated work. Separately, the
logreg accuracy check (`(X_te_t @ w_probe_t + b_probe_t) > 0`) is a
hand-rolled affine-boundary evaluation that duplicates what
`probe_lib.LinearBoundary.score` already does (used elsewhere in this same
file for plotting).

**Explicitly out of scope:** `main()`'s `b_dom = -pi["midpoint"]` line (in
the per-layer plotting loop) stays exactly where it is, unchanged. That's a
deliberate choice, not an oversight -- putting the DoM midpoint's sign flip
inline, visibly next to the un-flipped `w_probe`/`b_probe` pair, makes the
asymmetry legible at the call site instead of hiding it behind a
uniform-looking pair of `LinearBoundary`s that were actually built two
different ways. This plan must not change `plot_inputs`'s external contract
(`w_dom`, `midpoint` fields, as raw values, stay); it only removes the
duplicate *computation* of those values inside
`_binary_probe_metrics_all_layers`, and reuses `LinearBoundary` internally
for scoring only.

Independent of `adversarial_report_docs_plan.md` and
`adversarial_report_checkpoint_access_plan.md`. Should land before
`adversarial_report_main_decomposition_plan.md`, which will copy this
function's already-cleaned body into the extracted `_run_diagnostics`.

## Design

```python
def _boundary_accuracy(b: LinearBoundary, X, y) -> float:
    """Fraction of (X, y) classified correctly by `b.score(X) > 0`."""
    return float(((b.score(X) > 0) == y).mean())
```

In `_binary_probe_metrics_all_layers`'s per-layer loop, compute
`mu_lo`/`mu_hi`/`w_dom`/`midpoint` exactly once and build a `LinearBoundary`
from them purely for scoring (never stored, never returned):

```python
mu_lo = r_lo_tr.mean(dim=0)
mu_hi = r_hi_tr.mean(dim=0)
w_dom = (mu_hi - mu_lo).cpu().numpy()
midpoint = float(((mu_hi + mu_lo) / 2).cpu().numpy() @ w_dom)
dom_boundary = LinearBoundary(w_dom, -midpoint)
X_te_np = np.concatenate([r_lo_te.cpu().numpy(), r_hi_te.cpu().numpy()], axis=0)
y_te_np = np.concatenate([np.zeros(n_test), np.ones(n_test)])
dom_acc = _boundary_accuracy(dom_boundary, X_te_np, y_te_np)
delta_norm = float(np.linalg.norm(w_dom))
```

`_dom_accuracy` is deleted (its body is now inline, once, in the loop above
-- it was only ever called from this one site). The logreg accuracy check
becomes:

```python
probe_boundary = LinearBoundary(w_probe, b_probe)
logreg_acc = _boundary_accuracy(probe_boundary, X_te_np, y_te_np)
```

replacing the hand-rolled `(X_te_t @ w_probe_t + b_probe_t) > 0` torch
comparison. Note this switches the logreg accuracy computation from torch
tensors to the numpy `LinearBoundary` path -- both `X_te_np`/`y_te_np` are
already being built for `plot_inputs` and the DoM check today, so this isn't
new work, just reusing what's already computed instead of a second
torch-side comparison. `plot_inputs[layer]` keeps its current shape exactly
(`w_dom`, `midpoint`, `w_probe`, `b_probe`, `X_te`, `y_te`, `dist_lo`,
`dist_hi`) -- no caller-visible change.

## Steps

1. Add `_boundary_accuracy` near `LinearBoundary` in `probe_lib.py` (it's a
   generic scoring helper over that type, not specific to this report).
2. Rewrite `_binary_probe_metrics_all_layers`'s per-layer loop per Design
   above; delete `_dom_accuracy`.
3. Confirm `plot_inputs`'s dict shape is byte-for-byte the same keys as
   before (grep `plot_inputs\[layer\]` and every downstream `pi["..."]`
   read in `main()`) -- this plan must not touch any of those read sites.

## Verification

- New/updated unit test (or extend existing probe tests if present) for
  `_boundary_accuracy`: a `LinearBoundary` with a known `w`/`b` against a toy
  2-point dataset scores the expected accuracy.
- Smoke run: `python adversarial_report.py --tag <existing run> --ckpt last --detailed`
  on an existing checkpoint, and diff the printed `## 2. Probe-strength gap`
  table's DoM/logreg accuracy numbers against a pre-refactor run on the same
  checkpoint/seed -- must match to floating-point tolerance (same math, same
  RNG draws, just deduplicated).
- Confirm `main()`'s `b_dom = -pi["midpoint"]` line and its neighboring
  `LinearBoundary(pi["w_dom"], b_dom)` construction are unchanged (diff
  `main()` against pre-refactor -- should show zero changes outside the
  removed `_dom_accuracy` import/call).
- `black --check adversarial_report.py probe_lib.py`.

## Risks / caveats

Low risk -- same math, reordered to avoid recomputation. The one thing to
double check during review: `dom_boundary = LinearBoundary(w_dom, -midpoint)`
must use the same sign convention as `main()`'s existing
`LinearBoundary(pi["w_dom"], -pi["midpoint"])` construction (i.e. `b = -midpoint`,
matching `LinearBoundary.score`'s `w . x + b` convention against DoM's
`w . x > midpoint` decision rule) -- getting this sign backwards would silently
flip `dom_acc` to `1 - dom_acc` without erroring.
