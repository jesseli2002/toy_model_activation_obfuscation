# `adversarial_report.py` doc cleanup

## Context

Tech-debt review of `adversarial_report.py` (the Step-3-diagnostics report
entry point) surfaced doc smells: framing tied to a superseded roadmap doc,
and a couple of docstrings that document implementation mechanics rather than
contract. No behavior changes. Independent of every other
`adversarial_report_*_plan.md` from this review -- can land first, in any
order, with zero merge risk against the others.

## Items

### 1. Drop "Step 3" framing

The module docstring (lines 1-23) and the printed/written report's first
line (`_build_report`, `f"# Step 3 adversarial diagnostics — tag=..."`) frame
this file around the old Step 1/2/3 roadmap, which is superseded (see
`plans/archive/step3_plan.md`'s own note that it's stale). Reframe both
around what the script actually does, not the plan doc that used to justify
it:

- Module docstring: keep the substance (task fidelity / probe-strength gap /
  held-out c recovery, the `--detailed` and `--steer` opt-in diagnostics) but
  drop "Step 3" and "the report that IS the Step-3 deliverable" -- e.g. open
  with "Diagnostics for an adversarially-trained checkpoint: does it hide c
  or erase it?"
- `_build_report`'s header line: `f"# Adversarial diagnostics — tag={args.tag} ckpt={args.ckpt}"`.

### 2. Trim mechanism-heavy docstrings

- `_binary_probe_metrics_all_layers`'s docstring (currently ~20 lines)
  spends a full paragraph on which backend skips the numpy round-trip and
  why -- that's `probe_backend.py`'s contract, documented there already.
  Trim to: what the function returns (`metrics`, `plot_inputs`) and the one
  fact a caller actually needs that isn't obvious from the signature
  (`eval_noise_std` only affects the test forward pass, not the fit). Drop
  the backend-internals paragraph entirely.
- `_linear_y_reconstruction`'s docstring narrates the "why this test matters"
  reasoning at essay length (why layer `num_blocks` is a sanity anchor, what
  it means if `y` is linearly recoverable early). Keep one line stating the
  contract (fits `residual[layer] -> y` per layer, `c ~ U[1,2]`) and one line
  of pointer-style rationale, not the full argument -- e.g. "layer
  `num_blocks` should score ~1 (y is exactly linear in it); see module
  docstring for what an early-layer score implies."

## Steps

1. Rewrite the module docstring (item 1) + `_build_report`'s header line.
2. Trim `_binary_probe_metrics_all_layers`'s docstring (item 2).
3. Trim `_linear_y_reconstruction`'s docstring (item 2).

## Verification

- `black --check adversarial_report.py`.
- Read the rewritten docstrings back and confirm nothing that's actually
  load-bearing (the asymmetric held-out-pairs rationale, the noise-only-on-
  eval contract) got cut along with the mechanism narration -- these two are
  intentionally kept, not part of this trim.
- No test coverage needed; this is a comment/docstring-only change with no
  code path affected.

## Risks / caveats

None -- pure comment/docstring edits, no logic touched.
