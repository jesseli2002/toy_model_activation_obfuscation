# Consolidate checkpoint adv-config comments in `adversarial_report.py`

## Context

Three sites in `adversarial_report.py` independently inspect a loaded
checkpoint dict's optional `adv_config` (or, in one case, `probe_w`) and each
carries its own comment explaining which fields are safe to hard-index vs.
need a soft `.get()`:

- `main()` (deriving `penalty_layers`/`eval_noise_std`/`class_threshold`):
  "ck may carry no adversarial config at all ... that's the only case that
  falls back; once adv_config exists, penalty_layers is always a key in it
  ... so fail loudly on a missing key rather than silently defaulting."
- `_build_report` (deriving `lam`/`init`): near-identical comment, restating
  the same fallback rule for a different pair of fields.
- `_auroc_snapshots` (checking `"probe_w" in ck` to skip non-logreg
  snapshots): a related but *distinct* question -- not "does this checkpoint
  have any adv_config" but "is this specifically a
  `train_adversarial_logreg.py` checkpoint" (a plain `train_adversarial.py`
  checkpoint has `adv_config` but no `probe_w`).

This is comment duplication riding on top of logic that's each only 1-2
lines -- not enough independent logic to justify a shared accessor function
or wrapper class (three call sites, each reading different fields, one of
which is checking a genuinely different condition). The fix scoped here is
to state the fallback *rule* once and have the two same-question sites
(`main()`, `_build_report`) point back to it tersely, and give
`_auroc_snapshots` its own short, distinct comment rather than a
near-duplicate of the other two's.

If a fourth site needing this pattern shows up later, or these checks grow
beyond one field lookup each, revisit as a real accessor
(`get_adv_config(ck) -> dict | None`) at that point -- not preemptively here.

Independent of every other `adversarial_report_*_plan.md` from this review.
Touches lines `adversarial_report_main_decomposition_plan.md` will later
move into `_run_diagnostics`/`_build_report` -- land this one first so that
decomposition inherits the already-consolidated comments.

## Design

Write the rule once, at the *first* site it applies (`main()`, since it runs
before `_build_report`):

```python
    # ck may carry no adversarial config at all (a train_probe.py/
    # train_model_plot.py checkpoint) -- that's the only case that falls
    # back to a default. Once adv_config exists, every field on
    # AdversarialConfig/LogregAdversarialConfig is always present (this
    # applies wherever adv_config is read below and in _build_report), so
    # fail loudly (direct indexing) on a missing key there rather than
    # silently defaulting -- except fields that exist on only one of the two
    # config classes (e.g. `init`), which legitimately stay a soft `.get()`.
```

`_build_report`'s site drops its own restatement and instead gets a
one-line pointer: `# adv_config fallback rule: see main()`. `main()`'s
existing per-field reasoning about which specific fields are always-present
vs. soft-`.get()` stays local to each usage (that part is genuinely
call-site-specific, not duplicated).

`_auroc_snapshots` keeps its own short, accurate comment, distinguished from
the other two:

```python
        if "probe_w" not in ck:
            continue  # not a train_adversarial_logreg.py checkpoint (a plain
                      # train_adversarial.py checkpoint has adv_config but no
                      # probe_w/probe_b/probe_layers)
```

## Steps

1. Rewrite `main()`'s adv_config comment per Design.
2. Replace `_build_report`'s adv_config comment with the one-line pointer.
3. Rewrite `_auroc_snapshots`'s comment to be self-contained and distinct
   (no change to the `"probe_w" not in ck` check itself).

## Verification

- Read all three sites back and confirm: (a) no site duplicates another's
  full rationale, (b) `_build_report`'s pointer resolves to a real comment in
  `main()` that still matches (won't drift silently if one is edited without
  the other -- flag this as a standing review-time check, not something
  automatable here).
- `black --check adversarial_report.py`.
- No behavior change -- comment-only, verify with `git diff` showing no
  non-comment lines touched.

## Risks / caveats

None -- comment-only change. The only ongoing risk is the usual one for any
"see X" pointer comment: if `main()`'s adv_config handling is later
restructured (e.g. by `adversarial_report_main_decomposition_plan.md`) and
the pointer isn't updated to match, it goes stale. Worth a quick check when
that plan lands.
