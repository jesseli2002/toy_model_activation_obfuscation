# Closeout follow-up review

Small items spun out of `closeout_cleanup_plan.md` §4 that needed a closer
look later rather than a decision at the time. Not a full plan on its own —
a parking lot, in the same spirit as that plan's `analytic_feasibility/`
deferral.

## `sweep_inspect_training.py`

**Status: resolved, kept.** Was undecided between untrack (same class as
`check_dead_relu.py` / `benchmark_logreg_gpu.py` — one-off, imported by
nothing) and keep-as-background-work (same class as `sweep_threshold_report.py`,
which §4 keeps and documents). Settled as keep: the user curated its
`RUN_TAGS` directly (544f4dc, 1afed1e — moved the default off the
out-of-keep-set `sweep9_iter200k_tr*` onto
`sweep18_layer2_lam0.1_ramp200k_noise0.01_tr{0..9}`, inside group F, and
documented the list-of-1-vs-comparison usage inline), and it's listed in
`CLAUDE.md`'s "background work" section alongside `sweep_threshold_report.py`
and `sweep_group_report.py`. No further action needed.

This call was load-bearing, not just pending: `closeout_cleanup_plan.md`
§4's `adversarial_report.py` retirement drops `_plot_training_traces`'s
loss-curve-over-iterations coverage on the strength of this script already
covering the same use case (superimposed, smoothed `L_task` vs. iter across
same-settings runs) — that coverage claim now stands, since this script
stays tracked.

Notes, kept for context:

- It's an earlier, near-duplicate version of `sweep_group_report.py` — same
  shape (constants instead of a CLI, `RUN_TAGS`/curves+scatter over
  `history.jsonl` and checkpoints), before it grew into the `GROUPS`-based
  multi-comparison tool. Worth checking whether it predates
  `sweep_group_report.py` in git history and is fully superseded, rather than
  a genuinely separate use case.
- ~~Its default `RUN_TAGS` (`sweep9_iter200k_tr*`) point at runs **outside**
  the §1 keep set...~~ **Fixed** (commit 544f4dc, by the user directly): the
  default is now `sweep18_layer2_lam0.1_ramp200k_noise0.01_tr{0..9}`, inside
  group F. `CLAUDE.md` documents it as background work alongside
  `sweep_group_report.py`, edit `RUN_TAGS`-first, same convention.
