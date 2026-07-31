This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- adversarial_report_dom_probe_refactor_plan.md
    - Unify DoM/logreg accuracy scoring in `_binary_probe_metrics_all_layers`
      via `probe_lib.LinearBoundary`, removing a duplicated DoM mean-vector
      computation. Deliberately leaves `main()`'s visible
      `b_dom = -pi["midpoint"]` sign flip untouched.
- adversarial_report_checkpoint_access_plan.md
    - Consolidate near-duplicate comments (not logic -- each check stays a
      1-2 line inline check) explaining `adversarial_report.py`'s
      optional-`adv_config` checkpoint fallback rule, currently restated at
      3 call sites.
- adversarial_report_main_decomposition_plan.md
    - Split `adversarial_report.py`'s ~170-line `main()` into
      `_run_diagnostics` (compute) / `_build_report` (existing) /
      `_make_plots`, plus a `DiagnosticsResult` container. Land last among
      this set of 5, after the others have already cleaned up the code it
      moves.

Suggested landing order (each is an independently reviewable PR):
dom_probe_refactor, checkpoint_access, main_decomposition. (docs and save_plot
already landed.)

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
