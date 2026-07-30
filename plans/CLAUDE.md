This directory contains plans for agents.

- adversarial_report_docs_plan.md
    - Drop the stale "Step 3" framing from `adversarial_report.py`'s module
      docstring and report header; trim two docstrings that narrate
      implementation mechanics down to contract + pointer. No behavior change.
- adversarial_report_save_plot_plan.md
    - Shared `save_plot` helper for `adversarial_report.py`'s and
      `probe_lib.py`'s ~9 plot functions, deduping the savefig/close/print
      tail and fixing a real figure leak in `probe_lib.plot_probe`/
      `plot_probe_pca` without breaking `--show`.
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

Suggested landing order (each is an independently reviewable PR): docs,
save_plot, dom_probe_refactor, checkpoint_access, main_decomposition.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
