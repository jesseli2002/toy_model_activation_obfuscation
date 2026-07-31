This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- adversarial_report_main_decomposition_plan.md
    - Split `adversarial_report.py`'s ~170-line `main()` into
      `_run_diagnostics` (compute) / `_build_report` (existing) /
      `_make_plots`, plus a `DiagnosticsResult` container. Land last among
      this set of 5, after the others have already cleaned up the code it
      moves.

Suggested landing order (each is an independently reviewable PR):
main_decomposition. (checkpoint_access, docs, save_plot, and
dom_probe_refactor already landed.)

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
