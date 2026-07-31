This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- adversarial_report_terminology_plan.md
    - Rename `adversarial_report.py`'s "diagnostic(s)" vocabulary (module
      docstring, `parse_args` description, `_build_report` header,
      `DiagnosticsResult`) to "analysis" -- the hidden-vs-erased framing that
      motivated "diagnostic" is itself retired; this script produces a
      broader per-layer/per-metric breakdown, not a single pass/fail test.

Suggested landing order (each is an independently reviewable PR):
(docs, save_plot, dom_probe_refactor, checkpoint_access, and
main_decomposition already landed.)

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
