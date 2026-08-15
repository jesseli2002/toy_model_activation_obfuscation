This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- closeout_cleanup_plan.md
    - Getting the repo into a presentable, reproducible state for close-out: what
      to untrack, which run data to keep and where to host it, and the README /
      CLAUDE.md split. Interactive by design -- check in with the user at every
      step, and commit locally rather than pushing.
- closeout_run_manifest.md
    - Companion data for the above: the 494 candidate runs to preserve, grouped
      by which analysis script needs them, with per-run sizes.
- closeout_followup_review.md
    - Parking lot for small closeout items spun out of closeout_cleanup_plan.md
      that need a closer look later rather than a decision now (currently just
      sweep_inspect_training.py's untrack-or-keep call).

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
