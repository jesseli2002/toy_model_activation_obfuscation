This directory contains plans for agents.

- logreg_run_config_plan.md
    - Stop splatting the adversarial config into the checkpoint's top level (which made
      --resume warn on every state key); give each run directory two artifacts -- a verbatim
      copy of the --config input and a fully-resolved config.json that finally includes the
      model architecture -- with the schema owned by one new dataclass in config.py.
      Also makes architecture flags an error under --resume/--fork-from.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. They document design rationale as of when they were written, not
current project state -- file paths, flags, and checkpoints they mention may no
longer exist. Only open one if you specifically need the history behind a past
decision, and verify anything you find against current code before reusing it.
