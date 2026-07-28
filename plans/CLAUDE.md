This directory contains plans for agents.

- high_level_plan.md
    - (COMPLETED) Original high-level planning for the project. Superseded; scope has shifted substantially since.
- step3_plan.md
    - (COMPLETED) Focused plan for Step 3 (adversarial training). Supersedes the stale Step 3
      sections of detailed_plan.md.
- detailed_plan.md
    - (COMPLETED) Original generated plan at project start (i.e. greenfield plan).
- train_adversarial_logreg_plan.md
    - (COMPLETED) Implementation plan for adversarial training with logistic regression probe.
- resid_stream_noise_plan.md
    - (COMPLETED) Add absolute Gaussian noise to the residual stream during adversarial
      training, so the model can no longer evade the probe by shrinking its c-encoding
      without bound. Touches model.py, config.py, and both train_adversarial*.py.
- config_dataclass_dedup_plan.md
    - (COMPLETED) Extract the shared _LEGACY_DEFAULTS/to_dict/from_dict backfill idiom
      duplicated across ResidualMLPConfig/AdversarialConfig/LogregAdversarialConfig in
      config.py into one base class; folds in doc deduplication too.
- train_adversarial_logreg_cleanup_plan.md
    - (COMPLETED) Complexity/separation-of-concerns/docs cleanup scoped to
      train_adversarial_logreg.py only (checkpoint-save closure, pre-call assert
      placement, history-dict helper, validation/provisioning split, doc fixes).
      Deliberately excludes train_adversarial.py, which may be sunset soon.
- model_noise_blob_plan.md
    - Replace the gen-state snapshot/reset dance in train_adversarial_logreg.py's
      train_steps (used to replay identical noise across the explode-check/redo
      forward passes) with an explicit, model-owned noise blob: a new
      ResidualMLP.generate_noise() method, and forward()/task_output() accepting
      either a Generator (draws fresh, same as today) or a pre-drawn blob (replays
      it). Deliberately breaks bit-identical RNG reproducibility vs. pre-refactor
      runs/checkpoints (accepted by user) and defers folding noise_std into the
      blob itself.
- logreg_run_config_plan.md
    - Stop splatting the adversarial config into the checkpoint's top level (which made
      --resume warn on every state key); give each run directory two artifacts -- a verbatim
      copy of the --config input and a fully-resolved config.json that finally includes the
      model architecture -- with the schema owned by one new dataclass in config.py.
      Also makes architecture flags an error under --resume/--fork-from.
- rare_flags_config_plan.md
    - (COMPLETED) Move most train_adversarial_logreg.py hyperparameters into a
      required --config JSON file persisted as runs/<tag>/config.json; adds a new
      --fork-from <tag> flag (branch a new experiment off an existing run's checkpoint
      with freshly-specified hyperparameters, vs. --resume which now strictly continues
      the same experiment unchanged).
