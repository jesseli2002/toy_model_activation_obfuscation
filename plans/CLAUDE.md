This directory contains plans - both for agents and for humans. The ones specified below are for agents; any files not explicitly listed are plans/todo lists written by the user, for the user. This means that they may be incomplete or misleading - you probably shouldn't be reading from them, since there's likely a better place to get the information you're looking for.

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
    - (NOT STARTED) Extract the shared _LEGACY_DEFAULTS/to_dict/from_dict backfill idiom
      duplicated across ResidualMLPConfig/AdversarialConfig/LogregAdversarialConfig in
      config.py into one base class; folds in doc deduplication too.
- train_adversarial_logreg_cleanup_plan.md
    - (NOT STARTED) Complexity/separation-of-concerns/docs cleanup scoped to
      train_adversarial_logreg.py only (checkpoint-save closure, pre-call assert
      placement, history-dict helper, validation/provisioning split, doc fixes).
      Deliberately excludes train_adversarial.py, which may be sunset soon.
- rare_flags_config_plan.md
    - (NOT STARTED, SCOPING ONLY) Move rarely-used train_adversarial_logreg.py CLI flags
      to a config file persisted into the run directory. Which flags move, file format,
      and precedence are intentionally undecided -- first step is asking the user.
