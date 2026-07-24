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
