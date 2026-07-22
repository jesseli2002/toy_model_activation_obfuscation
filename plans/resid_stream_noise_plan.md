# Residual-stream noise for adversarial training

## Context

In `train_adversarial_logreg.py`, the model struggles to make the two probe
classes coincide: any residual difference is detected by the logreg probe and
amplified by the `StandardScaler` fold (`w_eff = w / scale_`, probe_backend.py:54),
because a direction carrying vanishing c-signal also has vanishing variance.
The ratio stays O(1) and the target recedes.

**The hypothesis this change tests** (user's, and the reason for the design):
the model is currently evading the probe by *shrinking* its c-encoding without
bound. Adding an absolute noise floor to the residual stream makes shrinking
useless past a point — c must survive transit to the output to compute `sat`,
so it must be carried at amplitude ≳ σ. With shrinking closed off and the probe
still at full resolution, the model is forced into a qualitatively different
strategy: **encoding c nonlinearly, at high amplitude**.

Note the deliberate asymmetry: **the model pays the noise penalty, the probe is
exempt.** This is intentional. The probe is meant to out-resolve the model, so
that "shrink below the floor" is not an escape. An earlier idea — flooring the
probe's whitening denominator at σ² — is explicitly **rejected**: it relieves
pressure on the model, which is the opposite of the goal.

Scope: noise plumbing only. Extra instrumentation and any nonlinear readout are
out of scope and will be handled separately.

## Settled design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Noise scale | **Absolute** σ, default **0.1** | Relative/RMS noise is scale-free and buys nothing. `r_0 = x_full @ W_E` anchors the stream at fixed scale, so absolute is well-defined. 0.1 rather than 0.05 to keep the floor binding despite redundant encoding (see Risks). |
| Injection points | After blocks 0..num_blocks−2, i.e. onto caches **1..num_blocks−1** | Exactly the penalized hidden layers. Embedding (cache 0) and final residual (cache num_blocks → y) are not injected into directly. |
| Propagated? | **Yes** | Noise enters the stream and is seen by all downstream blocks. This is what forces c to be carried robustly. |
| `L_task` forward pass | **Noisy** | Load-bearing. This is the *only* thing that forbids shrinking — c must reach the output through the noised layers. A clean task loss would let the model shrink c to 1e-6 and ace the task, making the noise inert. |
| Probe fit **and** penalty | **Clean** pass, both | Probe keeps full resolution (the point). Both on the same pass so `w_eff`'s amplification self-cancels, and so the training constraint matches `adversarial_report.py`, which probes clean activations. |

**Do not move the penalty onto the noisy pass.** A clean-fitted `scale_` on an
unused direction gives `w_eff` ~1e7; evaluated against noisy activations where
that direction carries σ, the penalty picks up a noise term of order
`1e7 · σ · 2/√B` ≈ 10³ against an O(1) signal. Fit and evaluation must share a pass.

## Step 1 — `model.py`

Add two defaulted params to `ResidualMLP.forward`; every other call site in the
repo keeps calling it unchanged and stays clean:

```python
def forward(self, x_full, return_cache=False, noise_std: float = 0.0, generator=None):
    r = x_full @ self.W_E
    caches = [r]
    for i, block in enumerate(self.blocks):
        r = r + block(r)
        if noise_std > 0.0 and i + 1 < self.num_blocks:
            r = r + noise_std * torch.randn(
                r.shape, device=r.device, dtype=r.dtype, generator=generator
            )
        caches.append(r)
    ...
```

Mirror the same two params on `task_output` (the entry point
`train_adversarial.py` uses for its task loss). Annotate with jaxtyping per
CLAUDE.md.

Noise is a **training** choice, not architecture — it does **not** go in
`ResidualMLPConfig`, and `forward` takes it as an argument rather than storing it.

## Step 2 — `config.py`

Add `resid_noise_std: float = 0.1` to **both** `AdversarialConfig` and
`LogregAdversarialConfig`, with `_LEGACY_DEFAULTS["resid_noise_std"] = 0.0`.
This is the documented forward-default-vs-legacy-default divergence idiom
(cf. `probe_loss`: forward `"lda"`, legacy `"squared"`) — pre-noise checkpoints
reconstruct noise-free, new runs get 0.1 unless `--resid-noise-std 0`.

## Step 3 — `train_adversarial_logreg.py` (primary target)

- Add `--resid-noise-std`, defaulting to `LogregAdversarialConfig.resid_noise_std`.
  This script's CLI is split into argument groups (PR #51); it belongs in
  `g_adv` ("adversarial objective"), not `g_opt` — it is part of the hiding
  pressure, not a plain optimization knob.
- In `train_steps`, split the single forward into two over the **same** `x_full`
  (`gen` is already threaded into `train_steps`):

```python
# task: noisy pass — this is what forbids shrinking
y_pred_full = model.forward(x_full, noise_std=args.resid_noise_std, generator=gen)
l_task = torch.mean((y_pred_full[:, :num_x] - y) ** 2)

# probe fit + penalty: clean pass, full resolution
_, caches = model.forward(x_full, return_cache=True)
```

- The one-time init probe fit (`model.forward(x_full, return_cache=True)` feeding
  `probe.fit(cat_init...)`, ~line 527) already uses a clean pass — no change.
- Thread `resid_noise_std` into `LogregAdversarialConfig(...)` and the `[adv]`
  banner print.

## Step 4 — `train_adversarial.py` (secondary — user has largely moved off LDA)

Minimal consistent treatment; the probe path here already uses a separate pinned
sub-batch, so the split is free:

- Add `--resid-noise-std`, thread into `AdversarialConfig`.
- Task pass only: `model.task_output(x_full, noise_std=..., generator=gen)`.
- `_probe_caches` and `_delta_means_from_x` / `eval_delta_norms` stay **clean**.

## Verification

Smoke-level only — the real script takes several hours, so no end-to-end run and
no σ sweep here; hand off to the user to run and inspect.

1. **Quick calibration.** On `runs/nx32/checkpoints/best.pt`, print per-layer
   clean activation RMS and `‖Δμ‖` between c=1 and c=2 (one forward pass, seconds).
   Confirm σ=0.1 sits well below `‖Δμ‖` (expected O(1)); report the numbers rather
   than silently changing the default if the measured scale is far off.
2. `--help` on both training scripts; `black --check` on all touched files.
3. `noise_std=0.0` is bit-identical to today's `forward` — this is what guards
   every other call site (`train_probe.capture_layers_dict`, `data.eval_max_err`,
   `analytic.py`, `adversarial_report.py`).
4. With `noise_std>0`: `caches[0]` identical across repeated calls, `caches[1:]`
   differ, and repeated calls with a re-seeded `generator` reproduce exactly.
5. `AdversarialConfig.from_dict` / `LogregAdversarialConfig.from_dict` on an
   existing pre-noise checkpoint backfills `resid_noise_std=0.0`.
6. Existing tests: `python -m pytest test_torch_logreg.py`.
7. Short smoke run of each script under a throwaway `--tag` with `--max-iters` ~200,
   confirming it trains, checkpoints, and `l_task` floors near σ².

## Risks / caveats

- **Redundancy weakens the floor.** The model can encode c across k directions
  for effective SNR `a√k/σ`, so with d_model=256 the true floor may sit up to
  ~16× below σ. This is the reason for defaulting to 0.1 rather than 0.05. If it
  still produces no behavioral change, raise σ rather than lowering it.
- **Task cost is real and expected.** On the noisy pass, task MSE floors near
  σ² = 1e-2 and max_err near 3–4σ. Noise at cache num_blocks−1 reaches `y` through
  the last block's residual path and `W_U`, which is unavoidable given that layer
  must be penalized. `data.eval_max_err` runs on a clean pass, so the reported
  max_err still reflects the learned function and should stay far better.
