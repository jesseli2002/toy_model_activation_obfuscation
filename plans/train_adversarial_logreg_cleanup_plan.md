# `train_adversarial_logreg.py` internal cleanup

## Context

Tech-debt review of `train_adversarial_logreg.py` (the entry point for
adversarial training against a simultaneous, stateful logreg probe) surfaced
several fixable issues, discussed and settled with the user. This plan is
scoped **only** to this file (plus the small `config.py`/`probe_backend.py`
touch points it implies) -- it deliberately does **not** attempt to
deduplicate shared orchestration logic against `train_adversarial.py`, since
that script is slated for a possible near-term sunset and investing in
cross-script dedup against code that may be deleted is premature. If that
sunset doesn't happen, cross-script dedup can be a separate future plan.

This plan can land independently of `config_dataclass_dedup_plan.md` (no
hard ordering dependency either way -- `LogregAdversarialConfig` gains a base
class there, but nothing here depends on that change).

## Items

### 1. Checkpoint-save closure

Currently `save_checkpoint(path, record, model, opt, best_loss, hidden_layers,
adv_config)` is called at 4 sites in `main()` (best-loss improvement,
`ckpt-interval`, `save-every-n`, final save, plus the `KeyboardInterrupt`
handler) with the same argument list repeated each time. `train_adversarial.py`
already solves this with a closure defined once:

```python
def save(path):
    save_checkpoint(path, record, model, opt, best_loss, hidden_layers, adv_config)
```

`record` and `best_loss` are read from the enclosing scope at call time
(closures are late-binding on reads), which is safe even though both are
reassigned across iterations. Port this pattern in; replace all 4+1 call
sites with `save(path)`.

### 2. Class-imbalance asserts: pre-call placement

Keep both existing asserts (they're useful signal-of-intent, not just
paranoia) but reposition each to sit immediately before the call whose
precondition it's guarding, and simplify the subsample path:

```python
label = x_full[:, num_x] >= args.class_threshold
cat_live = concat_caches_torch(caches, hidden_layers)

t_probe0 = time.time()
if it % args.probe_retrain_interval == 0:
    X = cat_live.detach()
    X_fit = X[:: args.probe_subsample]        # no-op slice at 1
    label_fit = label[:: args.probe_subsample]
    assert label_fit.any() and (~label_fit).any(), (
        "subsampled probe batch has only one class present -- lower "
        "--probe-subsample or raise --batch-size."
    )
    fit_probe(probe, X_fit, label_fit, PROBE_STEP_MAX_ITER)
    affine = probe.get_affine(device)
probe_dt = time.time() - t_probe0

assert label.any() and (~label).any(), (
    "batch has only one probe class present -- check --class-threshold "
    "against c's range."
)
l_probe = score_penalty(cat_live, affine, label, args.probe_loss_kind)
```

The `if args.probe_subsample > 1: ... else: ...` branch is dropped entirely
(`X[::1]` and `label[::1]` are no-ops). The full-batch assert **must** stay
outside the `if it % probe_retrain_interval == 0:` block and unconditional --
`score_penalty` runs every iteration regardless of whether the probe was
just refit, so the guard needs to run every iteration too.

### 3. History-dict helper

Replace the two hand-built, already-drifted history dicts (log-interval:
8 keys; final: 6 different keys, missing `probe_dt`/`model_dt`/`lam_eff`,
using `best_loss` in place of `loss`) with one helper built from
`TrainRecord`:

```python
def _history_entry(record: TrainRecord, **extra) -> dict:
    d = dataclasses.asdict(record)
    del d["affine"]  # tensors aren't JSON-serializable, not needed in history
    d.update(extra)
    return d
```

- Log-interval site: `history.append(_history_entry(record, max_err=me))`.
- Final site: `history.append(_history_entry(record, loss=best_loss, l_task=None, l_probe=None, max_err=me, final=True))`.

One schema, one place that defines it; per-call `**extra` covers where the
final entry legitimately differs rather than letting the two literal dicts
drift independently.

### 4. Validation / provisioning split

Full separation of "validate all args" from "provision all resources" isn't
achievable: hidden-layer-range validation depends on `num_blocks`, and in
warm-start mode `num_blocks` only exists after the checkpoint is loaded. So
instead of a strict two-phase split, make each tier an explicit, separately
named step:

- **Tier 1 -- pure-arg validation** (`parse_args`, no I/O): unchanged --
  `--warmstart`/`--no-warmstart` mutual exclusion stays exactly where it is.
- **Tier 2 -- cheap existence check** (right after `parse_args()`, before the
  heavy-import boundary): move `os.path.exists(args.warmstart)` up to fire
  immediately, using only `os` (already imported pre-boundary). This is a
  behavior improvement, not just refactor-for-its-own-sake: today the check
  happens *after* run-dir setup, so `--tag-force` can delete an existing
  `runs/<tag>` directory before a bad `--warmstart` path is ever caught.
  Moving the check earlier closes that latent footgun.
- **Tier 3 -- model-dependent validation**: extract the warmstart-vs-scratch
  branch into its own `resolve_model(args, device) -> (model, model_config)`
  function, and call `_resolve_hidden_layers(args.penalty_layers,
  model_config.num_blocks)` as an explicit, visibly-sequential step right
  after it, rather than both being inlined into the middle of `main()`.

### 5. Doc fixes

- Remove the stale comment on the `ConvergenceWarning` filter (`main()`,
  currently reads "...doesn't belong in probe_backend.py, which is meant to
  be reusable") -- this documents a past agent's own uncertainty, not
  current design; the filter itself stays (it's a deliberate tradeoff between
  probe convergence and training performance, not a bug).
- Rewrite `save_checkpoint`'s docstring to state the contract, not the
  mechanism: "Save all training state needed to resume from disk, atomically
  (a SIGINT can't corrupt the file)." Drop the line narrating *how*
  `_defer_keyboard_interrupt` achieves that -- that belongs on
  `_defer_keyboard_interrupt` itself, which already documents it.
- `--probe-subsample` / `--probe-retrain-interval` help text: add a `(see PR
  #36)` reference (number only, no full URL) pointing at the performance
  motivation for these flags.

## Steps (suggested order -- each is an independently reviewable diff)

1. History-dict helper (#3) + unit test.
2. Checkpoint-save closure (#1).
3. Class-imbalance assert repositioning + subsample simplification (#2).
4. Validation/provisioning split (#4) + unit test for the extracted pieces.
5. Doc fixes (#5).

## Verification

- New unit tests:
  - `_history_entry`: schema has no `affine` key; `**extra` both adds new
    keys and overrides colliding ones (e.g. `loss`).
  - `_resolve_hidden_layers`: `"all"` resolves to `1..num_blocks-1`; layer 0
    raises `SystemExit`; layer `== num_blocks` warns (capture via
    `capsys`/`caplog`) and proceeds; out-of-range raises `SystemExit`.
  - Tier-2 existence check: a bad `--warmstart` path exits before any
    `runs/<tag>` directory is touched (this specific check needs no torch,
    so it's a fast, light unit test -- distinct from `resolve_model`, which
    does need torch to actually load a checkpoint and stays an integration-
    level check).
- `--help` still exits before the heavy-import boundary (unchanged behavior,
  quick manual check).
- Smoke run: short end-to-end run under a throwaway `--tag`, `--max-iters`
  ~100-200, confirming training/checkpointing/`history.json` still work,
  plus one explicit `--resume` round-trip to confirm state restores
  correctly (this path was previously hard to unit-test given how coupled
  `main()` was; still an integration-level check here, not a unit test).
- `black --check train_adversarial_logreg.py`.

## Risks / caveats

- The full-batch assert (#2) must not end up nested inside the
  `probe_retrain_interval` conditional by accident during the move --
  double check indentation/placement lands it unconditional, right before
  `score_penalty`.
- Tier-2's earlier existence check changes error *ordering* (now fails
  before `runs/<tag>` is touched) -- this is intended and was discussed as a
  fix, not a regression, but call it out in the PR description since it's an
  observable behavior change beyond pure refactor.
