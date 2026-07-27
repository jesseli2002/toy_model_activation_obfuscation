# Run-config artifacts for `train_adversarial_logreg.py`

## Context

Three related problems, all downstream of one design flaw.

`save_checkpoint` (`train_adversarial_logreg.py`) splats `**adv_config.to_dict()` into the
checkpoint's **top level**, so hyperparameters share a namespace with training state
(`model`, `opt`, `iter`, `best_loss`, `probe_w/b/layers`). `--resume` then hands that whole
flat dict to `LogregAdversarialConfig.from_dict(rck)`, which warns about every state key it
doesn't recognize. Reproduced with a fresh tiny run followed by `--resume`:

```
UserWarning: LogregAdversarialConfig.from_dict: dropping unrecognized key(s)
['best_loss', 'config', 'iter', 'model', 'opt', 'probe_b', 'probe_layers', 'probe_w']
```

Filtering at the call site would hide the message but keep the flaw: the two namespaces
stay merged, so a future config field named `iter` or `config` would silently collide, and
`from_dict`'s warning — whose real job is flagging a checkpoint written by a *newer*
version — stays permanently useless, because state keys are always "unrecognized".

Separately, `runs/<tag>/config.json` records only the adversarial hyperparameters. The model
architecture (`num_x`, `d_model`, `d_mlp`, `num_blocks`) appears nowhere in the run
directory, so a run's settings cannot be audited or compared from one place. And the
architecture CLI flags are accepted-then-silently-discarded under `--resume`/`--fork-from`,
where model dimensions cannot legally change anyway.

**Outcome:** a run directory becomes a complete, self-describing record of what defined the
run; the checkpoint stays authoritative for resuming and stops flattening config into state;
no warning fires on any normal use.

## Constraints and decisions

Settled with the user; treat these as given rather than re-deciding them.

- **The checkpoint stays authoritative on `--resume`.** `config.json` is a read-only record,
  not an input. Hand-editing it is detected and warned about, never applied.
- **No backward compatibility with existing checkpoint files.** Checkpoints written before
  this change do not need to resume, and there is no legacy-layout fallback to write. Old
  runs under `runs/` are historical artifacts.
- **`--fork-from` keeps requiring a complete `--config`** — no inheritance, no partial
  overlay. Everything except model dimensions is freely changeable at fork time.
- **Per-invocation bookkeeping flags stay out of the run config.** `--max-iters`,
  `--log-interval`, `--ckpt-interval`, `--save-every-n`, `--probe-backend` legitimately
  differ on every invocation of the same run; recording them in a write-once file would
  misrepresent them.
- **Architecture flags become an error** when explicitly passed alongside
  `--resume`/`--fork-from`, rather than being silently ignored.
- **The run-config schema is described in exactly one place** — the dataclass below. Do not
  restate the key list in module docstrings, function docstrings, or comments; refer to the
  class instead. This is the main thing to get right in review.

## Target layout

Two artifacts per run directory, both written once at tag creation (fresh run or
`--fork-from`), both read-only thereafter. `--resume` writes neither.

- `runs/<tag>/input_config.json` — a **verbatim copy** of the file passed to `--config`. Its
  purpose is to stay directly reusable as a later `--config` argument, so the
  copy-an-existing-run's-config-and-edit-one-knob fork workflow keeps working.
- `runs/<tag>/config.json` — the **fully resolved** run config: model architecture,
  adversarial hyperparameters (including `--lam`, which never appears in the input file, and
  `penalty_layers` resolved from `"all"` to an explicit list), and fork provenance.

Checkpoints keep their state keys unchanged, with the adversarial config **nested** under a
single key instead of splatted:

```
{ model, config, opt, iter, best_loss, probe_w, probe_b, probe_layers,
  adv_config: { ...LogregAdversarialConfig... } }
```

`ResidualMLPConfig` stays under `config` — `ResidualMLP.load` (`model.py`) needs it there,
and it keeps a stray `.pt` self-describing. `config.json`'s model block is written *from*
`model.config`, so it is derived, never independently resolved.

## Changes

### `config.py`

Add the dataclass that owns the `config.json` schema. It is the single source of truth for
that file's shape:

```python
@dataclass
class ForkedFrom:
    tag: str
    iter: int


@dataclass
class LogregRunConfig:
    """Complete definition of one train_adversarial_logreg.py run, serialized
    to runs/<tag>/config.json as that run directory's read-only record.

    Sole definition of that file's schema -- document it here, not in the
    script that writes it.

    `model` is frozen at tag creation (--fork-from inherits it from the source
    checkpoint); `adversarial` is freshly resolved for every new tag;
    `forked_from` is present only on a forked tag.
    """

    model: ResidualMLPConfig
    adversarial: LogregAdversarialConfig
    forked_from: ForkedFrom | None = None
```

with `to_dict()` (`dataclasses.asdict` recurses through the nested dataclasses; omit
`forked_from` entirely when `None`) and a `from_dict()` that delegates to each member's
existing `from_dict`, so `_LEGACY_DEFAULTS` backfill still applies per block.

No changes to `_CheckpointConfigMixin`, `from_dict`, or `_LEGACY_DEFAULTS` semantics — the
existing warning is correct and becomes meaningful again once nothing but real config fields
reach it.

### `train_adversarial_logreg.py`

1. **`save_checkpoint`** — pass `adv_config=adv_config.to_dict()` instead of
   `**adv_config.to_dict()`.

2. **Resume path** — `LogregAdversarialConfig.from_dict(rck["adv_config"])`. A checkpoint
   without that key predates this change; raise a `SystemExit` in this module's `[error]`
   style saying so and pointing at `--fork-from`/a fresh run, rather than letting a raw
   `KeyError` escape.

3. **`_write_config_json`** — rename to something covering both artifacts (e.g.
   `_write_run_config`). Takes `model_config`, `adv_config`, `forked_from`, and the path of
   the `--config` file; assembles a `LogregRunConfig`, writes `config.json`, and copies the
   input file to `input_config.json` (`shutil.copyfile`; `shutil` is already imported).
   Called with `model.config` — the model is already built at both the fresh and fork call
   sites.

4. **`_check_config_json`** — with no legacy fallback this collapses to: load `config.json`,
   `LogregRunConfig.from_dict`, compare `.adversarial` against the checkpoint-restored
   config and `.model` against `model.config`, warn on mismatch or on the file being absent
   entirely. Still never rewrites the file.

5. **Architecture flags** — add a small `argparse.Action` that records explicitly-passed
   dests on the namespace. It must set the value normally, so `default=` and
   `ArgumentDefaultsHelpFormatter` keep working and `--help` still shows the defaults. Apply
   it to `--num-x/--d-model/--d-mlp/--num-blocks`; `parse_args` then calls `p.error` if any
   of those was explicit alongside `--resume`/`--fork-from`, naming the offending flags and
   stating that the architecture comes from the checkpoint being restored.

6. **Module docstring** — keep the three run-mode descriptions (fresh / `--resume` /
   `--fork-from`), and replace the `config.json` prose with a two-line description of the
   two run-directory artifacts by *purpose*, deferring the schema to `LogregRunConfig`. No
   key lists.

### `train_adversarial.py`

One-line change in its `save()` closure: nest `adv_config=adv_config.to_dict()` the same
way, so both training scripts write one checkpoint layout and `adversarial_report.py` needs
only one accessor. Its `--resume` path rebuilds `adv_config` from CLI args and never reads
it back from the checkpoint, so nothing else there is affected.

### `adversarial_report.py`

Reads three adversarial keys straight off the checkpoint's top level, which would silently
fall back to defaults once nested: `ck.get("class_threshold", 1.5)`, `ck.get('lam')` /
`ck.get('init')`, and `ck.get("penalty_layers")`. Route them through a local
`ck.get("adv_config", {})` — the `{}` default matters, since this script is also pointed at
checkpoints from `train_probe.py`/`train_model_plot.py` that carry no adversarial config at
all, and the existing per-key `.get` defaults must keep covering that case. `probe_w` /
`probe_b` / `probe_layers` stay top-level: they are probe *state*, not config.

## Tests

`test_config.py` — round-trip `LogregRunConfig` through `to_dict`/`from_dict`; confirm
`forked_from` is absent from the dict when `None` and restored when present; confirm per-block
`_LEGACY_DEFAULTS` backfill still applies through the nested `from_dict`.

`test_train_adversarial_logreg.py` — update the existing `TestConfigJson` cases
(`test_write_then_check_matching_config_prints_no_warning`,
`test_check_config_json_mismatch_warns_and_leaves_file_untouched`,
`test_write_config_json_records_forked_from`) for the new schema, the added `model_config`
argument, and the renamed writer. Add:

- **Regression for the reported bug**: build a dict shaped like `save_checkpoint`'s output,
  run it through the resume-path extraction under `warnings.simplefilter("error")`, and
  assert the config round-trips with no warning raised.
- A genuinely unrecognized key *inside* the nested `adv_config` still warns — the warning's
  real job, which must survive this change.
- A checkpoint with no `adv_config` key exits with the `[error]` message from change (2).
- `--num-x` (or another arch flag) with `--resume`, and again with `--fork-from`, exits
  non-zero naming the flag. Follow the existing subprocess-based CLI test
  (`test_bad_fork_from_tag_exits_before_touching_run_dir`) for the pattern.
- `input_config.json` is written as a byte-for-byte copy of the `--config` file, and the
  resolved `config.json` differs from it (contains `lam`, the model block, and resolved
  `penalty_layers`).

## Verification

1. `pytest test_config.py test_train_adversarial_logreg.py`
2. Reproduce the original report, now clean — `-W error::UserWarning` turns any regression
   into a hard failure:
   ```
   python -W error::UserWarning train_adversarial_logreg.py --config configs/default.json \
       --num-x 4 --d-model 8 --num-blocks 3 --max-iters 2 --tag cfgcheck --tag-force
   python -W error::UserWarning train_adversarial_logreg.py --resume --tag cfgcheck --max-iters 4
   ```
   Inspect `runs/cfgcheck/` for both artifacts: `config.json` with `model` + `adversarial`
   blocks, and `input_config.json` identical to `configs/default.json`.
3. Fork it, reusing the recorded input config to prove that workflow:
   ```
   python -W error::UserWarning train_adversarial_logreg.py --fork-from cfgcheck \
       --tag cfgcheck_fork --config runs/cfgcheck/input_config.json --lam 0.9 --max-iters 6
   ```
   Confirm `runs/cfgcheck_fork/config.json` carries `forked_from`, the inherited `model`
   block, and `lam: 0.9`.
4. Confirm the arch-flag guard: `--resume --tag cfgcheck --d-model 16` errors and names the
   flag.
5. `python adversarial_report.py --tag cfgcheck` runs against a new-layout checkpoint. Also
   run a short fresh `train_adversarial.py` run and report on it, to check the shared
   accessor against that script's checkpoints.
6. Delete the scratch tags afterwards (`runs/cfgcheck*` and any `train_adversarial.py` scratch tag).
