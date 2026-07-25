# Move rarely-used CLI flags to a config file (+ `--fork-from`)

## Context

`train_adversarial_logreg.py` had 27 flags across 6 argument groups. Most of
the probe/data/optimizer hyperparameters were rarely touched run-to-run,
bloating the CLI. Scoping conversation with the user (see below) settled on
moving most of them into a required `--config PATH` JSON file, and along the
way surfaced a second, related need: a clean way to change hyperparameters
mid-lineage without losing training progress, which became `--fork-from`.

This plan is the settled result of that conversation -- it supersedes the
"ask the user" scaffolding of the original version of this file wholesale.

## Settled design decisions

**File format:** JSON. `config.py` dataclasses already round-trip through
`to_dict`/`from_dict` (dataclass <-> dict <-> JSON), no new dependency.

**Three-way flag split** (replaces the old CLI/config precedence question --
there is no overlap, so no precedence merge is needed):

1. **Bookkeeping** -- run-control, never part of the saved experiment
   config, never frozen on `--resume`, always a CLI flag: `--tag`,
   `--resume`, `--fork-from`, `--tag-force`, `--log-interval`,
   `--ckpt-interval`, `--save-every-n`, `--max-iters`, `--probe-backend`,
   `--warmstart`/`--no-warmstart`, `--num-x`/`--d-model`/`--d-mlp`/
   `--num-blocks`. None of these get recorded for reproducibility (user's
   call: not worth it -- architecture is already separately saved via
   `ResidualMLPConfig` in the checkpoint, and `--warmstart` is moot once
   `train_adversarial.py` is sunset and superseded by `--fork-from`).
2. **CLI-common hyperparameters** -- part of the saved `LogregAdversarialConfig`,
   frozen on `--resume`, stay as CLI flags because they're touched often:
   `--lam`, `--penalty-layers`.
3. **Config-file-only hyperparameters** -- part of the saved
   `LogregAdversarialConfig`, frozen on `--resume`, no CLI flag at all,
   *required* keys in the `--config` JSON file (missing key = fail loudly,
   not a silent default): `--lam-warmup-iters`, `--class-threshold`,
   `--resid-noise-std`, `--probe-C`, `--probe-init-iters`,
   `--probe-loss-kind`, `--probe-subsample`, `--probe-retrain-interval`,
   `--x-p-outer`, `--x-threshold`, `--grad-clip`, `--batch-size`, `--lr`,
   `--seed`.

**"Fail loudly on missing key" mechanism:** give config-file-only fields on
`LogregAdversarialConfig` no Python-level default (`@dataclass(kw_only=True)`
lets required and defaulted fields coexist regardless of declaration order,
available since Python 3.10; repo runs 3.14). Constructing
`LogregAdversarialConfig(**cli_common_kwargs, **config_file_dict)` then raises
naturally if the file is missing a required key -- catch that `TypeError` and
re-raise as a `SystemExit` naming the missing key(s), matching this file's
existing `[error] ...` convention. This is orthogonal to the existing
`_LEGACY_DEFAULTS` backfill mechanism (`config.py`'s `_CheckpointConfigMixin`),
which stays untouched -- it protects genuinely old checkpoints predating this
feature, not something this plan needs to preserve compatibility with itself
(user confirmed no backward-compat requirement for this specific feature).

**Persisted copy:** every run that resolves a *new* `LogregAdversarialConfig`
(a fresh run or a `--fork-from`) writes `runs/<tag>/config.json` = the full
resolved config (`adv_config.to_dict()`, CLI-common + config-file fields
together) once, at tag-creation time. This is a new artifact -- today's
checkpoint-embedded copy requires loading a torch checkpoint to inspect;
`config.json` doesn't. **It is write-once and read-only thereafter** for the
lifetime of that tag: subsequent `--resume` calls never rewrite it (see
below), so hand-editing it post-hoc has no effect except tripping the
mismatch warning on the next resume. This "inert after creation" rule should
be stated in the module docstring and printed at startup, per the original
plan's own risk about silent-precedence support burden.

**`--resume <tag>` (existing flag, narrowed meaning):** continues the *same*
experiment. Model weights, optimizer state, and iteration count restore from
`runs/<tag>/checkpoints/last.pt`, exactly as today. All hyperparameters
(CLI-common and config-file) are restored from the checkpoint's embedded
`LogregAdversarialConfig` (`from_dict`) -- **not** re-read from CLI or the
config file; no hyperparameter-changing flags are accepted in this mode.
Only bookkeeping flags (e.g. a bigger `--max-iters` to keep training past
where it stopped) may differ across resume invocations. `runs/<tag>/config.json`
is read once for a sanity check -- compare (`==`, free via the dataclass) its
fields against the checkpoint's restored config; if they differ (someone
hand-edited the file since it was written), print a warning and proceed with
the checkpoint's values, **leaving the disk file untouched** -- do not
overwrite it either to "fix" the mismatch or to reflect current values.

**`--fork-from <source_tag>` (new flag, combined with `--tag <new_tag>`):**
branches a new experiment off an existing run's progress. Requires `--tag` to
name a *new* tag (same collision guard as a fresh run: refuses to clobber an
existing `runs/<new_tag>` without `--tag-force` -- no code change needed
here, the existing `if run_dir(args.tag) exists and not args.resume` check
already covers it since `--fork-from` is a distinct flag from `--resume`).
Model weights + optimizer state + iteration count load from
`runs/<source_tag>/checkpoints/last.pt` (same restore mechanics as
`--resume`, just reading from `source_tag`'s directory instead of the
current tag's). `history.json` for the new tag starts as `source_tag`'s
history truncated to the fork point, concatenated with the new tag's own
entries going forward -- so loss curves stay continuous for plotting.
Unlike `--resume`, hyperparameters are **freshly resolved** from this
invocation's CLI-common flags + `--config` file (exactly like a fresh run),
not inherited from the source tag. The new tag's `config.json` additionally
records `forked_from: {tag: <source_tag>, iter: N}` for provenance --
recoverable both explicitly (this field) and implicitly (the history's
iteration numbers jump from the inherited portion to the new tag's own,
continuing monotonically from N).

**`--warmstart`/`--no-warmstart`:** unchanged, stays CLI (pre-decided,
independent of "rarely used"). Note it's vestigial under `--resume` and
`--fork-from` -- still required by argparse's mutual-exclusion check, but its
loaded weights are immediately overwritten by the restored checkpoint. Not
worth special-casing given `--warmstart` itself is expected to be superseded
by `--fork-from`-style lineage once `train_adversarial.py` is sunset.

## Implementation steps

1. **`config.py`:** split `LogregAdversarialConfig` fields per the three-way
   list above. Config-file-only fields lose their Python default (use
   `@dataclass(kw_only=True)`); CLI-common fields (`lam`, `penalty_layers`)
   keep theirs (argparse still references them as flag defaults). Leave
   `_LEGACY_DEFAULTS`/`to_dict`/`from_dict` behavior unchanged.
2. **`load_run_config` helper** (new, likely in `train_adversarial_logreg.py`
   or `config.py`): reads the `--config` JSON file, merges with CLI-common
   args into one `LogregAdversarialConfig`, converts a missing-key
   `TypeError` into a `SystemExit` naming the specific missing key(s).
3. **`parse_args()`:** remove the 14 config-file-only flags. Add `--config
   PATH` (required unless `--resume`) and `--fork-from TAG` (mutually
   exclusive with `--resume`). Keep `--lam`, `--penalty-layers`, and all
   bookkeeping flags as-is.
4. **Tier-2 validation** (`__main__` guard, alongside the existing
   `--warmstart`-exists check): `--resume`/`--fork-from` mutual exclusion;
   `--config` required unless `--resume`; if `--fork-from`, check
   `runs/<source_tag>/checkpoints/last.pt` exists early, before mutating
   `runs/<tag>`.
5. **`main()`:** branch fresh / resume / fork.
   - Extract the current inline "restore model+optimizer+iter+history from a
     checkpoint" block (today gated on `args.resume`) into a small helper
     parameterized by *which tag's directory* to restore from, since resume
     and fork now share it (resume: `args.tag`; fork: `args.fork_from`).
   - Fresh/fork: build `adv_config` via `load_run_config`; write
     `runs/<tag>/config.json` once.
   - Resume: build `adv_config` via `LogregAdversarialConfig.from_dict` off
     the restored checkpoint; read-and-diff (not rewrite)
     `runs/<tag>/config.json` against it, warning on mismatch.
   - Fork: also seed `history` from `runs/<source_tag>/log/history.json`
     truncated to the restored iteration, and add `forked_from` to the
     written `config.json`.
6. **Docstring** update: describe the three run modes (fresh /
   `--resume` / `--fork-from`) and that `config.json` is write-once,
   read-only-thereafter.

## Verification

- Unit test: `LogregAdversarialConfig` -> JSON -> back reproduces the
  original (dataclass `__eq__`); a config file missing a required key raises
  a clear, key-naming error.
- Smoke test: fresh run with `--config` + `--lam`/`--penalty-layers` on the
  CLI; confirm `runs/<tag>/config.json` matches the checkpoint-embedded
  config exactly.
- Smoke test: `--resume` with a different `--max-iters` continues training
  past the original stop point; confirm `runs/<tag>/config.json` is
  byte-identical before/after (untouched).
- Smoke test: hand-edit `runs/<tag>/config.json` after a fresh run, then
  `--resume`; confirm a warning prints, training proceeds on the checkpoint's
  values, and the hand-edited file is left as-is (not overwritten in either
  direction).
- Smoke test: `--fork-from <source_tag> --tag <new_tag>` with a different
  `--config`/`--lam`; confirm the new tag's `config.json` differs from the
  source's as expected, includes `forked_from`, and `history.json`'s
  iteration numbers increase monotonically across the fork boundary.
- Regression check: a colliding `--tag` under `--fork-from` without
  `--tag-force` still errors, same as a fresh run would.

## Risks / caveats

- Splitting hyperparameters across CLI + config file remains a complexity
  cost in principle, but the resulting CLI is now genuinely small (2
  hyperparameter flags -- `--lam`, `--penalty-layers` -- plus bookkeeping/init
  flags), so this should net-declutter rather than just relocate complexity.
- `--fork-from` reuses most of `--resume`'s restore-from-checkpoint code path;
  refactor carefully so the two can't silently diverge in *what* they
  restore (weights/optimizer/iter should be identical between them -- only
  the hyperparameter-config source differs).
- The "`config.json` is write-once, read-only-diff-after" rule needs to be
  discoverable (module docstring + startup print), or a hand-edit that
  silently does nothing becomes exactly the kind of "why didn't my change
  take effect" support burden the original plan's risk section warned about.
- `_LEGACY_DEFAULTS` must stay untouched -- it's an unrelated, pre-existing
  concern (old-checkpoint compatibility), not something this plan's
  "no backward compat needed" scoping applies to.
