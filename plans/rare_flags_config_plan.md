# Move rarely-used CLI flags to a config file

## Context

`train_adversarial_logreg.py` has 27 flags across 6 argument groups. A
handful are exercised nearly every run (`--warmstart`, `--tag`, `--lam`,
`--max-iters`); most of the rest (probe hyperparameters, subsampling/retrain
cadence, class threshold, etc.) are rarely touched run-to-run. The user wants
to move the rarely-touched ones out of the CLI into a config file, which
should also be persisted into the run's own directory (`runs/<tag>/...`) for
reproducibility -- similar in spirit to how `LogregAdversarialConfig` is
already embedded in every checkpoint, but as a plain, human-readable file
rather than something you have to load a checkpoint to inspect.

The user explicitly does **not** want to decide which flags move as part of
scoping this plan -- that decision, and several design choices below, are
left for whoever executes this plan to raise with the user first.

`--warmstart`/`--no-warmstart` specifically stays a CLI flag regardless of
the rest of this plan -- the user has said they prefer that construction as
CLI, independent of "is this a rarely-used flag."

## Before writing an implementation section: ask the user

This plan is intentionally left unfinished past this point. The executing
agent's first step should be a conversation with the user (not a unilateral
decision) covering:

1. **Which flags move.** A concrete list, going group-by-group through the
   current CLI (`g_init`, `g_adv`, `g_probe`, `g_opt`, `g_book` in
   `parse_args`), each flagged as "stays CLI" / "moves to config file" /
   "arguable, ask." `--warmstart`/`--no-warmstart` is pre-decided (stays).
2. **File format.** JSON is the path of least resistance -- `config.py`'s
   dataclasses already round-trip through `to_dict`/`from_dict` (see
   `config_dataclass_dedup_plan.md`) which is exactly dataclass <-> dict <->
   JSON, no new dependency. TOML/YAML would need a new library dependency
   (stdlib `tomllib` only reads, doesn't write) for not-obviously-better
   ergonomics here. Recommend JSON, but confirm with the user before
   building around it.
3. **Precedence.** If a value is set in both the config file and on the CLI,
   which wins? (Recommend: CLI explicit override wins, config file fills in
   the rest, dataclass field default is the last resort -- but confirm.)
4. **Where the persisted copy lives, and what it's a copy of.** The
   `LogregAdversarialConfig` dict is already saved inside every checkpoint
   via `save_checkpoint`. Does "also get saved to the corresponding run
   directory" mean: (a) that's already satisfied and nothing new is needed,
   or (b) the user wants a standalone `runs/<tag>/config.json` (or similar)
   readable without loading a torch checkpoint? If (b), decide the exact
   path/filename and whether it's written once at run start or refreshed
   each checkpoint.
5. **`--resume` interaction.** On `--resume`, should the config file be
   re-read, or should the checkpoint's already-restorable
   `LogregAdversarialConfig.from_dict` be treated as authoritative (current
   behavior for everything else on resume)? These can disagree if the config
   file was hand-edited between runs.

## Once scope is settled

Come back and fill in: a settled-design-decisions table (answers to the 5
questions above), an implementation steps section (likely: a
`load_run_config(path) -> dict` helper, argparse wiring so config-file-eligible
flags default to a sentinel rather than their current hardcoded default so the
merge step can tell "explicitly passed on CLI" from "using the default"),
and a verification section (round-trip test for the config file loader;
smoke test that a config-file-driven run and an equivalent all-CLI-flags run
produce the same `LogregAdversarialConfig`).

## Risks / caveats (preliminary, revisit once scoped)

- Splitting configuration across two sources (CLI + file) is itself a
  complexity cost -- make sure the "which flags move" list in question 1
  actually reduces total cognitive load rather than just relocating it (e.g.
  don't migrate a flag that's almost always overridden anyway).
- Whatever precedence rule is chosen (question 3) needs to be discoverable
  at the CLI (e.g. in `--help` text or an early startup print), or debugging
  "why did my run use C=1.0 instead of the C=2.0 in my config file" becomes
  its own support burden.
