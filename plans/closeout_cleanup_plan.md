# Repo closeout cleanup plan

Plan only — nothing here is implemented yet.

## Goal

A reproducer, starting from a fresh clone, should be able to **recreate the
process that produced the writeup figures, in a guided manner**. "Guided" is the
operative word: the repo should walk them through the background work too (how
the loss/AUROC thresholds were picked, how sweeps were compared against each
other), not just hand them a script that emits a PNG. Failed experiments are out
of scope — they don't need to be reproducible, and mostly shouldn't be kept.

A secondary goal, worth real effort on its own: **keep the trained models
themselves.** A reproducer who wants to poke at a model that learned to hide
from a probe should be able to load one and experiment, without a GPU-week.
This is why checkpoints are kept even where a cached scalar would redraw the
figure.

## How to execute this plan

Three things make this different from a normal plan, and an executing agent
must respect all three:

1. **This is interactive, not autonomous.** Most steps involve a judgement call
   about what's worth keeping — that judgement is the user's, not the agent's.
   Check in at every step boundary and whenever a step turns up something the
   plan didn't anticipate. Do not batch several steps and present a fait
   accompli.
2. **Commit locally; do not push to GitHub.** This plan puts potentially large
   binary data under version control. A local commit is reviewable and
   revertable; a push is neither. Nothing here goes to a remote — GitHub or
   Hugging Face — until the user says so explicitly, after reviewing the local
   commits. In particular, do *not* use the usual `git_push` / PR workflow.
3. **"Untrack" never means "delete".** Every removal in this plan is:
   ```
   git rm --cached <path>          # or -r --cached for a directory
   echo '<path>' >> .git/info/exclude
   ```
   The file stays on disk. `.git/info/exclude` (not `.gitignore`) is the right
   home for these, because they're this working copy's private state, not
   something a reproducer's clone should carry. Nothing in this plan authorises
   an `rm`.

`plans/` is itself a staging ground for this work and is dropped only at the
very end, on explicit user command — see §6. It is entirely possible that this
plan finishes while `plans/` still exists, and that one or more further planning
rounds happen first.

## 0. Sizing — what we're actually dealing with

Measured, not estimated. Details per run in
[`plans/closeout_run_manifest.md`](closeout_run_manifest.md).

| | Size |
|---|---|
| `runs/` in full (1423 run dirs) | **70 GB** |
| The 525 candidate runs, full directories | 9.0 GB |
| The same 525, keeping only `{best,last}.pt` + configs + logs | **764 MB** |
| …of which `checkpoints/{best,last}.pt` | 620 MB |
| …of which `logs/history.jsonl` | 144 MB |
| `plot/metrics_cache.json` | 200 KB |

Two consequences:

- **Dropping the `iter_*.pt` series is a 12× trim** (9.0 GB → 764 MB) and is
  where essentially all the savings come from. Nothing else is worth optimising
  by comparison.
- **Local disk is not a constraint.** 203 GB free on the volume. `runs_publish/`
  can be a plain dereferencing copy (`cp -L`) — no hardlink/symlink games
  needed. Being selective about *which runs* get copied still matters for
  reviewability and for whatever we eventually upload; being selective to save
  local disk does not.

### Where the 764 MB should live

764 MB is too much for a plain git repo and awkward for Git LFS (quota, and
every `git clone` pays for it even for someone who only wants to read the code).
**Recommend Hugging Face Hub** as the default: it is built for exactly this,
has no per-clone tax, versions the data, and lets the repo stay small with a
download snippet in `CLAUDE.md`.

Alternatives, in rough order of preference if HF is rejected: a GitHub Release
attachment (doesn't count against LFS quota, but unversioned and clumsy to
update); Git LFS in this repo (simplest to wire up, worst clone experience); a
separate data-only git repo.

**This decision gates any `git add` of a `.pt` file.** Because we're committing
locally and not pushing, a mistaken LFS commit is a history rewrite to undo.
Curate `runs_publish/` first (§2), decide the destination second, wire up
tracking third — in that order, with a check-in between each.

## 1. Which runs to keep

**Status: reviewed and agreed.** The full list — 525 runs, grouped, with
per-run sizes — is in [`plans/closeout_run_manifest.md`](closeout_run_manifest.md).
Summary:

| Group | Source of the requirement | Runs | ckpt |
|---|---|---|---|
| A | `sweep3_lam0_tr{0..14}` — the λ=0 baseline arm reused by `sweep7_analysis.py` | 15 | 11 MB |
| B | `sweep7_lam{0.0001…0.5}_tr*` — the λ sweep | 149 | 112 MB |
| C | `sweep17_lr0.0015_iter100k_lam{0.01,0.1}_tr{0..9}` — width, num_x=32 | 20 | 15 MB |
| D | `sweep11_lr0.0015_iter200k_lam{0.01,0.1}_tr{0..9}` — width, num_x=64 | 20 | 53 MB |
| E | `sweep14_lr0.0015_iter400k_lam{0.01,0.1}_tr{0..9}` — width, num_x=128 | 20 | 203 MB |
| F | `sweep18_layer{2,4,6,8,10}_lam{0.01,0.032,0.1}…_tr{0..9}` — probed layer | 150 | 113 MB |
| G | `sweep19_layers{2-4,2-6,2-8,2-10,4-8}_lam{…}_tr{0..9}` — layer pairs, Pareto view | 150 | 113 MB |
| H | `baseline_no_c` — c-blind control, from `train_no_c.py` | 1 | 0.2 MB |

`make_publish_plots.py` (`sweep7_lam0.1_tr0`) and
`make_publish_plot_train_dist_curve.py` (`sweep11_lr0.0015_iter200k_lam0.01_tr0`)
are already covered by B and D.

Findings from building the list, all now resolved:

- **`sweep7_lam0_*` no longer exists in `runs/`.** `sweep7_analysis.py` already
  reads its λ=0 arm from `LAM0_GLOB = "sweep3_lam0_tr*"`, so nothing is broken.
  `publish/sweep7_lam0_tr0/` still holds rendered figures under that name, but
  the writeup's `copy_plots.sh` pulls the λ=0 comparison figures from
  `publish/sweep3_lam0_tr0/best/` — so that `publish/` directory is stale
  output, not a live dependency. Nothing to preserve; just don't let it confuse
  the `CLAUDE.md` figure map.
- **`sweep19_layers2-6` (30 runs, 23 MB) is kept**, even though it appears in no
  `PARETO_COMPARISONS` entry and `sweep_layer_analysis.py` never touches it. The
  cost is negligible and it makes the sweep19 layer-pair family complete
  (2-4, 2-6, 2-8, 2-10, 4-8) rather than arbitrarily missing one.
- **Cache coverage cross-check:** of 795 `plot/metrics_cache.json` entries,
  360 distinct tags. 330 are in the keep set, and **the only 30 outside it are
  exactly the `sweep19_layers2-6` runs**. This says no *surprise* tag turned up,
  which is a useful sanity check — but it is not proof the enumeration is
  complete, because absence-from-cache clearly doesn't imply absence-of-consumer:
  164 keep-set runs have no cache entry at all (all of group A and all of group
  B, since `sweep7_analysis.py` uses `CKPT="best"` and a different
  `MetricSpec`). Those metrics recompute on first plot.
- **Kept scripts audited for run dependencies** (the goal statement puts the
  background work in scope, but the run-list rule only named the four figure
  script families, so these needed checking explicitly):
  - `sweep_threshold_report.py` — reads `RUN_GLOB = "sweep7_lam*_tr*"` only,
    entirely inside group B. No extra runs. (It uses `CKPT="last"` where
    `sweep7_analysis.py` uses `"best"`; both are kept per run, so fine.)
  - `sweep_group_report.py` — its three surviving examples (§4) point only at
    groups C/D/E/F/G. No extra runs.
  - `probe_ideal_y.py`, `analytic.py`, `train_no_c.py` — no *existing* run data
    needed. `train_no_c.py` is what *produces* `runs/baseline_no_c`; it's a
    standalone trainer with its hyperparameters as module constants, not a
    consumer.
- **`runs/baseline_no_c` is added to the keep set** (group H). 192 KB, so
  free. It doesn't fit the standard per-run shape — `checkpoints/last.pt` and
  `logs/{history.jsonl,report.md,README.md}`, but no `best.pt`,
  `config.json`, or `input_config.json`, because `train_no_c.py` writes a
  reduced layout. Copy what exists; don't synthesise the missing files.
  `logs/README.md` records the exact command that produced it and should be
  preserved as-is.

### What to keep per run

`checkpoints/best.pt`, `checkpoints/last.pt`, `config.json`,
`input_config.json`, `logs/history.jsonl`, `logs/report.md`. Drop the
`iter_*.pt` series.

`config.json` and `input_config.json` are deliberately included: between them
they are the ground truth of what a run actually ran with, which is why
`configs/` can shrink to a single example (§6).

`logs/history.jsonl` is 144 MB of the 764 MB. **Decision: keep it, in full.**
It's what makes `sweep_group_report.py`'s curves view and the train-loss
trajectory figures reproducible (`load_history` reads it directly), and
downsampling it would mean shipping something that isn't the raw log. Revisit
only if the chosen destination imposes a size limit.

## 2. Curate `runs_publish/`

A staging directory, built and iterated on **outside `runs/`** so the live run
tree is never mutated.

Note `runs_publish` matches **neither** `.gitignore` pattern (`runs` and
`runs_tmp*`). Add it to `.git/info/exclude` as the very first action — per the
§"How to execute" convention it's this working copy's staging state, not
something a clone should carry — so that 764 MB never shows up as untracked and
`git add`-able while it's being worked out.

1. `mkdir runs_publish`
2. For each kept tag, copy the six paths above, **dereferencing symlinks**.
   Group H (`baseline_no_c`) has a reduced layout — no `best.pt`, no
   `config.json`/`input_config.json`, plus an extra `logs/README.md` recording
   its exact training command. Copy what exists rather than failing on the
   missing files, and keep that README.
   `checkpoints/last.pt` is a symlink to an `iter_N.pt` in 513 of the 525 runs;
   copied as a link with `iter_*.pt` excluded, it dangles. `cp -L` (or
   `rsync -L`) fixes this — verify afterwards that `runs_publish` contains zero
   symlinks.
   - The 11 exceptions are `sweep3_lam0_tr{0..10}`, whose `last.pt` is already
     a real file.
   - `best.pt` is a real file everywhere; `best` and `last` never resolve to the
     same file, so both are genuinely two checkpoints.
3. Verify: every run has both `.pt` files non-empty, `config.json` parses, and
   the total matches §0's 764 MB.
4. Load one checkpoint per group and confirm it opens (**on the remote GPU box
   or by the user — not locally**; see the CPU/GPU rule in `CLAUDE.local.md`).

Only once this is reviewed does anything get committed or uploaded.

## 3. Make the shipped state usable without a GPU

Not a compute-savings argument — the metric recompute is minutes, not GPU-hours,
and the cache is a nice-to-have for iterating on plot layout. It's an
**accessibility** argument: a reproducer with no GPU should still be able to
redraw the aggregate figures, and today they can't.

Two independent blockers (`sweep_lib/metrics.py:136`, `sweep_lib/cache.py:33`):

1. Every accessor (`task_loss`, `auroc`, `linear_y_r2`, …) early-returns `None`
   on `not self.has_checkpoint(tag)` before consulting the cache.
2. The cache key embeds `file_fingerprint = "{st_size}:{st_mtime_ns}"`. Any
   transport that doesn't preserve mtime — a git clone, an HF download, a plain
   `cp` — misses every entry. The scheme is built around `rsync -a`.

Fix: content-hash the fingerprint (or fall back to one when mtime is
unavailable), and let an accessor serve a cache hit without `has_checkpoint`.
That's a genuine improvement regardless of publishing — the current scheme
silently invalidates on any copy.

Then ship `plot/metrics_cache.json` (200 KB) alongside. Verify on a
checkpoint-less tree that the aggregate scripts run to completion rather than
skipping every tag. Note from §1 that groups A and B aren't in the cache, so
either accept that the `sweep7` figure needs checkpoints, or regenerate the
cache to cover them before shipping.

## 4. Scripts

Built from an import graph over `git ls-files '*.py'`, rooted at the writeup
figure scripts. (Docstring cross-references aren't import edges, and entry
points have zero inbound refs by definition — neither is evidence of dead code.)

**Keep — produces a writeup figure or is imported by one that does:**
`make_publish_plots.py`, `make_publish_plot_train_dist_curve.py`,
`sweep7_analysis.py`, `sweep_width_analysis.py`, `sweep_layer_analysis.py`,
`adversarial_report.py`, `train_adversarial_logreg.py`, `sweep_lib/*`,
`probe_lib.py`, `probe_backend.py`, `probe_newton.py`, `torch_logreg.py`,
`checkpoint_lib.py`, `model.py`, `data.py`, `config.py`, `paths.py`,
`stableadamw.py`, `rate_meter.py`.

**Keep — decided:**
- `probe_ideal_y.py` — demonstrates the task is actually solvable, which is
  exactly the kind of background work a reproducer needs. Document it in
  `CLAUDE.md` (§5) so its purpose is discoverable; nothing imports it.
- `sweep_threshold_report.py` — how the loss/AUROC thresholds were chosen. This
  is the "background work" the goal statement calls for.
- `analytic.py` — backs the publicly linked analytic-solution writeup.
- `train_no_c.py` — trains the `baseline_no_c` c-blind control, which
  empirically confirms the analytic c-blind loss floor is reachable. That floor
  is the certificate the whole experiment leans on ("a model scoring below it
  must be reading c"), so it belongs in the reproducible set.

### Print the c-blind baseline loss

**New work item.** The baseline's achieved loss is currently only recoverable
by reading raw artifacts — the last line of
`runs/baseline_no_c/logs/history.jsonl` (`l_task = 5.563e-2` at iter 49999), or
by re-running the 50k-iteration training and watching its final
`[done]` line scroll past. `adversarial_report.py --tag baseline_no_c` prints a
max-abs-error and then explicitly skips everything else as a c-blind run, so it
doesn't surface the number either.

A reproducer needs to see, in one command, **the achieved baseline loss next to
the analytic floor it's testing** (`analytic.no_c_task_loss`), since the whole
point is the comparison. Options, cheapest first:

1. Teach `train_no_c.py` a `--report` flag that loads the existing run's
   history/checkpoint and prints achieved-vs-bound without training. It already
   computes `bound` and prints `eval/bound` during training, so the formatting
   logic exists — this is mostly a matter of not requiring a training loop to
   reach it.
2. Have `adversarial_report.py` print the achieved-vs-bound comparison in its
   §1 for c-blind runs, instead of only noting that probe analysis is skipped.

Prefer (1): it keeps the c-blind logic in the c-blind script and leaves
`adversarial_report.py` alone. Either way `CLAUDE.md` documents the command.
This runs a checkpoint load, so **the user or the remote box runs it, not the
agent locally.**
- `sweep_group_report.py` — **keep, but clean up.** See below.

**Untrack:**
- `sweep8_analysis.py`, `sweep13_analysis.py` — superseded sweeps, absent from
  the writeup, imported by nothing.
- `check_dead_relu.py`, `benchmark_logreg_gpu.py` — one-off diagnostic and a
  backend bake-off whose decision is long since made. *(Confirm: these were in
  the original delete list and weren't explicitly ruled on.)*
- `sweep_inspect_training.py` — *(confirm; not explicitly ruled on. Same class
  as the two above, but arguably an investigation tool like
  `sweep_threshold_report.py`.)*

**Defer — needs a closer look later:**
- `analytic_feasibility/` (8 files) — exploratory sympy work behind the analytic
  writeup. Some of it is worth keeping and some isn't; that assessment hasn't
  been made yet. Leave tracked and untouched for now, and revisit in a later
  round. Note `analytic_feasibility/initial_prompt.md` references
  `plans/high_level_plan.md`, so it's also a §6 dangling-reference site.

### `sweep_group_report.py` cleanup

Reframe it as **the documented example of ad-hoc comparison between sweeps** —
this is the script that shows *how* runs were compared while the research was
happening, which is squarely in scope for the guided-reproduction goal.

- It currently assigns `GROUPS` **twice** (`sweep_group_report.py:27` and `:35`);
  the first — the sweep7/11/14/17 model-size comparison — is dead, shadowed by
  the second. Collapse to a single assignment.
- Keep exactly these three examples, discarding the other ~11 commented blocks:
  1. **Layer pairs (sweep19)** — `"2"` / `"8"` / `"2,8"`. **This is the active,
     uncommented one** (it already is). Shows the does-penalizing-two-layers-help
     comparison.
  2. **Layer scan (sweep18)** — `"layer2"`…`"layer10"` at one λ. The headline
     layer comparison, and the closest analogue to `sweep_layer_analysis.py`.
  3. **Model size** — `"nx32"` / `"nx64"` / `"nx128"`. This is the currently-dead
     line-27 assignment; it moves in as a comment under the single `GROUPS`.
  All three point only at runs already in the §1 keep set, so nothing extra
  needs preserving.
- The docstring should say what the script is *for* (ad-hoc cross-sweep
  comparison, edit `GROUPS` to point it at whatever you're comparing) rather
  than describing whichever comparison happens to be active.

## 5. Docs

- **`README.md`** — human-written by the user, not generated. A project summary
  plus links to the full writeup. Only the publicly readable links belong here;
  the current stub's Google Doc links are private and useless to an outside
  reader. The [analytic solution writeup](https://jesseli2002.github.io/blog/blog/activation_obfuscation_analytic/)
  is public and stays.
- **`CLAUDE.md`** (tracked, new) — the operational half, and where the "guided"
  part of the goal actually gets delivered. Distinct from `CLAUDE.local.md`,
  which stays untracked and sandbox-specific; cross-check that no tracked file
  describes a workflow that only exists in this sandbox.

  Required contents (non-exhaustive):

  1. **Source structure** — what lives where and what each module is for.
     Entry points (`train_adversarial_logreg.py`, `adversarial_report.py`,
     the `sweep_*` and `make_publish_*` scripts) and how to invoke them; the
     supporting libraries (`sweep_lib/`, `probe_*`, `model.py`, `data.py`, …)
     as a second tier. Lead the training part with the one-command
     `--config configs/example.json` invocation (§6), so a reproducer can
     train something before understanding anything else.
  2. **What can be reproduced and how**, **ordered to match the writeup**. The
     writeup lives in a separate local checkout at
     `/work/blog/content/projects/toy-model-of-activation-obfuscation` (its
     Results section splits into three parts, each with its own page under
     `/work/blog/content/blog/`):
     - *Part 1 — analytic construction*: `analytic.py`, and the
       `analytic_feasibility/` material to the extent it survives §4's deferral.
       No run data needed. Alongside it, the **c-blind floor**: what
       `analytic.no_c_task_loss` certifies, and the command that prints
       `baseline_no_c`'s achieved loss against it (§4).
     - *Part 2 — single-run empirical result*: `make_publish_plots.py`, run
       against `sweep7_lam0.1_tr0` at `best` (the AUROC / curves / PCA / probe /
       steering / ROC-grid figures) and against `sweep3_lam0_tr0` at `best` for
       the two `L2_steer_dir_mag*` λ=0 comparisons.
     - *Part 3 — hyperparameter sweeps*: `sweep7_analysis.py` (λ sweep →
       `plot/sweep7/`), `make_publish_plot_train_dist_curve.py`
       (`sweep11_…_lam0.01_tr0` at `last` → the ID-vs-OOD figure),
       `sweep_width_analysis.py` (→ `plot/sweep_width/`),
       `sweep_layer_analysis.py` (→ `plot/sweep_layer/`, including the Pareto
       grid).

     The blog's `copy_plots.sh` scripts (in each writeup part's directory) are
     the authoritative figure→file map; use them to build this section rather
     than re-deriving it, but state the *commands*, not just the output paths.
  3. **The background work**, so the process is reproducible and not just the
     figures: `sweep_threshold_report.py` for how the loss/AUROC thresholds were
     chosen, `sweep_group_report.py` as the worked example of ad-hoc cross-sweep
     comparison, `probe_ideal_y.py` for why the task is solvable at all.
  4. **How run configs work** — how `--config` resolves hyperparameters and what
     the defaults are; that `configs/*.json` are inputs you point training at
     (see §6) and **not** a record of anything; and that **the ground truth for
     what a run actually used lives in the run directory**, preserved in the
     published data as two files: `input_config.json` (verbatim copy of the
     `--config` file, reusable as a later `--config` argument) and `config.json`
     (the fully-resolved config). Also cover the `--resume` / `--fork-from`
     modes, which read these rather than `--config`.
  5. **Where the run data lives and how to fetch it** (per §0's destination
     decision).
- `runs/runs_notes.md` was out of date and has been deleted — nothing to promote.
- `references/` has been deleted.

## 6. Repo hygiene

**Untrack the sandbox / harness setup.** `.git` is only 8.6 MB and no secrets
were found in history (`.mcp.json` was never tracked; no token-shaped strings),
so remove from HEAD only — a history purge would break existing clones for no
benefit.

- `project_utils/mcp/{git_push_broker,vast_remote_broker}.py`
- `project_utils/{check_sandbox_consistency,cleanup_worktrees}.py`
- `.claude/skills/parallel-remote-training/` (9 files)
- The four tests that orphan with them: `test_git_push_broker.py`,
  `test_vast_remote_broker.py`, `test_pool_health_liveness.py`,
  `test_queue_audit.py`
- Then drop the `!.claude/skills` negation from `.gitignore` (it exists only to
  keep that skill tracked), and re-check whether `pytest.ini`'s
  `norecursedirs = references .claude` still earns its keep now that
  `references/` is gone (verified: untracked, and no longer on disk).
- Footgun: `.claude/skills/` is write-blocked to Bash/git inside worktrees.
  `git rm --cached` only touches the index so it should be fine, but verify
  rather than assume, and confirm `pytest` is still green afterwards.
- These have a natural home in the already-separate `vast_setup/` repo — worth
  keeping there rather than losing.

**`configs/` (24 files)** — untrack all but **one explicitly-named example**. A
run's real config is `runs/<tag>/config.json`, which §1 keeps per run, so the
per-sweep files under `configs/sweep*/` are a redundant and partial second
source of truth. But `configs/` shouldn't vanish entirely: keeping one file
documents the *format*, shows that this is where you put a config to train
from, and gives a reproducer something to run on day one.

Note this is doubly redundant: `runs/<tag>/input_config.json` is already a
*verbatim* copy of whatever `--config` file the run was launched with, and §1
keeps it. So the published data carries both the input config and the resolved
one for every run — `configs/` adds nothing but a stale partial index.

- Keep exactly one file, and **rename it so its role is unmistakable**:
  `configs/default.json` → `configs/example.json`. "Default" invites the reading
  that it's a baseline the code falls back to; it isn't, and nothing in the code
  references the path (verified: only the `.tmp.py` throwaways mention
  `configs/` at all, so the rename is free). Each config is a complete,
  self-contained input to `--config` rather than a layered override, so one file
  documents the format fully.
- The point of keeping it is that **a reproducer can train something
  immediately**, without first reverse-engineering the schema from a run
  directory:
  ```
  python train_adversarial_logreg.py --config configs/example.json --tag my_first_run ...
  ```
  `CLAUDE.md` should lead the training section with exactly that command.
- `CLAUDE.md` must also state the distinction explicitly: `configs/` is *where
  you put a config to train from*, and is **not** the record of what any run
  used — that's `runs/<tag>/config.json` (resolved) and
  `runs/<tag>/input_config.json` (the file as passed).

**`plot/` and `publish/`** — stay gitignored. Figures are not committed; the
point is that they regenerate.

**`.gitattributes`** — the `*.png filter=lfs` line currently matches zero
tracked files. Since plots aren't being committed, drop it. If §0 lands on Git
LFS for the run data, this file gets rewritten for `*.pt` anyway — and that
must happen **before** the first `.pt` is committed, or it lands as a plain blob
and only a history rewrite fixes it.

**`plans/`** — dropped at the very end, on explicit user command, and possibly
not during this plan at all. Until then it's the staging ground for this work.
Independently of that: **references to `plans/*` from tracked non-plan files
should go now**, since they'll dangle. Current sites:
`config.py:153`, `conftest.py:4,21`, `test_config.py:62`,
`test_train_adversarial_logreg.py:2,7,609`,
`train_adversarial_logreg.py:24,768,850`,
`analytic_feasibility/initial_prompt.md:10`.
Rewrite each to say the thing rather than cite the plan — most are one-line
design notes where the citation is the only fragile part.

**Add `pyproject.toml`.** There is currently no dependency manifest at all,
which blocks reproduction outright. Pin at least torch, numpy, scipy,
matplotlib, jaxtyping.

**LICENSE** — MIT, already added.

## Suggested order

Each step ends with a check-in. Nothing is pushed anywhere.

1. ~~§1 — user reviews the run manifest.~~ **Done:** 525 runs agreed —
   `sweep19_layers2-6` kept, `history.jsonl` kept in full, `baseline_no_c`
   added as group H. No open items.
2. §6 mechanical untracking (sandbox setup, `configs/`, `.gitattributes`,
   orphaned tests). Confirm `pytest` green. Small local commits.
3. §4 script decisions — untrack the agreed set, clean up
   `sweep_group_report.py`, add the c-blind baseline-loss report (user or
   remote box runs the verification).
4. §6 `plans/*` reference rewrites.
5. §2 build and review `runs_publish/`.
6. §0 decide the destination (HF vs LFS vs Release), then wire up tracking.
7. §3 cache-fingerprint fix; verify GPU-free aggregate plotting.
8. §5 `README.md` (user) and `CLAUDE.md` (agent, describing the final tree).
9. `pyproject.toml`.
10. Push — only on explicit user command.
11. `plans/` — dropped only on explicit user command, possibly in a later round.

Keep commits small and self-contained; the untracking steps are mechanical and
the run-data work is not.
