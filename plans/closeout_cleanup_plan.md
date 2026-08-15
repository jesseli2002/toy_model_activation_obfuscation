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
| The 494 candidate runs, full directories | 8.9 GB |
| The same 494, keeping only `{best,last}.pt` + configs + logs | **732 MB** |
| …of which `checkpoints/{best,last}.pt` | 597 MB |
| …of which `logs/history.jsonl` | 134 MB |
| `plot/metrics_cache.json` | 200 KB |

Two consequences:

- **Dropping the `iter_*.pt` series is a 12× trim** (8.9 GB → 732 MB) and is
  where essentially all the savings come from. Nothing else is worth optimising
  by comparison.
- **Local disk is not a constraint.** 203 GB free on the volume. `runs_publish/`
  can be a plain dereferencing copy (`cp -L`) — no hardlink/symlink games
  needed. Being selective about *which runs* get copied still matters for
  reviewability and for whatever we eventually upload; being selective to save
  local disk does not.

### Where the 732 MB should live

732 MB is too much for a plain git repo and awkward for Git LFS (quota, and
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

**Status: pending user review.** The full list — 494 runs, grouped, with
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
| G | `sweep19_layers{2-4,2-8,2-10,4-8}_lam{…}_tr{0..9}` — layer pairs, Pareto view | 120 | 90 MB |

`make_publish_plots.py` (`sweep7_lam0.1_tr0`) and
`make_publish_plot_train_dist_curve.py` (`sweep11_lr0.0015_iter200k_lam0.01_tr0`)
are already covered by B and D.

Findings from building the list — all need a user decision or at least a nod:

- **`sweep7_lam0_*` no longer exists in `runs/`.** `sweep7_analysis.py` already
  reads its λ=0 arm from `LAM0_GLOB = "sweep3_lam0_tr*"`, so nothing is broken.
  But `publish/sweep7_lam0_tr0/` still holds rendered figures under that name,
  i.e. a published figure is attributed to a tag with no run behind it. The
  reproducer docs must name that baseline `sweep3_lam0_tr0`.
- **`sweep19_layers2-6` (30 runs) is in the keep set only if `sweep_group_report.py`
  keeps its "2,6" example.** It appears in no `PARETO_COMPARISONS` entry, so
  `sweep_layer_analysis.py` never touches it. See the §4 question.
- **Cache coverage cross-check:** of 795 `plot/metrics_cache.json` entries,
  360 distinct tags. 330 are in the keep set; **the only 30 outside it are
  exactly the `sweep19_layers2-6` runs** — a clean confirmation that the
  enumeration has no missing consumer. Conversely 164 keep-set runs have no
  cache entry at all: all of group A and all of group B, because
  `sweep7_analysis.py` uses `CKPT="best"` and a different `MetricSpec`. Those
  metrics recompute on first plot.

### What to keep per run

`checkpoints/best.pt`, `checkpoints/last.pt`, `config.json`,
`input_config.json`, `logs/history.jsonl`, `logs/report.md`. Drop the
`iter_*.pt` series.

`config.json` is deliberately included: it is the ground truth of what a run
actually ran with, which is why `configs/` can be untracked (§6).

`logs/history.jsonl` is 134 MB of the 732 MB and is **a separate decision**.
Keeping it is what makes `sweep_group_report.py`'s curves view (and the
train-loss trajectory figures) reproducible, since `load_history` reads it
directly. Dropping it halves the upload but breaks a script §4 keeps. Default:
keep, but flag it if the destination has a size limit.

## 2. Curate `runs_publish/`

A staging directory, built and iterated on **outside `runs/`** so the live run
tree is never mutated. Gitignored while it's being worked out.

1. `mkdir runs_publish`
2. For each kept tag, copy the six paths above, **dereferencing symlinks**.
   `checkpoints/last.pt` is a symlink to an `iter_N.pt` in 483 of the 494 runs;
   copied as a link with `iter_*.pt` excluded, it dangles. `cp -L` (or
   `rsync -L`) fixes this — verify afterwards that `runs_publish` contains zero
   symlinks.
   - The 11 exceptions are `sweep3_lam0_tr{0..10}`, whose `last.pt` is already
     a real file.
   - `best.pt` is a real file everywhere; `best` and `last` never resolve to the
     same file, so both are genuinely two checkpoints.
3. Verify: every run has both `.pt` files non-empty, `config.json` parses, and
   the total matches §0's 732 MB.
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
- `train_no_c.py` — the `baseline_no_c` control run.
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
- Keep 2–3 examples, exactly one of them uncommented. Currently ~14 commented
  blocks survive from live use. **Question for the user below.**
- The docstring should say what the script is *for* (ad-hoc cross-sweep
  comparison, edit `GROUPS` to point it at whatever you're comparing) rather
  than describing whichever comparison happens to be active.

## 5. Docs

- **`README.md`** — human-written by the user, not generated. A project summary
  plus links to the full writeup. Only the publicly readable links belong here;
  the current stub's Google Doc links are private and useless to an outside
  reader. The [analytic solution writeup](https://jesseli2002.github.io/blog/blog/activation_obfuscation_analytic/)
  is public and stays.
- **`CLAUDE.md`** (tracked, new) — the operational half: what each script is,
  how to run it, how to regenerate each writeup figure with exact commands,
  where the run data lives and how to fetch it, and what `probe_ideal_y.py` /
  `sweep_threshold_report.py` / `sweep_group_report.py` are for. This is where
  the "guided" part of the goal actually gets delivered.
  - Distinct from `CLAUDE.local.md`, which stays untracked and sandbox-specific.
    Cross-check that no tracked file describes a workflow that only exists in
    this sandbox.
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
  `references/` is gone.
- Footgun: `.claude/skills/` is write-blocked to Bash/git inside worktrees.
  `git rm --cached` only touches the index so it should be fine, but verify
  rather than assume, and confirm `pytest` is still green afterwards.
- These have a natural home in the already-separate `vast_setup/` repo — worth
  keeping there rather than losing.

**`configs/` (24 files)** — untrack all of it. A run's real config is
`runs/<tag>/config.json`, which §1 keeps per run, so `configs/` is a redundant
and partial second source of truth.

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

1. §1 — user reviews the run manifest; resolve `sweep19_layers2-6` and the
   `history.jsonl` question.
2. §6 mechanical untracking (sandbox setup, `configs/`, `.gitattributes`,
   orphaned tests). Confirm `pytest` green. Small local commits.
3. §4 script decisions — untrack the agreed set, clean up
   `sweep_group_report.py`.
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
