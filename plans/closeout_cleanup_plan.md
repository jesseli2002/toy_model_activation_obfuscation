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
- **Local disk is not a constraint.** 203 GB free on the volume. `runs/`
  can be a plain dereferencing copy (`cp -L`) — no hardlink/symlink games
  needed. Being selective about *which runs* get copied still matters for
  reviewability and for whatever we eventually upload; being selective to save
  local disk does not.

### Where the 764 MB should live

**Status: decided and done.** Hugging Face Hub, per the recommendation below.
Uploaded to the private model repo `cooleytukey/toy_model_of_activation_obfuscation`
(user created it and issued a repo-scoped fine-grained token), under a `runs/`
path in the repo matching the local layout. Verified server-side: 2623 files
across all 525 run tags, matching the local `runs/` build exactly. The repo
was public by default when created; flipped to private before uploading.

**Sandbox networking gotcha, for whoever runs the next upload/download:** this
environment's outbound network is allowlisted per-host. `huggingface.co` is
enough for auth and small-file (non-LFS) commits, but large files go through
HF's **Xet** storage backend, not classic LFS — that needs
`cas-server.xethub.hf.co` allowlisted too, or `hf upload`/`hf download` hangs
at `0.00B/0.00B` indefinitely rather than failing fast (the sandbox denies the
connection silently; the client has no way to distinguish that from a slow
network). If large-file transfers stall with no byte progress after the
hashing step, this is the first thing to check.

764 MB is too much for a plain git repo and awkward for Git LFS (quota, and
every `git clone` pays for it even for someone who only wants to read the code).
Hugging Face Hub is built for exactly this, has no per-clone tax, versions the
data, and lets the repo stay small with a download snippet in `CLAUDE.md`.

Alternatives, in rough order of preference if HF is rejected: a GitHub Release
attachment (doesn't count against LFS quota, but unversioned and clumsy to
update); Git LFS in this repo (simplest to wire up, worst clone experience); a
separate data-only git repo.

**This decision gates any `git add` of a `.pt` file.** Because we're committing
locally and not pushing, a mistaken LFS commit is a history rewrite to undo.
Curate `runs/` first (§2), decide the destination second, wire up
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

`make_one_run_plots.py` (`sweep7_lam0.1_tr0`, `sweep3_lam0_tr0`) and
`plot_train_dist_curve.py` (`sweep11_lr0.0015_iter200k_lam0.01_tr0`,
renamed from `make_publish_plot_train_dist_curve.py`)
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
  `logs/{history.jsonl,README.md}`, but no `best.pt`, `config.json`, or
  `input_config.json`, because `train_no_c.py` writes a reduced layout. (It
  happens to also have a `logs/report.md`, dropped per the correction below
  like every other run's.) Copy what exists; don't synthesise the missing
  files. `logs/README.md` records the exact command that produced it and
  should be preserved as-is.

### What to keep per run

`checkpoints/best.pt`, `checkpoints/last.pt`, `config.json`,
`input_config.json`, `logs/history.jsonl`. Drop the `iter_*.pt` series.

**Correction from the §2 build:** `logs/report.md` — originally in this list
— is dropped entirely, not just "kept if present". It's not a training
artifact; it's a saved dump from a manual `adversarial_report.py` invocation,
so it only exists where someone happened to run and save one (122 of 1423
runs in `runs_all/`; 39 of the 525-run keep set). Since it's a regenerable
report rather than raw data, and its presence is arbitrary rather than
meaningful, it doesn't belong in the published per-run file set.

`config.json` and `input_config.json` are deliberately included: between them
they are the ground truth of what a run actually ran with, which is why
`configs/` can shrink to a single example (§6).

`logs/history.jsonl` is 144 MB of the 764 MB. **Decision: keep it, in full.**
It's what makes `sweep_group_report.py`'s curves view and the train-loss
trajectory figures reproducible (`load_history` reads it directly), and
downsampling it would mean shipping something that isn't the raw log. Revisit
only if the chosen destination imposes a size limit.

## 2. Curate `runs/`

**Status: done.** Built from `runs_all/` into `runs/` in the main checkout
(script: `build_runs.tmp.py`, left in place as a throwaway artifact — safe to
delete). Verified: 525 run dirs, group counts match the manifest exactly
(A:15 B:149 C:20 D:20 E:20 F:150 G:150 H:1), 0 symlinks, no zero-byte `.pt`
files, all `config.json` parse, 741 MB total (vs. the 764 MB estimate — the
gap is `logs/report.md`, see the correction above). User reviewed and
confirmed ("runs looks good"). Step 3 below (load a checkpoint per group) is
still open — needs the remote GPU box or the user, not an agent locally.

**Update (supersedes the `runs_publish/` staging name below):** the user moved
the live run tree from `runs/` to `runs_all/` on disk, freeing `runs/` up to be
built *directly* as the curated set — no separately-named staging directory
needed, since `runs_all/` now plays that "don't mutate the live tree" role and
`runs/` already matches `.gitignore`'s `runs` pattern (inert to git until
deliberately tracked). This also sets up §0's reproducibility test for free:
once the curated `runs/` has been pushed to HF Hub, wiping and re-downloading
`runs/` from the Hub confirms a fresh clone can actually reproduce from the
published data. Source is `runs_all/` throughout; target is `runs/`.

1. For each kept tag, copy the six paths above, **dereferencing symlinks**.
   Group H (`baseline_no_c`) has a reduced layout — no `best.pt`, no
   `config.json`/`input_config.json`, plus an extra `logs/README.md` recording
   its exact training command. Copy what exists rather than failing on the
   missing files, and keep that README.
   `checkpoints/last.pt` is a symlink to an `iter_N.pt` in 513 of the 525 runs;
   copied as a link with `iter_*.pt` excluded, it dangles. `cp -L` (or
   `rsync -L`) fixes this — verify afterwards that `runs/` contains zero
   symlinks.
   - The 11 exceptions are `sweep3_lam0_tr{0..10}`, whose `last.pt` is already
     a real file.
   - `best.pt` is a real file everywhere; `best` and `last` never resolve to the
     same file, so both are genuinely two checkpoints.
2. Verify: every run has both `.pt` files non-empty, `config.json` parses, and
   the total matches §0's 764 MB.
3. Load one checkpoint per group and confirm it opens (**on the remote GPU box
   or by the user — not locally**; see the CPU/GPU rule in `CLAUDE.local.md`).

Only once this is reviewed does anything get committed or uploaded.

## 3. The metrics cache — resolved, no code change

**Status: resolved.** This section originally proposed a code fix (content-hash
the cache fingerprint instead of mtime) framed as a GPU-accessibility problem.
Both premises turned out wrong on inspection, corrected here rather than left
to mislead a later reader:

- **Not a GPU problem.** `has_checkpoint(tag)` (`sweep_lib/metrics.py:136`)
  checks only whether the run's `.pt` file exists on disk; `MetricStore`
  takes `device` as a plain string and every caller already does `"cuda" if
  torch.cuda.is_available() else "cpu"`. Since §1/§2 ship the actual
  checkpoints as part of the published `runs/`, a reproducer who downloads
  the curated set has the `.pt` files, so `has_checkpoint()` is `True` and
  the aggregate scripts already run end-to-end on CPU — nothing to fix.
- **The mtime fingerprint (`sweep_lib/cache.py:33`) stays as-is.** It's a
  deliberate design (built around `rsync -a`), not a bug; recompute is a
  one-time cost the user is fine paying on first setup rather than
  content-hashing checkpoints to make a copied cache portable.

**Consequence: `metrics_cache.json` is not published.** A cache built from
mtime fingerprints would miss nearly every entry after an HF download (which
doesn't preserve mtime) — shipping it would just be 200KB of dead weight, not
a head start. It stays local-only, in `plot/` (gitignored, as it already is).
`CLAUDE.md` (§5) should say the cache rebuilds automatically and silently on
first use; no separate step needed.

## 4. Scripts

Built from an import graph over `git ls-files '*.py'`, rooted at the writeup
figure scripts. (Docstring cross-references aren't import edges, and entry
points have zero inbound refs by definition — neither is evidence of dead code.)

**Keep — produces a writeup figure or is imported by one that does:**
`make_one_run_plots.py`, `plot_train_dist_curve.py`,
`sweep7_analysis.py`, `sweep_width_analysis.py`, `sweep_layer_analysis.py`,
`train_adversarial_logreg.py`, `sweep_lib/*`,
`probe_lib.py`, `probe_backend.py`, `probe_newton.py`, `torch_logreg.py`,
`checkpoint_lib.py`, `model.py`, `data.py`, `config.py`, `paths.py`,
`stableadamw.py`, `rate_meter.py`.

**Keep — decided:**
- `probe_ideal_y.py` — demonstrates the task is actually solvable, which is
  exactly the kind of background work a reproducer needs. **Since merged into
  `analytic.py`** (as `sample_ideal_y`/`verify_ideal_y_decodability`) rather
  than kept as a standalone file — no separate entry point survives, so drop
  this bullet's file name where §5 documents it and point at `analytic.py`
  instead.
- `sweep_threshold_report.py` — how the loss/AUROC thresholds were chosen. This
  is the "background work" the goal statement calls for.
- `analytic.py` — backs the publicly linked analytic-solution writeup.
- `train_no_c.py` — trains the `baseline_no_c` c-blind control, which
  empirically confirms the analytic c-blind loss floor is reachable. That floor
  is the certificate the whole experiment leans on ("a model scoring below it
  must be reading c"), so it belongs in the reproducible set.

  No new tooling is needed to read its result: `logs/history.jsonl`'s last
  line carries the achieved `l_task` (5.563e-2 at iter 49999) to read off
  directly (the earlier draft of this note also pointed at `adversarial_report.py
  --tag baseline_no_c`; that script is retired below, so the history.jsonl
  route is now the only one). `CLAUDE.md` just needs to point at it, and at
  `analytic.no_c_task_loss` for the floor being compared against.
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
- `adversarial_report.py` — **newly decided (this session).** See
  "`adversarial_report.py` retirement" below for the full rationale and the
  fallout it requires elsewhere in this plan.

### `adversarial_report.py` retirement

**Status: executed.** Commits d636cd7 (probe_lib.py symbol moves),
51618cc (`paths.plot_dir` ckpt arg), 042911e (`make_one_run_plots.py`
porting + `plot/` move), efd8e59 (`plot_train_dist_curve.py` `plot/`
move), e668a3e (untrack), 2899347 (dangling-reference cleanup), ffca64c
(`.gitignore`). `pytest`: 171 passed. Both open items below were
resolved by the user rather than left to the executing agent: **ckpt
nesting stays `plot/<tag>/<ckpt>/`** (unchanged from the old
`publish/<tag>/<ckpt>/` shape, just the root renamed — `paths.plot_dir`
grew the optional `ckpt` arg as anticipated); **`copy_plots.sh` in the
blog checkout is explicitly out of scope** — the user will handle that
cross-repo update separately, so it still points at `publish/` and will
need fixing before the writeup's figure-refresh step works again.

`adversarial_report.py` is the
per-run diagnostic report (arbitrary `--tag`/`--ckpt`, `--detailed`, `--steer`,
training-trace plots) that `make_one_run_plots.py` and `plot_train_dist_curve.py`
were both built to specialize away from — see their docstrings. Since neither
kept script covers the training-trace / probe-AUROC-over-training plot (and it
was assessed as unreliable in practice — see below), and `sweep_group_report.py`
already covers cross-run loss-curve comparison, nothing left depends on
`adversarial_report.py` staying a standalone entry point. It moves to
**Untrack**, alongside `sweep8_analysis.py` etc. above.

This is a bigger untrack than the others in this section — it's imported by
two scripts this plan keeps, and it currently determines where two kept
scripts write their output. Concretely:

1. **Two symbols need a new home before the untrack lands, or `make_one_run_plots.py`
   and `sweep_threshold_report.py` break on import:**
   - `_steer_vectors` — imported by `make_one_run_plots.py`. Move into
     `probe_lib.py` (where the rest of the shared plotting/analysis helpers
     already live) rather than inlining it into `make_one_run_plots.py`, since
     nothing about it is writeup-specific.
   - `plot_learned_curves` — imported by `sweep_threshold_report.py`. Same
     move, into `probe_lib.py`. (`analytic_demo.py` has a same-named function,
     but it's a different signature over the analytic — not adversarially
     trained — model with no noise argument; a naming coincidence, not a
     duplicate to merge.)
2. **Two plots need porting into `make_one_run_plots.py`**, per the earlier
   report-tool assessment in this session — and they're not equal-effort:
   - **Layer distributions** (`_plot_layer_distributions`, → `{tag}_c{lo}-{hi}_layer_dist.png`)
     is nearly free: `make_one_run_plots.py`'s `_run_analysis` already computes
     `gap_plot_inputs` for every hidden layer, which is all this plot consumes.
   - **Held-out gap** (`_plot_heldout_gap`, → `{tag}_heldout_gap.png`,
     currently only drawn under `adversarial_report.py --detailed`) needs the
     *computation* ported too, not just the plot — `make_one_run_plots.py`'s
     `_run_analysis` has no held-out-pairs analysis at all today. Since
     `make_one_run_plots.py` has no CLI surface for this kind of per-run
     parameter (it dropped `--tag` for the same reason — see
     `7e5e51a`), `held_out_pairs` becomes a module constant there, matching
     `STEER_LAYERS`/`NOISE_GRID_EVAL_MULTS`'s existing convention, not a new flag.
3. **Training traces are dropped, not ported**, on the assessment from this
   session: the probe-AUROC-over-training half of `_plot_training_traces` is
   unreliable in practice, and the loss-curve half is already covered for the
   multi-run comparison use case by `sweep_inspect_training.py` (superimposed,
   smoothed `L_task` vs. iter from `history.jsonl`, same as `_plot_training_traces`
   minus the reference-loss overlay lines). **This makes `sweep_inspect_training.py`'s
   keep/untrack call in `closeout_followup_review.md` load-bearing** — it must stay
   tracked, or this coverage claim silently stops being true for a later reviewer.
   `_plot_probe_gap` (`{tag}_probe_gap.png`) is dropped too, for the DoM/logreg
   series specifically because `make_one_run_plots.py`'s existing `_plot_auroc_line`
   (`{tag}_auroc_bar.png`) already covers the same DoM-vs-logreg-per-layer
   comparison, just as AUROC instead of fixed-threshold accuracy — accepted as
   the better metric. The one thing `_plot_auroc_line` doesn't carry over is the
   LDA series (`_plot_probe_gap` has DoM/logreg/LDA; `_plot_auroc_line` has no LDA
   AUROC computed at all) — dropped, not ported; flag if that turns out to matter
   for a specific writeup figure later.
4. **`publish/` → `plot/`, and `make_one_run_plots.py` takes over `plot/<tag>`
   directly.** `make_one_run_plots.py` and `plot_train_dist_curve.py` currently
   write to `publish/<tag>/<ckpt>/` (`PUBLISH_DIR` module constant in each) —
   a separate namespace from `paths.plot_dir(tag)` (`plot/<tag>/`), which today
   only `adversarial_report.py` writes to. That separation existed to avoid the
   two tools' outputs colliding in the same directory; once `adversarial_report.py`
   is retired, `plot/<tag>/` has exactly one writer per tag again, so both
   scripts move onto `paths.plot_dir(tag)` and `publish/` goes away — bringing
   them inline with how `sweep7_analysis.py` etc. already write straight to
   `plot/sweep7/` with no intermediate namespace.
   - **Open question, not yet decided:** both scripts support `--ckpt both`
     (writing best- and last-checkpoint figures in the same invocation) and
     every filename is `{tag}_*.png` with no checkpoint discriminator, so a
     flat `plot/<tag>/` would silently let one checkpoint's figures overwrite
     the other's. Whether to keep a `plot/<tag>/<ckpt>/` subdirectory (in which
     case `paths.plot_dir` is the natural place to grow an optional `ckpt`
     argument, since retiring `adversarial_report.py` leaves it with no other
     caller) or fold `ckpt` into the filename instead needs a call before this
     step is implemented.
   - The **cross-repo fallout**: the blog's `copy_plots.sh` scripts
     (`/work/blog/content/blog/hiding_experiment/copy_plots.sh`,
     `.../hiding_sweep/copy_plots.sh`) hardcode `~/ml/toy_probe_hiding/publish/<tag>/<ckpt>/...`
     source paths for every figure they copy. These must be updated in lockstep
     with the rename, in the blog checkout, or the writeup's own figure-refresh
     step silently breaks. `.gitignore`'s `publish` line comes out once nothing
     writes there anymore.
5. **Mechanical note, since revised:** `.git/info/exclude` was expected to be
   blocked from this worktree (per step 3 of the Suggested Order below, which
   had to defer it to merge-back) — that turned out not to apply here; the
   write to `.git/info/exclude` for `adversarial_report.py` succeeded directly
   from this worktree. `git commit -m ... -- <pathspec>` still reliably fails
   in this sandbox; a plain `git commit` with nothing else staged remains the
   workaround.

**`analytic_feasibility/` — resolved, then fully untracked.** Assessed
against what a reproducer needs per the user's two goals (an MWE linkable
from the blog, and the code behind the blog's probe figures):
- **Kept, then moved out:** `simplified_demo.py` (byte-identical to the
  `demo.py` already linked from the analytic blog post) moved to
  `analytic_demo.py` at the repo root; `probe_v_channels.py` (generates the
  three `v_channels_{2d,hist,roc}.png` figures embedded in that post) merged
  into `analytic.py` as `sample_v_channels()`/`verify_v_channels()`. With
  both gone, `analytic_feasibility/` no longer has any tracked file — the
  directory itself is untracked (nothing left to `git rm --cached`).
  `analytic.py` now also runs every construction's verification as a
  pass/fail report and saves all its plots (including the merged
  `v_channels_*` figures) under `plot/analytic/`; `analytic_demo.py`'s
  `--out-dir` default moved from `.` to `plot/analytic` to match.
- **Untracked** (kept on disk, per the usual `git rm --cached` +
  `.git/info/exclude` convention): `README.md`, `initial_prompt.md`,
  `period2_decode.py`, `period2_net.py`, `search_exact.py`,
  `search_results.json`, `verify_feasibility.py`. None feed a blog figure or
  are imported by the two keepers. `README.md`'s untrack was an explicit
  user call (it's the only record of the feasibility investigation's
  no-go-theorem narrative, but out of scope for both stated goals).
  `initial_prompt.md`'s dangling reference to `plans/high_level_plan.md`
  (the §6 concern) is now moot since the file is untracked, not deleted.

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
     Entry points (`train_adversarial_logreg.py`,
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
       `analytic.no_c_task_loss` certifies, and how to check `baseline_no_c`
       against it (the final `l_task` in its `history.jsonl`).
     - *Part 2 — single-run empirical result*: `make_one_run_plots.py`, which
       runs against both writeup tags in one invocation — `sweep7_lam0.1_tr0`
       (the AUROC / curves / PCA / probe / steering / ROC-grid / layer-distribution
       / held-out-gap figures) and `sweep3_lam0_tr0` (the two
       `L2_steer_dir_mag*` λ=0 comparisons). Output moves to `plot/<tag>/`
       (see "`adversarial_report.py` retirement" in §4) once that lands.
     - *Part 3 — hyperparameter sweeps*: `sweep7_analysis.py` (λ sweep →
       `plot/sweep7/`), `plot_train_dist_curve.py`
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
     comparison, `analytic.py`'s `sample_ideal_y`/`verify_ideal_y_decodability`
     for why the task is solvable at all.
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

**Note for the next agent:** steps were done out of the order below, at the
user's direction — §2 and §0 (run data + HF upload) went first, before the
mechanical untracking in steps 2–4. Those are still fully open; nothing in §4
or §6 has been touched yet.

1. ~~§1 — user reviews the run manifest.~~ **Done:** 525 runs agreed —
   `sweep19_layers2-6` kept, `history.jsonl` kept in full, `baseline_no_c`
   added as group H. No open items.
2. ~~§6 mechanical untracking (sandbox setup, `configs/`, `.gitattributes`,
   orphaned tests).~~ **Done** in three small commits (587585b, 7b9205b,
   30d5b3e): sandbox/harness files + the parallel-remote-training skill +
   4 orphaned tests; `configs/` down to a renamed `example.json`;
   `.gitattributes`'s dead `*.png filter=lfs` line. All index-only removal,
   files left on disk, recorded in `.git/info/exclude`.
   - The `!.claude/skills` negation was already dropped in `969a851`
     (before this session) — nothing left to do there.
   - `pytest.ini`'s `norecursedirs = references .claude` re-check: `references/`
     is untracked and empty on disk (not literally absent, as the §6 note
     assumed, but inert) — leaving the entry is harmless either way, not
     changed.
   - `pytest` after: 303 passed, 1 pre-existing failure
     (`test_sweep_lib.py::test_empty_auroc_thresholds`, predates this
     session per `git log`, unrelated to anything touched here) — flagging
     per `CLAUDE.local.md`'s "don't ignore warnings/failures" rule rather
     than silently treating it as green.
3. ~~§4 script decisions.~~ **Done**, in four commits (a0f6f2f, 5cf5245,
   1edeb95, 860919b — hashes on branch `worktree-golden-herding-nest`, not
   yet merged to `master`): untracked `sweep8_analysis.py`,
   `sweep13_analysis.py` (no confirmation needed), reframed
   `sweep_group_report.py` per spec, and — after checking in with the user —
   untracked `check_dead_relu.py`/`benchmark_logreg_gpu.py` and kept
   `sweep_inspect_training.py` tracked, parked in the new
   `plans/closeout_followup_review.md` for a later ambiguous-call review.
   `pytest`: 171 passed (this worktree's tracked-file count; no failures).
   - ~~**Found but not fixed:** `sweep_layer_analysis.py` and
     `sweep_width_analysis.py` (both kept) have docstring lines
     ("see sweep13_analysis.py for why") that will dangle for a fresh-clone
     reproducer now that `sweep13_analysis.py` is untracked.~~ **Fixed** in a
     follow-up commit: both docstrings now state the reason inline (some
     runs' final checkpoint predates their last logged history entry) instead
     of citing the untracked file.
   - **`.git/info/exclude` entries deferred**, across all four §4 commits —
     this worktree's isolation guard blocks writes to the shared common-dir
     file. Needed at merge-back: `sweep8_analysis.py`, `sweep13_analysis.py`,
     `check_dead_relu.py`, `benchmark_logreg_gpu.py` (same deferral already
     applies to §6's untracked files from step 2, done in the non-worktree
     checkout where the exclude file *was* writable — only step 3's files
     are affected).
   - **Sandbox quirk hit twice:** `git commit -m ... -- <pathspec>` reliably
     fails here ("no changes added to commit") even when exactly those paths
     are staged; a plain `git commit` with nothing else staged works. Worth
     remembering for future commits in this worktree.
4. ~~§6 `plans/*` reference rewrites.~~ **Done** (65d88d4): all ten cited
   sites rewritten to say the thing rather than cite the plan; verified with
   `git grep plans/` across `*.py`/`*.md` that no tracked non-plan file
   references `plans/` anymore.
5. ~~§2 build and review `runs/`.~~ **Done** — see §2 status above.
6. ~~§0 decide the destination, then wire up tracking.~~ **Done** — Hugging
   Face Hub, uploaded to the private repo
   `cooleytukey/toy_model_of_activation_obfuscation`. See §0 status above,
   including the Xet networking gotcha if the next agent needs to
   upload/download again.
   - **Open follow-up:** the reproducibility test this setup enables —
     wipe local `runs/` and re-download from the HF repo — hasn't been run
     yet. Worth doing once §2 step 3 (checkpoint load-check) also happens.
   - ~~Also open: move `plot/metrics_cache.json` to `runs/metrics_cache.json`
     and include it in the HF repo.~~ **Reversed** — §3 (below) now says not
     to publish it at all; an mtime-keyed cache would miss almost everything
     after an HF download anyway.
7. ~~§3.~~ **Resolved, no code change needed** — see §3 above. Folded into
   §5's docs (note that the cache rebuilds silently on first use).
8. ~~§5 `README.md` (user) and `CLAUDE.md` (agent, describing the final tree).~~
   **Done.** `README.md` is a short, user-written stub pointing at the
   writeup and `CLAUDE.md`. `CLAUDE.md` covers source structure (entry point
   + one-command training invocation), the writeup's three parts in order
   with the exact commands per `copy_plots.sh`, the background-work scripts,
   the config/run-config distinction, and the HF Hub download command —
   kept intentionally short per the user (agents read it in full; humans are
   pointed there too).
9. `pyproject.toml`. **Not started.**
10. Push — only on explicit user command.
11. `plans/` — dropped only on explicit user command, possibly in a later round.

Keep commits small and self-contained; the untracking steps are mechanical and
the run-data work is not.
