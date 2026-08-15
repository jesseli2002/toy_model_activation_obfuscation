# Repo closeout cleanup plan

Plan only — nothing here is implemented yet. Scope is git-tracked content only;
gitignored files (`runs/`, `plot/`, `publish/`, `vast_setup/`) are out of scope
except where a decision is to *start* tracking some of them.

(Caveat: `plans/` is itself under review in item 5 below. If `plans/` is dropped,
this file moves or goes with it.)

## 0. Key finding, drives everything else

The writeup figures split into two classes with very different data needs:

- **Per-run publish plots** (`make_publish_plots.py`,
  `make_publish_plot_train_dist_curve.py`) do probe refits / PCA / steering, so
  they genuinely need the checkpoint `.pt`. Only **3-4 run tags** are involved.
- **Aggregate sweep plots** (`by_layer`, `by_width`, `loss_vs_auroc*`,
  `pareto_grid`, `lam_sweep_task0.01`) consume only scalars through
  `sweep_lib/metrics.py`'s `MetricStore`, and those scalars are already
  persisted in `plot/metrics_cache.json` — **200 KB, 795 entries**.

But the cache cannot currently be used without the checkpoints, for two
independent reasons (`sweep_lib/metrics.py:136`, `sweep_lib/cache.py:33`):

1. Every accessor (`task_loss`, `auroc`, `linear_y_r2`, ...) early-returns
   `None` on `not self.has_checkpoint(tag)`, before consulting the cache.
2. The cache key embeds `file_fingerprint` = `"{st_size}:{st_mtime_ns}"`. Git
   does not preserve mtime, so **even a committed checkpoint would miss every
   cache entry after a fresh clone** — the fingerprint scheme is built around
   `rsync -a`, not git.

This is the fork in the road for the LFS work:

| | Payload | Reproducer experience |
|---|---|---|
| **A. Commit slim checkpoints** for all 7 aggregate sweeps | ~370 MB `.pt` + ~220 MB `history.jsonl` | Plots regenerate, but every metric recomputes from scratch (GPU-hours) because fingerprints miss |
| **B. Make the cache usable standalone** + commit it, and commit `.pt` only for the 3-4 per-run plots | **~2 MB** | Aggregate plots regenerate in seconds; per-run plots regenerate from real checkpoints |

**Recommend B.** Two small changes unlock it: switch `file_fingerprint` to a
content hash (or fall back to one when a `.pt` is absent), and let the
accessors serve a cache hit without `has_checkpoint`. That is a genuine
improvement to the code, not a hack for publishing — the current scheme
silently invalidates on any file copy that doesn't preserve mtime.

Verify before committing to B: that the 795 cache entries actually cover every
tag/metric the four aggregate scripts request (run them against a
checkpoint-less tree and count skips).

## 1. Runs to keep, via Git LFS

Under option B the required set is small:

- `sweep7_lam0.1_tr0` — most of the `sweep7_lam0.1_tr0_*` writeup figures
- `sweep3_lam0_tr0` — the two `L2_steer_dir_mag*` comparison figures
- `sweep11_lr0.0015_iter200k_lam0.01_tr0` — `id_vs_ood_result.png`
- `sweep7_lam0_tr0` — present in `publish/`; confirm whether the writeup uses it

Per run, keep `checkpoints/{best,last}.pt`, `logs/{history.jsonl,report.md}`,
`config.json`, `input_config.json`. Drop the `iter_*.pt` series (that is what
makes a run 19 MB instead of ~1 MB).

Sequencing and footguns:

1. **Add `*.pt` (and `*.jsonl` if kept) to `.gitattributes` BEFORE the first run
   is committed.** Otherwise they land as plain blobs and only a history rewrite
   fixes it.
2. Smoke-test with one run: commit, push, confirm `git lfs ls-files` shows a
   pointer. A prior LFS lock-verify failure on push was transient — retry, do
   not reconfigure.
3. **`last.pt` is a symlink to `iter_N.pt`.** Committed as-is while excluding
   `iter_*.pt`, a clone gets a dangling link. Materialize it as a real file.
4. `runs/` is in `.gitignore`; add negations for exactly the kept paths.
5. `runs/CLAUDE.md` opens with "Not git-tracked" — false once this lands.
6. Confirm the account's LFS storage/bandwidth allowance. If option A is chosen
   after all, prefer a GitHub Release attachment or a data-only repo; Release
   assets don't count against the LFS quota.

## 2. Untrack the sandbox / harness setup

Not just `project_utils/` — the same class of thing is tracked in two places.
`.git` is only 8.6 MB and no secrets were found in history (`.mcp.json` was
never tracked; no token-shaped strings), so **remove from HEAD only**; a history
purge would break existing clones for no benefit.

Remove:
- `project_utils/mcp/{git_push_broker,vast_remote_broker}.py`
- `project_utils/{check_sandbox_consistency,cleanup_worktrees}.py`
- `.claude/skills/parallel-remote-training/` (9 files — vast.ai pool manager,
  queue scripts; same "don't force my setup on reproducers" argument)

And the four tests that orphan with them:
- `test_git_push_broker.py`, `test_vast_remote_broker.py` (→ `project_utils/mcp`)
- `test_pool_health_liveness.py`, `test_queue_audit.py` (→ the skill's scripts)

Then:
- Delete the `!.claude/skills` negation from `.gitignore` (it exists only to
  keep that skill tracked, and the user wants sandbox setup out of the shared
  gitignore).
- Re-check whether `pytest.ini`'s `norecursedirs = references .claude` still
  earns its keep.

Suggestion, not deletion: `project_utils/mcp/*` has a natural home in the
already-separate `vast_setup/` repo — likely worth keeping for yourself.

## 3. Old scripts

Built from an actual import graph over `git ls-files '*.py'`, rooted at the two
writeup copy scripts. (Note: docstring cross-references are not import edges,
and entry points have zero inbound refs by definition — neither is evidence of
being dead.)

**Keep — produces a writeup figure or is imported by one that does:**
`make_publish_plots.py`, `make_publish_plot_train_dist_curve.py`,
`sweep7_analysis.py`, `sweep_width_analysis.py`, `sweep_layer_analysis.py`,
`adversarial_report.py`, `train_adversarial_logreg.py`, `sweep_lib/*`,
`probe_lib.py`, `probe_backend.py`, `probe_newton.py`, `torch_logreg.py`,
`checkpoint_lib.py`, `model.py`, `data.py`, `config.py`, `paths.py`,
`stableadamw.py`, `rate_meter.py`.

**Delete — nothing imports them and they produce no writeup figure:**
- `probe_ideal_y.py` (101 lines, zero refs anywhere)
- `check_dead_relu.py` (88 lines, one-off diagnostic, Jul 22)
- `benchmark_logreg_gpu.py` (219 lines, backend bake-off, decision long since made)

**Decide — orphaned analyses for superseded sweeps:**
- `sweep8_analysis.py`, `sweep13_analysis.py` — imported by nothing; sweeps 8
  and 13 don't appear in the writeup. Likely delete.
- `sweep_threshold_report.py`, `sweep_group_report.py`,
  `sweep_inspect_training.py` — no inbound imports, but they're interactive
  investigation tools rather than dead code. Keep if the repo is meant to show
  method, delete if it's meant to show only results.
- `train_no_c.py` + `analytic.py` — feed the `baseline_no_c` control run and the
  published analytic-solution writeup. Keep `analytic.py`; `train_no_c.py` is
  only useful with the baseline run, so it goes or stays with that run.
- `analytic_feasibility/` (8 files) — exploratory sympy work behind the analytic
  writeup, which is publicly linked. Keep, but it needs its README checked.

## 4. README — highest-leverage item for "presentable"

Currently a WIP stub pointing at three Google Docs, at least two of which are
probably private, i.e. useless to an outside reader. Replace with: what the
experiment is, the headline result, how to regenerate each writeup figure
(exact commands), what's in `runs/`, and a link to the public analytic writeup
only. Keep the project-log links only if they're actually shareable.

Also: `runs/runs_notes.md` is a genuinely valuable lab notebook that is
currently untracked — worth promoting to a tracked `docs/` file.

## 5. Remaining judgement calls

- **`plans/` (17 archived plan docs + 2 `CLAUDE.md`s):** keep as project
  history, or drop? If dropping, `grep -rn 'plans/' $(git ls-files)` first —
  `conftest.py` docstrings cite `plans/rare_flags_config_plan.md` and would
  dangle. `plans/CLAUDE.md` and `plans/archive/CLAUDE.md` are agent-workflow
  instructions and should go regardless of the above.
- **`configs/` (24 files across sweeps 9-18):** keep only the ones matching runs
  that survive item 1, or keep all as an experiment record?
- **`.gitattributes`:** the existing `*.png filter=lfs` line currently matches
  zero files (`plot/` and `publish/` are gitignored). Either drop it or start
  committing the writeup PNGs deliberately.
- **`CLAUDE.local.md` is untracked and stays that way** — but confirm no tracked
  file references sandbox-only workflow that won't exist for a reproducer.
- Consider adding a LICENSE and a `requirements.txt` / `pyproject.toml`; there
  is currently no dependency manifest, which blocks reproduction outright.

## Suggested order

1. Decide A vs B (item 0) — everything about `runs/` depends on it.
2. Untrack sandbox setup + orphaned tests (item 2); confirm `pytest` still green.
3. Delete dead scripts (item 3, the unambiguous three first).
4. `.gitattributes` + LFS smoke test, then commit the kept runs (item 1).
5. README rewrite (item 4) last, so it describes the final tree.

Keep these as separate small PRs — the deletions are mechanical and the LFS work
is not.
