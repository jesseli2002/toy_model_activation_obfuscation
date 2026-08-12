---
name: parallel-remote-training
description: Run training runs on the vast.ai remote instance. Use when user wants training done on the remote box, especially for a batch of runs or otherwise parallelized training.
---

# Environment
When a remote is set up for the first time, a `mutagen` session is started to sync local source files to the remote, and a `rsync` daemon is set up to pull `./runs` data back from the remote. Abstracting away the implementation, the end effect is that source files (`*.py`, `configs/**`, `.claude/worktrees/**`, `.claude/skills/**`) should be present on the remote. Changes made locally to those files will be reflected on the remote, but with some delay; use the `sync_flush` MCP endpoint to ensure changes show up. Similarly, `runs/` data gets brought back automatically, with the exception of tags matching `debug_*` (used for smoke tests).

WARNING: nothing at the transport layer prevents two remotes from running the same tag, and the damage is silent. The train script's same-tag guard is per-box (it only sees that box's `runs/`, seeded once at creation), and every box rsyncs into one local `runs/` with last-writer-wins per file — so two boxes on one tag interleave their checkpoints with no error anywhere. See "Keeping tags unique across instances" below for the funnel that prevents this and the audit that detects it.

For implementation details, see the vast_setup/ directory on the local machine.

## Getting files onto the remote
Due to sandbox restrictions, it's non-obvious the best way to get key files onto the remote.

Recall that the local machine uses worktree isolation, so your work (including helper scripts) is typically somewhere in ./.claude/worktrees, and won't show up in the main repo-level. However, the syncing daemons should copy those files to the remotes, so you can write files locally and find them on the remote machine.
If you need a remote file to be somewhere outside of .claude/worktrees/*, you can first write the file locally, then `mv` them using the vastai broker. Alternatively, you can directly create files on the remote using shell commands.

Most of the project directory itself is not writable by the agent user on the remote (only `runs/`), by the way file permissions are set up. Thus, operational files (such as run configs or pool managers) belong outside it, typically in your remote home directory. In particular, config files for runs accept absolute paths, so there isn't an issue with putting config files there.

# Parallel remote training

Throughput on the remote GPU is often CPU-launch-bound, not GPU-bound — running
several training processes concurrently can raise aggregate it/s well
above one process alone. This skill drives
`.claude/skills/parallel-remote-training/scripts/vast_pool_manager.sh`,
a small bash pool manager that launches queued commands up to a live-editable
concurrency target, polled from a plain text file every 5s. Concurrency can be
retuned by editing that file without killing or restarting already-running jobs.

## Start running on the remote

Bringing one instance from "box exists" to "running at tuned throughput".
Track it as a to-do list — one entry per numbered step, per instance.

1. **Generate the run commands.** Build/check the pending pool with the
   sweep's manifest script (e.g. `generate_sweep8.py build`/`status` —
   sweep-specific), or with the generic `scripts/sweep_pool.py build` if
   there is no such script yet. Then carve out this instance's share:
   `sweep_pool.py assign --instance NAME --n-runs N`, sized by **run
   count**, not param-count weight (training is overhead-bound, not
   FLOP-bound — see project memory — so weight is a poor wall-clock
   proxy). Assign a modest chunk while compute is still coming online,
   the whole remaining pool once it isn't.
2. **Place the operational files** on the remote **outside the repo
   directory** (e.g. `~/sweep_scratch/`), not under the synced project dir:
   - *queue file*: one training command per line
     (`python train_adversarial_logreg.py --config ... --tag ...`). Push
     the `.delta.txt` that `assign` wrote with `queue_append.sh` over
     `remote_exec` — never a bare `cat >>`, see that script's header.
   - *concurrency file*: a single integer, re-read every ~5s; start at 1.
     A stopfile at `${CONC}.stop` (e.g. `conc.txt.stop`) tells the manager
     to finish in-flight jobs and exit, and is the only way to shut one
     down — it does not exit on its own (see step 3).
   - the job logs and the manager's own log.
3. **Launch the pool manager**, detached so it survives disconnects — via
   `remote_exec`, `setsid nohup bash vast_pool_manager.sh QUEUE CONC
   LOGDIR MGRLOG PROJECT_DIR VENV_ACTIVATE < /dev/null > pool_stdout.log
   2>&1 & disown`. Poll with further `remote_exec` calls tailing `MGRLOG`
   (don't rely on `remote_exec`'s own timeout — it can drop the connection
   without killing the remote process, which is what detaching protects
   against). Its header comment has the exact argument order, the locking
   contract (`QUEUE.lock`, `QUEUE.launched_idx`) and progress one-liners
   (`grep -c 'rc=[1-9]' MGRLOG`). **It idle-polls instead of exiting when
   the queue drains** — `STOPPED`/`ALL DONE` in `MGRLOG` appears only on an
   intentional stopfile shutdown, so anything else means it is still
   running, possibly idling. A dead manager will not notice appended work,
   so confirm it is alive before concluding "topped off and busy".
4. **Find the concurrency level** by ramping up from 1 and measuring, per
   "## Ramping concurrency" below, until throughput stops improving.
5. **Report an ETA** to the user, estimated from **iteration throughput**:
   read the running jobs' it/s and compare against each run's iteration
   count. No script does this — sweeps are heterogeneous, so a
   job-completion rate averaged over a queue says nothing reliable about
   the runs still in it; judge it case by case. Before any run has
   produced iterations, label the estimate as a guess and revisit.

## Keeping tags unique across instances

`assignments.json` (under `sweep_pool.py --out-dir`) is the single registry
of which instance owns a tag. Everything that puts work on a queue —
`assign`, `requeue`, `reassign` — records it there, and `assign` pops from
one shared pending pool under a lock, so **duplicates are impossible as long
as every queue mutation goes through one of those commands**. Retries are
the path that historically bypassed it: use `sweep_pool.py requeue`
(`--resume-tag` for a checkpoint resume, which it will only allow on the
instance holding that checkpoint; `--retry-tag` for a fresh `<tag>_retryN`)
rather than hand-writing a line into `queue_append.sh`.

To verify the invariant against what is *actually* on the remotes:

1. `queue_audit.py fetch-request --instances instances.json` prints the
   `fetches` argument for the broker's `fetch_files` tool (queue,
   `launched_idx` and `manager.log` for every registered instance).
2. Pass it to `fetch_files`, which lands them under
   `/tmp/vast-remote-broker-fetch/<instance>/<remote path>`. That
   destination is fixed by the server, not caller-chosen.
3. `queue_audit.py check --instances instances.json --out-dir POOL_DIR`
   exits nonzero on any ERROR: the same tag queued on two instances, a
   non-resume repeat on one instance, two occurrences of one tag both still
   awaiting dispatch (the manager fills free slots without looking at tags,
   so those can start together — repeated `--resume` retries are fine, but
   only one may be pending at a time), a tag with no registry row, or a tag
   queued somewhere other than its registered owner. Duplicates are
   classified by `launched_idx` — still in the undispatched tail means
   `queue_trim.sh` can fix it, already dispatched means it is clobbering now
   and needs a human. A queue copy that is missing or stale is reported
   rather than treated as clean.

For periodic unattended health-checking of a sweep already underway
(stall/failure detection, auto-retry of known-flaky failures, rebalancing,
and instance-down handling — all monitor-executed, not just recommended),
see `.claude/skills/parallel-remote-training/monitor_brief.md`.

## Scheduling the check-in subagent

Once the pool is fully assigned (every instance's queue holds the whole
remaining sweep, not just an initial chunk — see above), set up a
recurring **~2h** check via the `schedule` skill, running as a fresh
subagent each time (no standing context growth, no stale/uncached-token
cost). The routine's stored prompt is `monitor_brief.md`'s only input —
it must carry the sweep's `instances.json` path explicitly, and may
carry extra sweep-specific detail if you judge it relevant (see
`monitor_brief.md`'s "Input" section). As of this writing, run it as a
**local** scheduled subagent, not a cloud routine (cloud support here is
unevaluated).

## Ramping concurrency

Bump the concurrency file by **one GPU's worth of jobs at a time** — i.e.
by `N`, the instance's GPU count (1 on a single-GPU box;
more on a multi-GPU box, so each step adds one job per GPU rather than
skewing the count across them). After each bump, wait **30s to warm up**,
then measure over a **30s window** (iter-count delta at the window's
start and end) — this is plenty to read true instantaneous throughput
and stays well under the ~5min prompt-cache TTL, so your own context
doesn't go cold mid-probe. Cross-check against `nvidia-smi` GPU
utilization% (per-GPU on a multi-GPU box), which should track aggregate
it/s.

- A drop in **both** rate and GPU util after adding workers is a real
  regression, not noise. Back off by killing just the newest `N` workers
  (the ones added in that last bump):
  - If a worker already wrote a checkpoint, it just needs `--resume` later.
  - If not, delete its `runs/<tag>` dir first — the script refuses to
    restart an existing tag without `--resume`/`--tag-force`.
- Use `--rate-meter-window` with a short value specifically
  while probing concurrency levels for a faster read; revert to whatever's
  standard for the actual production runs once you've settled on a level.
- GPU utilization% is only one clue out of many, and a 100% utilization (or near 100%) does not necessarily mean the limit has been reached. Power draw similarly represents a partial clue, not a smoking gun.

## Adding additional runs or instances
- If the user asks to add additional runs or spins up another instance, **use a forked subagent to generate the manifest, update the remote queues, evaluate concurrency, etc.** Let a subagent handle the details of everything, so that your (long-running) context remains clean.

## Gotchas

- **Don't trust idle-looking spare capacity.** A CPU-only process run
  alongside a GPU sweep, even with plenty of apparent CPU headroom by load
  average, silently stole real GPU throughput in practice. A/B any such
  overlap in isolation before trusting it.
- **Source edits must happen locally**, then `sync_flush` before launching.
  The remote checkout gets overwritten by mutagen, so editing on-box is a dead
  end (see `vast_setup/CLAUDE.md`).
- **Any tool that mutates `queue.txt` outside the manager itself**
  (`queue_trim.sh`, `queue_append.sh`, or a future one) must hold
  `QUEUE.lock` across its read-decide-write sequence, same as the manager
  does — see `vast_pool_manager.sh`'s header comment. Skipping this can
  desync `launched_idx.txt` from `queue.txt`'s actual contents, which
  corrupts every downstream decision (trim, top-off sizing, audit).
- **Trim/reassign is proven safe against an actively-launching manager**
  (validated live: lock contention against a real 5s poll loop, correct
  undispatched-tail isolation, `launched_idx` untouched) — a monitor
  agent may execute rebalancing autonomously, not just recommend it (see
  `monitor_brief.md`).
- **A relative config path in a queued command resolves against the
  remote's default cwd (the synced main checkout), not the worktree** —
  so a config that only exists in a worktree silently fails to be found
  there. Pin both the config path and the training script path to their
  absolute location under the synced worktree in remote commands.

## Script index

All under `.claude/skills/parallel-remote-training/scripts/`. Each carries
its own header comment or docstring with the full contract — argument
order, locking, failure modes — so treat this as a map, not a reference.

**Run on the remote** (via `remote_exec`, next to the live queue):

- `vast_pool_manager.sh` — the dispatch loop itself. Launched once per
  instance at setup, detached, and left running for the whole sweep.
- `queue_append.sh` — the only safe way to add lines to a running
  instance's queue. Every delta produced locally gets pushed with this.
- `queue_trim.sh` — removes undispatched lines off a queue's tail, for
  rebalancing between instances or clearing a duplicate.

**Run locally** (they read local files and cannot reach a remote; use the
broker's `fetch_files` to bring remote files to them):

- `sweep_pool.py` — the pool's bookkeeping and the registry that keeps
  tags unique: `build`/`status`/`assign` at setup, `requeue` for retries,
  `reassign` for moves. Everything that puts work on a queue goes through
  here. Deliberately estimates no ETAs (see step 5 above).
- `pool_health.py` — one instance's stall/failure summary from its
  `manager.log`, plus a job-completion rate used only for top-off sizing
  (not for ETAs — it averages over heterogeneous runs). Pass `--queue`/
  `--sync-host` as well for a real three-state stall verdict: `manager.log`
  alone can't distinguish a dead box from one long quiet run, so liveness
  comes from the running jobs' `history.jsonl` mtimes in the local `runs/`
  tree instead.
- `queue_audit.py` — verifies the remote queues against the registry
  (`fetch-request` to build the fetch, `check` to audit). Used per
  monitor wake and before declaring a sweep done.
- `sweep_status.py` — read-only running/queued/complete/failed breakdown
  with a rough ETA for in-flight jobs, meant for a person to run directly
  at a terminal (no agent, no token cost) via `--instances instances.json`.
  Unlike the others above, its default `--source ssh` fetches over plain
  `ssh ALIAS cat PATH` itself rather than needing `fetch_files` first
  (`--source local` still works from inside an agent session).
