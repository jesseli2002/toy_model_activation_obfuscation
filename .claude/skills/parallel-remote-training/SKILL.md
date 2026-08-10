---
name: parallel-remote-training
description: Run training runs on the vast.ai remote instance. Use when user wants training done on the remote box, especially for a batch of runs or otherwise parallelized training.
---

# Environment
When a remote is set up for the first time, a `mutagen` session is started to sync local source files to the remote, and a `rsync` daemon is set up to pull `./runs` data back from the remote. Abstracting away the implementation, the end effect is that source files (`*.py`, `configs/**`, `.claude/worktrees/**`, `.claude/skills/**`) should be present on the remote. Changes made locally to those files will be reflected on the remote, but with some delay; use the `sync_flush` MCP endpoint to ensure changes show up. Similarly, `runs/` data gets brought back automatically, with the exception of tags matching `debug_*` (used for smoke tests).

WARNING: nothing at the transport layer prevents two remotes from running the same tag, and the damage is silent. The train script's same-tag guard is per-box (it only sees that box's `runs/`, seeded once at creation), and every box rsyncs into one local `runs/` with last-writer-wins per file — so two boxes on one tag interleave their checkpoints with no error anywhere. See "Keeping tags unique across instances" below for the funnel that prevents this and the audit that detects it.

For implementation details, see the vast_setup/ directory on the local machine.

## Setting up
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

## Setup

1. **Queue file**: one training command per line (`python train_adversarial_logreg.py --config ... --tag ...`).
2. **Concurrency file**: a single integer, read every ~5s. A stopfile at
   `${CONC}.stop` (e.g. `conc.txt.stop`) tells the manager to finish
   in-flight jobs and exit cleanly — it does NOT exit on its own when the
   queue merely drains (see next point), so this is the only way to
   actually shut one down.
3. Put both — plus logs and the manager's own log — **outside the repo directory**
   on the remote (e.g. `~/sweep_scratch/`), not under the synced project dir.
4. Launch detached so it survives disconnects — via `remote_exec`, launch a
   `setsid nohup bash vast_pool_manager.sh QUEUE CONC LOGDIR MGRLOG
   PROJECT_DIR VENV_ACTIVATE < /dev/null > pool_stdout.log 2>&1 & disown`,
   then poll with further `remote_exec` calls that tail `MGRLOG` (don't rely on
   `remote_exec`'s own timeout — it can drop the connection without killing the
   remote process, which is what detaching protects against).
5. See `vast_pool_manager.sh`'s header comment for exact argument order,
   the locking contract (`QUEUE.lock`, `QUEUE.launched_idx`), and
   progress-checking one-liners (`grep -c 'rc=[1-9]' MGRLOG`).
   **The manager idle-polls instead of exiting when the queue drains** — a
   dead manager only appending doesn't restart it, so before assuming
   "topped off and busy," confirm the process is actually still alive
   (e.g. check `MGRLOG` for a recent `STOPPED`/`ALL DONE` line, which now
   only appears on an intentional stopfile shutdown — anything else means
   it's still running, possibly idling).

## Sizing an initial assignment and reporting an ETA

For "spin up N instances, don't dump the whole pool on them yet, tell me
how long the full sweep will take" (the common ask when compute is coming
online incrementally):

1. Build/check the pending pool with your sweep's manifest script (e.g.
   `generate_sweep8.py build`/`status` — sweep-specific) or directly via
   the generic `.claude/skills/parallel-remote-training/scripts/sweep_pool.py`
   if the sweep script already delegates to it.
2. Assign a modest chunk sized by **run count**, not param-count weight —
   `sweep_pool.py assign --instance NAME --n-runs N`. Weight (params
   ratio) is a poor proxy for wall-clock cost here (training is
   overhead-bound, not FLOP-bound — see project memory), so size by a
   target time horizon and the instance's *measured* rate once you have
   one, not by weight.
3. Push the resulting `.delta.txt` onto the remote queue via
   `queue_append.sh` over `remote_exec` (never a bare `cat >>` — see that
   script's header for why).
4. To report an ETA: land each active instance's `manager.log` on the
   local disk with the broker's `fetch_files` tool — `sweep_pool.py eta`
   reads *local* files and can't reach a remote itself. Note a redirect
   inside a `remote_exec` command string (`cat manager.log > copy.log`)
   writes on the **remote**, so it does not do this. Then run
   `sweep_pool.py eta --instance-log NAME:local_copy.log[:launched_idx]`
   once per instance. ETA is `pending_runs / aggregate_jobs_per_hour`
   (throughput-based — NOT `avg_duration × pending_runs`, which ignores
   concurrency and overestimates by roughly the concurrency factor).
5. With no throughput data yet (brand new instance), report an
   assign-based estimate honestly labeled as a guess, and revisit once
   `sweep_pool.py eta` has real data.

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

Bump the concurrency file by **one** at a time. Wait for the rate to
re-stabilize before judging it, in poll chunks of **≤3 min** each (never
one long sleep) — this both keeps checks responsive and stays under the
~5min prompt-cache TTL, so your own context doesn't go cold mid-probe.
Cross-check against `nvidia-smi` GPU utilization%, which should track
aggregate it/s.

- A drop in **both** rate and GPU util after adding a worker is a real
  regression, not noise. Back off by killing just the newest worker:
  - If it already wrote a checkpoint, it just needs `--resume` later.
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
  corrupts every downstream decision (trim, ETA, top-off sizing).
- **Trim/reassign is proven safe against an actively-launching manager**
  (validated live: lock contention against a real 5s poll loop, correct
  undispatched-tail isolation, `launched_idx` untouched) — a monitor
  agent may execute rebalancing autonomously, not just recommend it (see
  `monitor_brief.md`).

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
  `reassign` for moves, `eta` for reporting. Everything that puts work on
  a queue goes through here.
- `pool_health.py` — one instance's throughput/stall/failure summary from
  its `manager.log`. Importable, and what `sweep_pool.py eta` uses
  underneath; call it directly for a per-instance read.
- `queue_audit.py` — verifies the remote queues against the registry
  (`fetch-request` to build the fetch, `check` to audit). Used per
  monitor wake and before declaring a sweep done.
- `sweep_status.py` — read-only running/queued/complete/failed breakdown
  with a rough ETA for in-flight jobs, meant for a person to run directly
  at a terminal (no agent, no token cost) via `--instances instances.json`.
  Unlike the others above, its default `--source ssh` fetches over plain
  `ssh ALIAS cat PATH` itself rather than needing `fetch_files` first
  (`--source local` still works from inside an agent session).
