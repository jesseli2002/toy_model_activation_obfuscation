---
name: parallel-remote-training
description: Run training runs on the vast.ai remote instance. Use when user wants training done on the remote box, especially for a batch of runs or otherwise parallelized training.
---

# Environment
When a remote is set up for the first time, a `mutagen` session is started to sync local source files to the remote, and a `rsync` daemon is set up to pull `./runs` data back from the remote. Abstracting away the implementation, the end effect is that source files (`*.py`, `configs/**`, `.claude/worktrees/**`, `.claude/skills/**`) should be present on the remote. Changes made locally to those files will be reflected on the remote, but with some delay; use the `sync_flush` MCP endpoint to ensure changes show up. Similarly, `runs/` data gets brought back automatically, with the exception of tags matching `debug_*` (used for smoke tests).

WARNING: there is NO control to prevent two runs with the same tag on different remotes from clobbering each other. It is up to you to ensure each tag is assigned to a unique remote.

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
4. To report an ETA: fetch each active instance's `manager.log` locally
   (`remote_exec cat manager.log > local_copy.log` — `sweep_pool.py eta`
   reads *local* files, it can't reach the remote itself), then run
   `sweep_pool.py eta --instance-log NAME:local_copy.log[:launched_idx]`
   once per instance. ETA is `pending_runs / aggregate_jobs_per_hour`
   (throughput-based — NOT `avg_duration × pending_runs`, which ignores
   concurrency and overestimates by roughly the concurrency factor).
5. With no throughput data yet (brand new instance), report an
   assign-based estimate honestly labeled as a guess, and revisit once
   `sweep_pool.py eta` has real data.

For periodic unattended health-checking of a sweep already underway
(stall/failure detection, auto-retry of known-flaky failures, and
*recommending* — not autonomously performing — rebalancing), see
`.claude/skills/parallel-remote-training/monitor_brief.md`.

## Scheduling the check-in agent

Once the pool is fully assigned (every instance's queue holds the whole
remaining sweep, not just an initial chunk — see above), set up a
recurring **~2h** check via the `schedule` skill, running as a fresh
agent each time (no standing context growth, no stale/uncached-token
cost). The routine's stored prompt is `monitor_brief.md`'s only input —
it must carry the sweep's `instances.json` path explicitly, and may
carry extra sweep-specific detail if you judge it relevant (see
`monitor_brief.md`'s "Input" section). As of this writing, run it as a
**local** scheduled agent, not a cloud routine (cloud support here is
unevaluated).

## Ramping concurrency

Bump the concurrency file by **one** at a time. Wait for the rate to
re-stabilize before judging it. Cross-check against `nvidia-smi` GPU
utilization%, which should track aggregate it/s.

- A drop in **both** rate and GPU util after adding a worker is a real
  regression, not noise. Back off by killing just the newest worker:
  - If it already wrote a checkpoint, it just needs `--resume` later.
  - If not, delete its `runs/<tag>` dir first — the script refuses to
    restart an existing tag without `--resume`/`--tag-force`.
- Use `--rate-meter-window` with a short value specifically
  while probing concurrency levels for a faster read; revert to whatever's
  standard for the actual production runs once you've settled on a level.
- GPU utilization% is only one clue out of many, and a 100% utilization (or near 100%) does not necessarily mean the limit has been reached. Power draw similarly represents a partial clue, not a smoking gun.

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
- **Trim/reassign is not yet proven safe to run unattended** against an
  actively-launching manager (only tested against a paused one locally,
  never with real concurrent load) — a monitor agent should *recommend*
  a trim/rebalance to the user rather than execute it autonomously, until
  someone's watched it happen live at least once.
