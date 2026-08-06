---
name: parallel-remote-training
description: Run training runs on the vast.ai remote instance. Use when user wants training done on the remote box, especially for a batch of runs or otherwise parallelized training.
---

# Setting up
Due to sandbox restrictions, it's non-obvious the best way to get key files onto the remote.

Recall that the local machine uses worktree isolation, so your work (including helper scripts) is typically somewhere in ./.claude/worktrees, and won't show up in the main repo-level. `mutagen` should sync your local work (in .claude/worktrees/*) to remote instances, so you can write files locally and find them on the remote machine.
If you need a remote file to be somewhere outside of .claude/worktrees/*, you can first write the file locally, then `mv` them using the vastai broker. Alternatively, you can directly create files on the remote using shell commands.

Most of the project directory itself is not writable by the agent user on the remote (only `runs/`), by the way file permissions are set up. Thus, operational files (such as run configs or pool managers) belong outside it, typically in your remote home directory. In particular, config files for runs accept absolute paths, so there isn't an issue with putting config files there.

# Parallel remote training

Throughput on the remote GPU is often CPU-launch-bound, not GPU-bound — running
several training processes concurrently (2-4x) can raise aggregate it/s well
above one process alone. This skill drives `project_utils/vast_pool_manager.sh`,
a small bash pool manager that launches queued commands up to a live-editable
concurrency target, polled from a plain text file every 5s. Concurrency can be
retuned by editing that file without killing or restarting already-running jobs.

## Setup

1. **Queue file**: one training command per line (`python train_adversarial_logreg.py --config ... --tag ...`).
2. **Concurrency file**: a single integer, read every ~5s.
3. Put both — plus logs and the manager's own log — **outside the repo directory**
   on the remote (e.g. `~/sweep_scratch/`), not under the synced project dir.
   Mutagen syncs `*.py` + `configs/` one-way local→remote and overwrites
   remote-side edits, so anything you write into the repo path is liable to be
   clobbered or never persist. Config files themselves are fine to reference
   from the repo (they get copied into `runs/<tag>/` on launch) — it's the
   scratch/queue/log files that need to live elsewhere.
4. Launch detached so it survives disconnects — via `remote_exec`, launch a
   `setsid nohup bash project_utils/vast_pool_manager.sh QUEUE CONC LOGDIR MGRLOG
   PROJECT_DIR VENV_ACTIVATE < /dev/null > pool_stdout.log 2>&1 & disown`,
   then poll with further `remote_exec` calls that tail `MGRLOG` (don't rely on
   `remote_exec`'s own timeout — it can drop the connection without killing the
   remote process, which is what detaching protects against).
5. See `project_utils/vast_pool_manager.sh`'s header comment for exact argu
   order and progress-checking one-liners (`grep -c 'rc=[1-9]' MGRLOG`).

## Ramping concurrency

Bump the concurrency file by **one** at a time. Wait for the rate to
re-stabilize before judging it. Cross-check against `nvidia-smi` GPU
utilization%, which should track aggregate it/s.

- A drop in **both** rate and GPU util after adding a worker is a real
  regression, not noise. Back off by killing just the newest worker:
  - If it already wrote a checkpoint, it just needs `--resume` later.
  - If not, delete its `runs/<tag>` dir first — the script refuses to
    restart an existing tag without `--resume`/`--tag-force`.
- Use `--rate-meter-window` (default 1000.0) with a short value specificall
  while probing concurrency levels for a faster read; revert to whatever's
  standard for the actual production runs once you've settled on a level.
- Tag concurrency-probing runs `debug_*` — excluded from sync/backup, so
  freely disposable.
- GPU utilization% is only one clue out of many, and a 100% utilization (or near 100%) does not necessarily mean the limit has been reached. Power draw similarly represents a partial clue, not a smoking gun.

## Gotchas

- **Don't trust idle-looking spare capacity.** A CPU-only process run
  alongside a GPU sweep, even with plenty of apparent CPU headroom by load
  average, silently stole real GPU throughput in practice. A/B any such
  overlap in isolation before trusting it.
- **Source edits must happen locally**, then `sync_flush` before launching
  the remote checkout gets overwritten by mutagen, so editing on-box is a dead
  end (see `vast_setup/CLAUDE.md`).
