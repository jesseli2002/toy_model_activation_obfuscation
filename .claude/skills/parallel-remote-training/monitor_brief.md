# Monitor brief: periodic parallel-remote-training health check

You are a **fresh agent invocation with no memory of previous checks** —
this doc is meant to be everything you need. Don't assume anything about
prior runs beyond what's in the files below; all real state is durable
(remote log/queue files, `instances.json`, `handoff.md`), not in any chat
history.

## What you're checking

One or more vast.ai instances, each running a `vast_pool_manager.sh`
dispatch loop against its own `queue.txt`/`conc.txt`/`manager.log` (see
`.claude/skills/parallel-remote-training/SKILL.md`), all pulling from a
single sweep's pending pool (`sweep_pool.py`/a sweep-specific manifest
script's `build`/`status`/`assign`).

## Input

Whatever scheduled this brief must supply the path to that sweep's
`instances.json` — this doc has no sweep baked in. The setup agent that
schedules the recurring check is also free to append sweep-specific
detail to the scheduled prompt (known quirks, a non-default safety
margin, etc.) if it judges that appropriate; treat any such addition as
an amendment to this brief, not a substitute for it.

If the path is missing from your invocation, or the file doesn't exist:
don't guess a path and don't wait for a human to answer — log it to
`handoff.md` (see below, creating the file next to wherever you'd expect
`instances.json`) and stop. A fresh agent has no prior context to fall
back on and typically nobody is watching live.

1. Read `instances.json` for the registered instances: `{alias:
   {scratch_dir, queue, conc, manager_log, launched_idx, project_dir,
   venv_activate}}`.
2. Cross-check against `mcp__vast-remote-broker__list_instances` — an
   instance running there but absent from `instances.json` is a real
   instance not being monitored; flag it, don't silently monitor it via
   guessed paths.
3. Read `handoff.md` (same directory as `instances.json`) before doing
   anything else — the previous wake's notes are your only memory.

## Each wake, per registered instance

1. **Fetch health data.** `remote_exec cat MANAGER_LOG` and `cat
   LAUNCHED_IDX`, save locally, then run
   `sweep_pool.py`'s underlying `pool_health.py` (or `sweep_pool.py eta`
   for the aggregate) against the local copies — these tools read local
   files only, they can't reach the remote themselves.
2. **Failing?** A nonzero-`rc` line in `MANAGER_LOG` — see "Auto-retry on
   failure" below before escalating; it's not automatically a human
   issue anymore.
3. **Stalled?** (`stalled: true`, or a rising failure count that doesn't
   match the auto-retry path) — don't self-diagnose deeply; surface to
   `handoff.md` and the user with the relevant `manager.log` tail. A dead
   GPU or a systematic failure (bad config, OOM, etc.) needs a human.
4. **Manager process actually alive?** A `STOPPED`/`ALL DONE` line in
   `manager.log` (or, for an older unfixed deployment, silence past when
   the queue should have drained) means the dispatcher itself has
   exited — appending more work would do nothing. Flag this distinctly
   from "stalled but alive"; it needs a restart, not a top-off.
5. **Running low?** Compare `high_water_mark` to the instance's queue
   length and its `jobs_per_hour_trailing` against your target safety
   margin (e.g. keep ≥2h of queued work at the current rate). If low:
   compute `n_runs = target_hours × jobs_per_hour_trailing`, run
   `sweep_pool.py assign --instance ALIAS --n-runs N`, then push the
   resulting delta via `queue_append.sh` over `remote_exec`.
   - If `jobs_per_hour_trailing` is `null` (no recent data — e.g. a
     freshly-started instance), use a conservative fixed-count top-off
     instead of a rate-derived one, and note in `handoff.md` that the
     next cycle should have real data.
   - Note: if the sweep was fully assigned upfront (the normal case once
     initial ramping is done), this should rarely trigger — treat it as
     a signal something's off (an instance burning through work far
     faster than expected) rather than routine.
6. **Over-provisioned relative to another instance's rate, or one
   instance is clearly faster/slower than assumed?** **Recommend**
   a `queue_trim.sh` + reassign to the user rather than doing it
   yourself — this path has only been tested against a paused manager
   locally, never against one actively launching jobs under real
   concurrent load. State the specific trim/reassign command you'd run
   and why, and let a human execute it the first few times.
7. **Any live concurrency experiment you choose to run** (bump
   `conc.txt`, observe, keep/revert) must poll in chunks of **≤3
   minutes** per wait step — never one long sleep — both to keep your
   own context from going stale mid-wake and to stay under the ~5min
   prompt-cache TTL. Log any change you make to `handoff.md` as passive
   context for the next wake, since you won't remember doing it.

## Auto-retry on failure

Unattended auto-retry is permitted here (unlike trim/rebalance) — this
is specifically what lets a sweep run through the night unattended.

1. For each `rc=[1-9]` line in `MANAGER_LOG` since the last known-good
   wake, tail the corresponding `job<N>.log`.
2. **Known-flaky signature** (bare `Segmentation fault` / a signal-style
   kill with no Python traceback — an observed hardware/remote-specific
   issue, not a code bug) → eligible for retry. Anything else (a
   traceback, OOM, assertion) → NOT eligible; escalate to `handoff.md`
   and the user instead.
3. Before retrying an eligible failure, validate `runs/<tag>/checkpoints/
   last.pt`: load it and check the model/optimizer tensors for NaN/Inf
   (and suspicious all-zero). Do this rather than trusting the file just
   because `save_checkpoint`'s atomic-write means it can't be
   half-written — the failure mode here is corrupted *content*, not a
   torn write.
   - Checkpoint looks clean → requeue `--resume --tag <tag>`, appended
     to that instance's queue via `queue_append.sh` (under the lock).
   - Checkpoint corrupted, or the check is inconclusive → don't resume.
     Requeue a from-scratch run under `<tag>_retry1` (increment if that
     tag's already taken) instead, appended at the end of the queue —
     not the front, so it doesn't delay other pending work.
4. Log every retry decision (tag, reason, resume vs. from-scratch) to
   `handoff.md`.

## Handoff notes

`handoff.md`, next to `instances.json` (already read per "Input" above),
is the only continuity between wakes. Append an entry before you exit —
even a one-liner ("checked, all healthy, nothing done"): timestamp, what
you saw, what you did, anything a human should know before the next
wake.

## Done condition

Sweep is complete when: the pending pool is empty (`sweep_pool.py
status`/your sweep's manifest-script `status` shows 0 pending) AND every
registered instance's queue has drained AND no `stalled`/nonzero
`failure_count` remains unexplained. This is a normal, expected outcome
on any given wake — no action needed beyond noting it in `handoff.md`.
Report it plainly rather than continuing to poll — whatever scheduled
this brief should stop firing once you say so.

## instances.json schema (if you need to create/update it)

```json
{
  "vtao": {
    "scratch_dir": "/home/agent/sweep_scratch",
    "queue": "/home/agent/sweep_scratch/queue.txt",
    "conc": "/home/agent/sweep_scratch/conc.txt",
    "manager_log": "/home/agent/sweep_scratch/manager.log",
    "launched_idx": "/home/agent/sweep_scratch/queue.txt.launched_idx",
    "project_dir": "/workspace/toy_probe_hiding",
    "venv_activate": "/workspace/.venv/bin/activate"
  }
}
```

Verify against the live instance rather than assuming this is current —
instances get rented and destroyed between sweeps. `launched_idx`/
`queue.lock` paths only exist once the fixed `vast_pool_manager.sh` (with
the drain/locking fix) is the one actually deployed there — an older
deployment won't have them; note that distinctly rather than treating a
missing sidecar as an error.
