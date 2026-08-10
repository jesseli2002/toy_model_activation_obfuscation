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

1. **Fetch health data.** Get the `fetches` argument from `queue_audit.py
   fetch-request --instances instances.json`, pass it to the broker's
   `fetch_files` tool, and run `pool_health.py` (or `sweep_pool.py eta`
   for the aggregate) against the local copies it writes — these tools
   read local files only, they can't reach the remote themselves. Do this
   once for all instances at the start of the wake; it also gives you the
   queues that step 8 audits. A `remote_exec` redirect writes on the
   remote, not here, so it is not a substitute.
2. **Failing?** A nonzero-`rc` line in `MANAGER_LOG` — see "Auto-retry on
   failure" below before escalating; it's not automatically a human
   issue anymore.
3. **Stalled?** (`stalled: true`, or a rising failure count that doesn't
   match the auto-retry path) — don't self-diagnose deeply; surface to
   `handoff.md` and the user with the relevant `manager.log` tail. A
   systematic failure (bad config, OOM, etc.) needs a human. If the
   instance is actually unreachable rather than just stalled, see
   "Instance unreachable" below instead — that one you can act on
   yourself.
   - **Known false positive**: `pool_health.py`'s `stalled` is derived
     purely from time-since-last-`manager.log`-line, but long single-shot
     runs (hours, no `--resume` restarts) never write intermediate lines
     between their own launch and finish. A multi-hour gap with no
     launch/finish events is expected here, not evidence of a dead
     manager. Before escalating, cheaply verify instead: `ps aux` on the
     instance (the job PIDs still alive, high CPU) and tail one running
     `job<N>.log` (iter count still advancing). Only escalate if those
     *also* look wrong.
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
   instance is clearly faster/slower than assumed?** Rebalance it
   yourself: `queue_trim.sh` the over-provisioned instance, `sweep_pool.py
   reassign` + `queue_append.sh` onto the under-provisioned one.
   `queue_trim.sh`'s mechanics are validated safe under real concurrent
   load (lock contention against an actively-launching manager) — no
   need to route this through a human anymore. Log what you moved and
   why to `handoff.md`.
7. **Audit tag uniqueness.** `queue_audit.py check --instances
   instances.json --out-dir POOL_DIR` against the copies fetched in step
   1. Nonzero exit means at least one ERROR. Treat a
   `duplicate-across-instances` whose occurrences are all still
   undispatched as yours to fix (`queue_trim.sh` the copy on the instance
   that doesn't own the tag per `assignments.json`, then re-audit); one
   with an already-dispatched occurrence is two runs clobbering a single
   `runs/<tag>` right now — stop, don't trim, surface it to the user and
   `handoff.md` immediately. `unregistered`/`misplaced` findings mean
   something reached a queue outside `sweep_pool.py`; log them and re-run
   the audit after correcting the registry.
8. **Any live concurrency experiment you choose to run** (bump
   `conc.txt`, observe, keep/revert) must poll in chunks of **≤3
   minutes** per wait step — never one long sleep — both to keep your
   own context from going stale mid-wake and to stay under the ~5min
   prompt-cache TTL. Log any change you make to `handoff.md` as passive
   context for the next wake, since you won't remember doing it.

## Auto-retry on failure

Unattended auto-retry is permitted here — this is specifically what lets
a sweep run through the night unattended.

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
   - Checkpoint looks clean → `sweep_pool.py requeue --instance ALIAS
     --resume-tag <tag>`.
   - Checkpoint corrupted, or the check is inconclusive → don't resume.
     `sweep_pool.py requeue --instance ALIAS --retry-tag <tag>` instead,
     which allocates the next free `<tag>_retryN` for you.

   Either way, push the delta it writes with `queue_append.sh` (under the
   lock), appended at the end of the queue — not the front, so it doesn't
   delay other pending work. Go through `requeue` rather than composing
   the line yourself: it is what puts the retry in `assignments.json`, and
   a retry that never lands there is invisible to the guarantee that no
   tag reaches two instances (see SKILL.md, "Keeping tags unique").
4. Log every retry decision (tag, reason, resume vs. from-scratch) to
   `handoff.md`.

## Instance unreachable

Distinct from "stalled but alive" above — this is `remote_exec` failing
outright, or `list_instances` showing it non-`running`. Unattended action
is permitted (reuses only already-validated primitives, no new
mechanism):

1. Don't try to salvage anything from the dead instance — you can't
   trim/checkpoint-validate what you can't reach.
2. Everything past its last-known `launched_idx` (from your most recent
   successful poll of it) was never dispatched — move it straight to
   another live instance's queue via `sweep_pool.py reassign` (pointing
   `--lines-file` at that undispatched tail, taken from your last fetched
   queue copy) plus `queue_append.sh`. Nothing to trim remotely; the
   source is gone, not just over-provisioned. Going through `reassign`
   rather than appending directly is what keeps `assignments.json`
   pointing at the instance that now owns each tag.
3. Everything at-or-before that `launched_idx` without a "finished" line
   in your last-known `MANAGER_LOG` has unknown fate. Don't attempt
   `--resume` elsewhere — cross-instance checkpoint transfer isn't
   supported by the current one-way sync (local↔each-remote, not
   remote↔remote), and `requeue --resume-tag` will refuse it. Treat it
   like an inconclusive checkpoint check: `sweep_pool.py requeue
   --instance OTHER --retry-tag <tag>`, appended at that queue's end.
4. Mark the instance distinctly dead in `handoff.md` (not just
   "stalled") so future wakes stop spending a `remote_exec` attempt on
   it. Re-enabling it is a human call — don't un-quarantine it yourself
   even if it later appears reachable again.

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
`failure_count` remains unexplained AND `queue_audit.py check` exits
clean. This is a normal, expected outcome
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
