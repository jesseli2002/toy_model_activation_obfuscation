# Monitor brief: periodic parallel-remote-training health check

You are a **fresh agent invocation with no memory of previous checks** —
this doc is meant to be everything you need. Don't assume anything about
prior runs beyond what's in the files below; all real state is durable
(remote log/queue files, `instances.json`), not in any chat history.

## What you're checking

One or more vast.ai instances, each running a `vast_pool_manager.sh`
dispatch loop against its own `queue.txt`/`conc.txt`/`manager.log` (see
`.claude/skills/parallel-remote-training/SKILL.md`), all pulling from a
single sweep's pending pool (`sweep_pool.py`/`generate_sweep8.py`-family
bookkeeping).

## Where to find things

1. Read `sweep8_scratch/instances.json` (or whatever the sweep's out-dir
   is — check with the user/prior context if unclear) for the registered
   instances: `{alias: {scratch_dir, queue, conc, manager_log,
   launched_idx, project_dir, venv_activate}}`. If this file doesn't
   exist yet, that's itself worth flagging — someone needs to create it
   (see schema note at the bottom) before this brief can run unattended.
2. Cross-check against `mcp__vast-remote-broker__list_instances` — an
   instance running there but absent from `instances.json` is a real
   instance not being monitored; flag it, don't silently monitor it via
   guessed paths.

## Each wake, per registered instance

1. **Fetch health data.** `remote_exec cat MANAGER_LOG` and `cat
   LAUNCHED_IDX`, save locally, then run
   `sweep_pool.py`'s underlying `pool_health.py` (or `sweep_pool.py eta`
   for the aggregate) against the local copies — these tools read local
   files only, they can't reach the remote themselves.
2. **Stalled or failing?** (`stalled: true`, or a nonzero/rising
   `failure_count` since you'd reasonably expect) — this is the priority
   signal. Don't try to self-diagnose deeply; surface it to the user with
   the relevant `manager.log` tail. A dead GPU or a systematic failure
   (bad config, OOM, etc.) both need a human, not a retry loop.
3. **Manager process actually alive?** A `STOPPED`/`ALL DONE` line in
   `manager.log` (or, for an older unfixed deployment, silence past when
   the queue should have drained) means the dispatcher itself has
   exited — appending more work would do nothing. Flag this distinctly
   from "stalled but alive"; it needs a restart, not a top-off.
4. **Running low?** Compare `high_water_mark` to the instance's queue
   length and its `jobs_per_hour_trailing` against your target safety
   margin (e.g. keep ≥2h of queued work at the current rate). If low:
   compute `n_runs = target_hours × jobs_per_hour_trailing`, run
   `sweep_pool.py assign --instance ALIAS --n-runs N`, then push the
   resulting delta via `queue_append.sh` over `remote_exec`.
   - If `jobs_per_hour_trailing` is `null` (no recent data — e.g. a
     freshly-started instance), use a conservative fixed-count top-off
     instead of a rate-derived one, and note in your report that the
     next cycle should have real data.
5. **Over-provisioned relative to another instance's rate, or one
   instance is clearly faster/slower than assumed?** **Recommend**
   a `queue_trim.sh` + reassign to the user rather than doing it
   yourself — this path has only been tested against a paused manager
   locally, never against one actively launching jobs under real
   concurrent load. State the specific trim/reassign command you'd run
   and why, and let a human execute it the first few times.
6. **Any live concurrency experiment you choose to run** (bump
   `conc.txt`, observe, keep/revert) must poll in chunks of **≤3
   minutes** per wait step — never one long sleep — so your own context
   doesn't go stale mid-wake. Log any change you make to a
   `conc_history.txt` sidecar (`timestamp new_value`) as passive context
   for the next wake, since you won't remember doing it.

## Done condition

Sweep is complete when: the pending pool is empty (`sweep_pool.py
status`/`generate_sweep8.py status` shows 0 pending) AND every registered
instance's queue has drained AND no `stalled`/nonzero `failure_count`
remains unexplained. Report this plainly rather than continuing to poll —
whatever scheduled this brief should stop firing once you say so.

## instances.json schema (if you need to create/update it)

```json
{
  "vtao": {
    "scratch_dir": "/home/agent/sweep8_scratch",
    "queue": "/home/agent/sweep8_scratch/queue.txt",
    "conc": "/home/agent/sweep8_scratch/conc.txt",
    "manager_log": "/home/agent/sweep8_scratch/manager.log",
    "launched_idx": "/home/agent/sweep8_scratch/queue.txt.launched_idx",
    "project_dir": "/workspace/toy_probe_hiding",
    "venv_activate": "/workspace/.venv/bin/activate"
  }
}
```

The `vtao` entry above reflects what was actually deployed there as of
this writing (confirmed via `remote_exec`, read-only) — reuse it if it's
still current, but verify rather than assume, since instances get rented
and destroyed between sweeps. `launched_idx`/`queue.lock` paths only
exist once the fixed `vast_pool_manager.sh` (with the drain/locking fix)
is the one actually deployed there — an older deployment won't have them;
note that distinctly rather than treating a missing sidecar as an error.
