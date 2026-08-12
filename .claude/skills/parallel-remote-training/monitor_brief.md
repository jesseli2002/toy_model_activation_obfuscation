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
   `fetch_files` tool, and run `pool_health.py` against the local copies
   it writes — these tools read local files only, they can't reach the
   remote themselves. Do this
   once for all instances at the start of the wake; it also gives you the
   queues that step 8 audits. A `remote_exec` redirect writes on the
   remote, not here, so it is not a substitute.

   For a real stall verdict rather than `unknown`, give `pool_health.py`
   its liveness inputs too — the fetched queue copy, the local `runs/`
   tree the remote rsyncs into, and the instance's sync heartbeat:

   ```
   python pool_health.py <fetched manager.log> \
     --launched-idx <fetched launched_idx> \
     --queue <fetched queue.txt> --sync-host <alias>
   ```

   `--runs-dir` defaults to the main checkout's `runs/`, which is right
   even when you're running from a worktree (worktrees don't have one).
   Nothing here touches the remote; run progress is read from the local
   mtimes the ~10s runs-pull keeps current.
2. **Failing?** A nonzero-`rc` line in `MANAGER_LOG` — see "Auto-retry on
   failure" below before escalating; it's not automatically a human
   issue anymore.
3. **Stalled?** Read `status`, not any single number. It is three-state:
   - `ok` — nothing to do. Note this includes a silent `manager.log`, as
     long as some running job is still progressing: one long run produces
     hours of silence between its own launch and finish lines, and that is
     healthy. (This used to be reported as `stalled: true` and cost every
     wake a manual `ps aux` detour to disprove — it no longer fires, and
     the old `stalled` field is gone.)
   - `stalled` — real. No running job's `history.jsonl` has advanced while
     the runs-pull sync is demonstrably healthy, or the manager has gone
     quiet with nothing running at all. Don't self-diagnose deeply:
     surface to `handoff.md` and the user with the `manager.log` tail. A
     systematic failure (bad config, OOM, etc.) needs a human.
   - `unknown` — the check couldn't reach a verdict; `diagnosis` says why.
     Usually either the liveness inputs weren't passed, the fetched
     `manager.log` copy is too old to trust, or the runs-pull loop's
     heartbeat is stale — in which case frozen `history.jsonl` mtimes mean
     "we stopped hearing from the box", which is *not* the same as "the box
     stopped working". Both possibilities are worth investigating; check
     the sync loop first (`sync_vastai.py status`), since it's the cheaper
     one to rule out and it's local.

   A rising failure count that doesn't match the auto-retry path is a
   separate escalation regardless of `status`. If the instance is actually
   unreachable rather than stalled, see "Instance unreachable" below —
   that one you can act on yourself.
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
   `handoff.md` immediately. A `concurrent-dispatch-risk` finding means
   one tag has two occurrences still awaiting dispatch — most likely a
   second retry appended before the first was launched; trim all but one
   before they start together. `unregistered`/`misplaced` findings mean
   something reached a queue outside `sweep_pool.py`; log them and re-run
   the audit after correcting the registry.
8. **Reassess optimal concurrency every wake, not just when something
   looks wrong.** The queue is heterogeneous (different model sizes,
   different `--max-iters`, different per-job GPU/memory footprints) —
   the concurrency level that was optimal for an earlier mix of running
   jobs can drift as the mix changes, in either direction. Treat this as
   routine upkeep, same tier as the health read in step 1, not an
   optional experiment you might skip.

   **Measure correctly, not off the printed `it/s` field.** Each job's
   printed `it/s` is a running average over *that job's entire lifetime*
   and keeps climbing for a while after startup regardless of
   concurrency — comparing two single-snapshot readings at different
   concurrency levels conflates "rate matured" with "concurrency
   changed" and reads as a false improvement. Instead, sample the last
   `iter <N>` line of every running `job<N>.log`, wait 30s to warm up,
   then measure over a 30s window (sample again 30s later), and sum `(iter_t1 - iter_t0)`
   across jobs, divided by elapsed wall-clock time — that is the
   aggregate instantaneous throughput. Only compare two such windows
   that each fall entirely within one concurrency setting.

   **Testing an increase:** bump `conc.txt` by one GPU's worth of jobs
   (i.e. by `N`, the instance's GPU count — see SKILL.md's "Ramping
   concurrency"), wait for the new jobs to
   land, measure per the method above. If the new level is a net
   regression (aggregate throughput *drops*, not just per-job rate — an
   increase that raises aggregate while lowering per-job rate is fine,
   that's the expected tradeoff of adding parallelism up to the real
   ceiling), **lowering `conc.txt` back down is not sufficient by
   itself** — it only stops future launches, and the newly-added job(s)
   that caused the regression keep running (and keep dragging down
   everyone else's throughput) for as long as they'd otherwise take,
   which can be hours. Actually remove them: SIGINT the specific job(s)
   this experiment just added (their `launched pid=...` lines are the
   newest in `manager.log`), following the same stop/confirm/ledger/
   requeue steps as "testing a decrease" immediately below — treat a
   failed increase-experiment as "now testing a decrease back to the
   prior level," not a separate case.

   **Testing a decrease is also permitted**, via SIGINT + `--resume`
   rather than waiting for natural completions to thin the pool out —
   useful when you suspect the *current* level (inherited from an
   earlier, different job mix) is already past the ceiling:
   1. Pick the candidate target a full `N` (GPU count) below the current
      level, same step size as an increase. Lower `conc.txt` to it
      **first**. The manager
      never kills already-running jobs on a lower target, it only stops
      backfilling finished ones — so this alone doesn't reduce the
      running count, but it prevents the manager from immediately
      refilling the slot(s) you're about to free, which would defeat
      the measurement.
   2. Pick the newest running job(s) to stop (their `launched pid=...`
      line is near the tail of `manager.log`) until the running count
      matches the candidate target.
   3. `kill -INT <pid>` on each, via `remote_exec`. This reaches the
      training process directly — `vast_pool_manager.sh` launches jobs
      as `bash -c "$CMD" ... &` with nothing following in the string, so
      bash execs straight into `python`, and the pid `manager.log`
      records for that job **is** the training process, not a wrapper.
   4. SIGINT is safe here: the training loop defers it, and only
      breaks/checkpoints/exits at an iteration boundary, so `last.pt`
      and `history.jsonl` are left consistent — confirmed by reading
      `train_adversarial_logreg.py`'s `_defer_keyboard_interrupt`/
      `save_checkpoint`/`_atomic_write`. The one gap is the first ~10s
      of a *fresh* run's startup (before the training loop begins, e.g.
      during initial probe fit) — a SIGINT landing there is uncaught and
      leaves no checkpoint. Not a concern in practice (Jesse: "losing
      data in initialization is not a worry, that takes <10s"); just
      don't be surprised if a just-launched job's stop leaves nothing to
      resume — requeue it as a fresh `--retry-tag` instead in that case.
   5. Confirm the stop actually landed (poll for `logs/job<N>.log`'s
      `[save] ... -> ...` line, or `manager.log`'s `finished pid=...
      rc=...` line for that pid) before measuring the new throughput —
      same 30s-warmup/30s-measure, iter-delta method as above.
   6. **Record every SIGINT you send in the manual-stop ledger in
      `handoff.md` immediately** (tag, pid, iteration it was stopped at,
      timestamp, `conc.txt` target you were testing) — see "Manual-stop
      ledger" below. This is required, not optional bookkeeping: without
      it, the next `rc=[1-9]` audit (yours or a future wake's) can't
      tell your own SIGINTs apart from real crashes.
   7. Requeue the stopped tag the normal way — `sweep_pool.py requeue
      --instance ALIAS --resume-tag <tag>` + `queue_append.sh`, appended
      at the queue's end (same path as an eligible auto-retry below) —
      once you're done with the experiment, or immediately if you expect
      it to relaunch at the new lower concurrency anyway.

   All polling here (either direction) uses the **30s-warmup/30s-measure**
   window per wait step — never one long sleep — both to keep your own
   context from going stale mid-wake and to stay under the ~5min
   prompt-cache TTL. Log
   the outcome (kept new level / reverted, and why) to `handoff.md`.

### Manual-stop ledger

`handoff.md` must carry a running ledger of every tag you've personally
SIGINT'd for a concurrency experiment (point 8) and not yet resolved —
e.g. a `## Manual stops` section with one line per entry:
`<tag> | pid=<pid> | stopped at iter <N> | <timestamp> | reason: <why> |
status: pending-resume / resumed`.

This is what makes `rc=[1-9]` counts interpretable: a SIGINT'd job exits
nonzero (an uncaught `KeyboardInterrupt` past the deferred block) and
looks identical, by exit code alone, to a real crash. Before triaging any
`rc=[1-9]` line (step 2 below, and the "Auto-retry on failure" section),
cross-check its tag against this ledger first. The count of *unexplained*
failures for step 2's "Failing?" and the "Done condition" check is
`(total rc=[1-9] lines) − (lines whose tag is a ledger entry)` — mark a
ledger entry `resumed` once you've requeued it, so it stops being
subtracted forever and a *second*, unexpected failure on that same tag
still surfaces normally.

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
registered instance's queue has drained AND no `stalled` status or nonzero
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
