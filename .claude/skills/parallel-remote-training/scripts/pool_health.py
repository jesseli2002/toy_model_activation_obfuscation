"""Sweep-agnostic health/rate reader for a vast_pool_manager.sh instance.

Reads one instance's manager.log (+ its launched_idx.txt sidecar) and prints
a JSON summary: how many jobs are currently running, the dispatch
high-water mark, how many have failed, whether the manager looks healthy,
and a *trailing-window* jobs/hour rate -- not a lifetime average, since the
mix of runs an instance is chewing through shifts across a sweep's stages
(see project memory: params-ratio weight is a bad throughput proxy, so
queue-sizing decisions should use this measured rate instead).

That rate is for coarse top-off sizing only, NOT for an ETA. It averages
over heterogeneous runs and counts failures as completions, so on a sweep
whose jobs differ in length it can be off by several-fold. Estimate
remaining time from the running jobs' iteration throughput instead.

Liveness: manager.log alone cannot answer "is this box still working?".
The manager only writes on launch/finish/idle, so one long run produces an
hours-long silence that is perfectly healthy -- time-since-last-log-line on
its own reports that as stalled, which is a false positive frequent enough
to make the signal worse than useless. The real progress signal is each
running job's `runs/<tag>/logs/history.jsonl`, appended (open/write/close,
so actually flushed) every --log-interval iters by the train script. Those
files rsync back to the local runs/ every ~10s with `-a`, which preserves
mtime, so the *local* copy's mtime is the remote's -- no clock skew and no
extra remote call. Point --runs-dir at that local tree to get real
liveness; without it this reports "unknown" rather than guessing.

Because a dead sync loop also freezes those mtimes, --sync-host reads the
pull loop's own heartbeat (written only after a successful rsync) to tell
"runs stopped progressing" apart from "we stopped hearing about them".

Knows about `--tag` and the runs/<tag>/logs/ layout, i.e. this project's
train script -- but still nothing sweep-specific (no configs, manifests, or
per-sweep naming), so it stays reusable by sweep_pool.py or any future
sweep.

Usage:
    python pool_health.py MANAGER_LOG [--launched-idx PATH] [--window-minutes 60]
    python pool_health.py MANAGER_LOG --queue QUEUE_COPY --sync-host vtao
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queue_audit import DEFAULT_TAG, tag_of  # noqa: E402

LAUNCH_RE = re.compile(
    r"^(?P<ts>\S+) launched pid=(?P<pid>\d+) idx=(?P<idx>\d+)/(?P<total>\d+):"
)
FINISH_RE = re.compile(r"^(?P<ts>\S+) finished pid=(?P<pid>\d+) \S+ rc=(?P<rc>-?\d+)")
IDLE_RE = re.compile(r"^(?P<ts>\S+) queue drained, idling")
STOPPED_RE = re.compile(r"^(?P<ts>\S+) STOPPED")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manager_log")
    p.add_argument(
        "--launched-idx",
        default=None,
        help="path to QUEUE.launched_idx sidecar (default: not read; "
        "high_water_mark comes from the log's launch lines instead, which "
        "undercounts if the process crashed between launching and logging -- "
        "prefer passing this explicitly)",
    )
    p.add_argument(
        "--window-minutes",
        type=float,
        default=60.0,
        help="trailing window for the jobs/hour rate estimate",
    )
    p.add_argument(
        "--stall-threshold-seconds",
        type=float,
        default=600.0,
        help="age since the last manager.log event, above which the manager is "
        "considered quiet. Quiet alone is NOT stalled -- it only means the "
        "verdict has to come from the running jobs instead",
    )
    p.add_argument(
        "--queue",
        default=None,
        help="local copy of this instance's queue.txt, used to map a running "
        "job's index to its --tag (required for run-liveness)",
    )
    p.add_argument(
        "--runs-dir",
        default=None,
        help="local runs/ tree the remote rsyncs into (default: <repo>/runs). "
        "Without a readable one, run-liveness is 'unknown', never 'stalled'",
    )
    p.add_argument(
        "--sync-host",
        default=None,
        help="ssh alias whose runs-pull heartbeat to read, to tell a dead sync "
        "loop apart from dead runs (see sync_vastai.py)",
    )
    p.add_argument(
        "--sync-state-dir",
        default=None,
        help="directory holding <host>.last_sync (default: "
        "<repo>/vast_setup/.sync_state)",
    )
    p.add_argument(
        "--run-stall-seconds",
        type=float,
        default=300.0,
        help="age of a running job's history.jsonl above which that job is "
        "considered stalled. A healthy mid-run job's local copy was measured "
        "updating every 12-18s (rsync-paced, not iteration-paced: entries are "
        "written far faster than the ~15s pull), so the default is ~17x the "
        "observed worst case",
    )
    p.add_argument(
        "--startup-grace-seconds",
        type=float,
        default=1800.0,
        help="how long after launch a job may have no history.jsonl at all "
        "(imports, data gen, and the first --log-interval iters) before its "
        "absence counts against it",
    )
    p.add_argument("--default-tag", default=DEFAULT_TAG)
    p.add_argument(
        "--max-snapshot-age-minutes",
        type=float,
        default=30.0,
        help="a fetched manager.log copy older than this can't support a "
        "'stalled' verdict (matches queue_audit.py's --max-age-minutes)",
    )
    return p.parse_args()


def _parse_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def read_log(path: str):
    launches = []  # (ts, pid, idx)
    finishes = []  # (ts, pid, rc)
    last_event_ts = None
    with open(path) as f:
        for line in f:
            m = LAUNCH_RE.match(line)
            if m:
                ts = _parse_ts(m["ts"])
                launches.append((ts, m["pid"], int(m["idx"])))
                last_event_ts = ts if last_event_ts is None else max(last_event_ts, ts)
                continue
            m = FINISH_RE.match(line)
            if m:
                ts = _parse_ts(m["ts"])
                finishes.append((ts, m["pid"], int(m["rc"])))
                last_event_ts = ts if last_event_ts is None else max(last_event_ts, ts)
                continue
            m = IDLE_RE.match(line) or STOPPED_RE.match(line)
            if m:
                ts = _parse_ts(m["ts"])
                last_event_ts = ts if last_event_ts is None else max(last_event_ts, ts)
    return launches, finishes, last_event_ts


def _repo_root() -> str | None:
    """The main checkout, even when called from a worktree -- runs/ and
    vast_setup/ are gitignored, so they exist only there."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return os.path.dirname(out.stdout.strip()) or None


def _age(path: str) -> float | None:
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return None


def tags_by_idx(queue_path: str, default_tag: str = DEFAULT_TAG) -> dict[int, str]:
    """Queue line number -> run tag. Line numbers are 1-based to match the
    manager's idx, which indexes QUEUE by line."""
    mapping = {}
    try:
        with open(queue_path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                tag, _ = tag_of(line, default_tag)
                if tag is not None:
                    mapping[lineno] = tag
    except OSError:
        return {}
    return mapping


def sync_heartbeat_age(host: str, state_dir: str | None) -> float | None:
    """Seconds since the runs-pull loop last completed an rsync for `host`,
    or None if there's no readable heartbeat.

    sync_vastai.py writes this only on rsync success, so a growing age means
    the pull loop is down or failing -- which would freeze every history.jsonl
    mtime below and otherwise masquerade as every run stalling at once.
    """
    if state_dir is None:
        root = _repo_root()
        if root is None:
            return None
        state_dir = os.path.join(root, "vast_setup", ".sync_state")
    path = os.path.join(state_dir, f"{host}.last_sync")
    try:
        with open(path) as f:
            return time.time() - float(f.read().strip())
    except (OSError, ValueError):
        return None


def run_liveness(
    running: list[tuple[float, str, int]],
    runs_dir: str,
    idx_to_tag: dict[int, str],
    run_stall_seconds: float,
    startup_grace_seconds: float,
) -> list[dict]:
    """Per running job, how long since its history.jsonl last grew.

    States: `advancing` (fresh entry), `stale` (nothing recent), `starting`
    (no file yet, still inside the startup grace), `never-started` (no file
    well past it), `unknown-tag` (queue line unavailable, so nothing to stat).

    One `stale` job is not a problem on its own and must not be escalated as
    one: a run goes quiet after its last logged iter while it finishes its
    closing eval/checkpoint, so jobs near completion routinely sit here for
    minutes. Only an all-stale box is evidence of a stall.
    """
    now = time.time()
    out = []
    for launch_ts, pid, idx in running:
        tag = idx_to_tag.get(idx)
        entry = {"idx": idx, "pid": pid, "tag": tag}
        if tag is None:
            entry["state"] = "unknown-tag"
            out.append(entry)
            continue
        path = os.path.join(runs_dir, tag, "logs", "history.jsonl")
        age = _age(path)
        entry["history"] = path
        entry["history_age_seconds"] = age
        if age is None:
            since_launch = now - launch_ts
            entry["state"] = (
                "starting" if since_launch <= startup_grace_seconds else "never-started"
            )
        else:
            entry["state"] = "advancing" if age <= run_stall_seconds else "stale"
        out.append(entry)
    return out


def _verdict(manager_quiet, jobs, have_liveness, sync_age, sync_stall_seconds):
    """Fold manager silence, per-run progress, and sync health into one
    status. Manager silence alone is never enough to call a stall -- that
    was the old false positive.
    """
    if not manager_quiet:
        return "ok", "manager logged an event recently"
    if not have_liveness:
        return "unknown", (
            "manager.log has been quiet, which is normal during a long run -- "
            "pass --queue/--runs-dir to check whether the runs are progressing"
        )
    if not jobs:
        return "stalled", (
            "manager.log has been quiet and no job is running -- the manager "
            "itself is not even idle-logging, so it likely died"
        )
    if any(j["state"] in ("advancing", "starting") for j in jobs):
        return "ok", "manager.log is quiet but running jobs are still progressing"

    sync_down = sync_age is None or sync_age > sync_stall_seconds
    if sync_down:
        reason = (
            "no readable sync heartbeat"
            if sync_age is None
            else f"last successful pull was {sync_age / 60:.0f} min ago"
        )
        return "unknown", (
            f"no running job's history.jsonl has advanced, but {reason} -- either "
            "the runs-pull loop is down (so these mtimes are just frozen) or the "
            "remote has stalled. Both need investigating; check the sync loop "
            "first, since it is the cheaper of the two to rule out"
        )
    return "stalled", (
        "the runs-pull sync is healthy but no running job's history.jsonl has "
        "advanced -- the remote is not making progress"
    )


def compute_health(
    manager_log: str,
    launched_idx: str | None = None,
    window_minutes: float = 60.0,
    stall_threshold_seconds: float = 600.0,
    queue: str | None = None,
    runs_dir: str | None = None,
    sync_host: str | None = None,
    sync_state_dir: str | None = None,
    run_stall_seconds: float = 300.0,
    startup_grace_seconds: float = 1800.0,
    default_tag: str = DEFAULT_TAG,
    max_snapshot_age_minutes: float = 30.0,
) -> dict:
    """Importable core of this script -- for callers that want the health
    dict in-process (no subprocess/JSON round-trip), as well as this
    module's own CLI."""
    launches, finishes, last_event_ts = read_log(manager_log)

    finished_pids = {pid for _, pid, _ in finishes}
    running = [(ts, pid, idx) for ts, pid, idx in launches if pid not in finished_pids]
    failures = sum(1 for _, _, rc in finishes if rc != 0)

    if launched_idx:
        try:
            with open(launched_idx) as f:
                high_water_mark = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            high_water_mark = None
    else:
        high_water_mark = max((idx for _, _, idx in launches), default=0)

    now = time.time()
    # Silence is measured against when this manager.log copy was *observed*,
    # not wall-clock. A fetched copy is a snapshot: comparing its newest
    # timestamp to now() charges every minute since the fetch to the manager,
    # so a stale fetch alone reported a stall (observed: a 17min-old copy
    # whose last event was 5s before the fetch, read as 17min of silence).
    # On a live log (read in place) mtime is now, so this is a no-op there.
    observed_at = os.path.getmtime(manager_log) if os.path.exists(manager_log) else now
    snapshot_age = max(0.0, now - observed_at)
    snapshot_stale = snapshot_age > max_snapshot_age_minutes * 60
    manager_quiet = (
        last_event_ts is None or (observed_at - last_event_ts) > stall_threshold_seconds
    )

    # Only resolve the default runs/ when liveness was actually asked for --
    # callers that just want the rate shouldn't pay for a git subprocess
    # per instance.
    if queue and runs_dir is None:
        root = _repo_root()
        runs_dir = os.path.join(root, "runs") if root else None
    have_liveness = bool(queue) and bool(runs_dir) and os.path.isdir(runs_dir)
    jobs = (
        run_liveness(
            running,
            runs_dir,
            tags_by_idx(queue, default_tag),
            run_stall_seconds,
            startup_grace_seconds,
        )
        if have_liveness
        else []
    )
    sync_age = sync_heartbeat_age(sync_host, sync_state_dir) if sync_host else None
    status, diagnosis = _verdict(
        manager_quiet, jobs, have_liveness, sync_age, run_stall_seconds
    )
    if snapshot_stale:
        # runs/ is live-synced but this manager.log copy isn't, so the two
        # legs disagree about "now": the running set is whatever was running
        # at fetch time, and a job that has since finished still gets its
        # history.jsonl aged against the live clock, reading as "stale".
        # Never escalate off that.
        note = (
            f"this manager.log copy is {snapshot_age / 60:.0f} min old, so the "
            "running-job set is out of date (jobs shown stale may simply have "
            "finished since) -- re-fetch before trusting it"
        )
        status = "unknown" if status == "stalled" else status
        diagnosis = f"{diagnosis}; {note}"

    window_start = now - window_minutes * 60
    recent_finishes = [(ts, rc) for ts, _, rc in finishes if ts >= window_start]
    if recent_finishes:
        # Denominator is the window itself, not the span between the first
        # and last finish in it -- two finishes 3min apart in a 60min
        # window means ~2/hr (little happened since), not ~20/hr (dividing
        # by the 3min sub-span). Cap to time-since-first-launch so a
        # manager that's only been alive 5min doesn't get diluted by a
        # window mostly spent not existing yet.
        window_hours = window_minutes / 60.0
        if launches:
            hours_since_first_launch = (now - launches[0][0]) / 3600.0
            window_hours = min(window_hours, hours_since_first_launch)
        jobs_per_hour = (
            len(recent_finishes) / window_hours if window_hours > 0 else None
        )
    else:
        jobs_per_hour = None

    return {
        "running_count": len(running),
        "high_water_mark": high_water_mark,
        "failure_count": failures,
        "total_finished": len(finishes),
        # Silence as of the snapshot; see observed_at above for why this is
        # not measured from now().
        "last_event_age_seconds": (
            (observed_at - last_event_ts) if last_event_ts is not None else None
        ),
        "snapshot_age_seconds": snapshot_age,
        # Three-state on purpose: "unknown" is what a quiet manager.log alone
        # buys you. Only run-liveness can promote that to "stalled".
        "status": status,
        "diagnosis": diagnosis,
        "manager_quiet": manager_quiet,
        "running_jobs": jobs,
        "sync_heartbeat_age_seconds": sync_age,
        "jobs_per_hour_trailing": jobs_per_hour,
        "trailing_window_minutes": window_minutes,
        "trailing_window_sample_size": len(recent_finishes),
    }


def main():
    args = parse_args()
    result = compute_health(
        args.manager_log,
        launched_idx=args.launched_idx,
        window_minutes=args.window_minutes,
        stall_threshold_seconds=args.stall_threshold_seconds,
        queue=args.queue,
        runs_dir=args.runs_dir,
        sync_host=args.sync_host,
        sync_state_dir=args.sync_state_dir,
        run_stall_seconds=args.run_stall_seconds,
        startup_grace_seconds=args.startup_grace_seconds,
        default_tag=args.default_tag,
        max_snapshot_age_minutes=args.max_snapshot_age_minutes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
