"""Sweep-agnostic health/rate reader for a vast_pool_manager.sh instance.

Reads one instance's manager.log (+ its launched_idx.txt sidecar) and prints
a JSON summary: how many jobs are currently running, the dispatch
high-water mark, how many have failed, whether the manager looks stalled,
and a *trailing-window* jobs/hour rate -- not a lifetime average, since the
mix of runs an instance is chewing through shifts across a sweep's stages
(see project memory: params-ratio weight is a bad throughput proxy, so
sizing/ETA decisions should use this measured rate instead).

No sweep-specific knowledge (doesn't know about tags, configs, or
manifests) -- reusable by generate_sweep8.py/sweep_pool.py or any future
sweep.

Usage:
    python pool_health.py MANAGER_LOG [--launched-idx PATH] [--window-minutes 60]
"""

import argparse
import json
import re
import time
from datetime import datetime

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
        help="age since the last log event, above which 'stalled' is reported",
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


def main():
    args = parse_args()
    launches, finishes, last_event_ts = read_log(args.manager_log)

    finished_pids = {pid for _, pid, _ in finishes}
    running = [pid for _, pid, _ in launches if pid not in finished_pids]
    failures = sum(1 for _, _, rc in finishes if rc != 0)

    if args.launched_idx:
        try:
            with open(args.launched_idx) as f:
                high_water_mark = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            high_water_mark = None
    else:
        high_water_mark = max((idx for _, _, idx in launches), default=0)

    now = time.time()
    stalled = (
        last_event_ts is not None
        and (now - last_event_ts) > args.stall_threshold_seconds
    )

    window_start = now - args.window_minutes * 60
    recent_finishes = [(ts, rc) for ts, _, rc in finishes if ts >= window_start]
    if len(recent_finishes) >= 2:
        span_hours = (recent_finishes[-1][0] - recent_finishes[0][0]) / 3600.0
        jobs_per_hour = (
            (len(recent_finishes) - 1) / span_hours if span_hours > 0 else None
        )
    else:
        jobs_per_hour = None

    result = {
        "running_count": len(running),
        "high_water_mark": high_water_mark,
        "failure_count": failures,
        "total_finished": len(finishes),
        "last_event_age_seconds": (
            (now - last_event_ts) if last_event_ts is not None else None
        ),
        "stalled": stalled,
        "jobs_per_hour_trailing": jobs_per_hour,
        "trailing_window_minutes": args.window_minutes,
        "trailing_window_sample_size": len(recent_finishes),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
