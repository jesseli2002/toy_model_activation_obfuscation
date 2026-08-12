"""Human-facing, read-only run-status report across vast_pool_manager.sh
instances -- what's running (and roughly when it'll finish), what's still
queued, and what's already finished (with which succeeded/failed).

Meant to be run directly by a person at a terminal, no agent involved --
that's the point: checking on a sweep shouldn't cost agent context/tokens.
By default it fetches queue.txt/manager.log/launched_idx itself over plain
`ssh <alias> cat <path>`, using the instance's key in instances.json as the
ssh alias (see this skill's module docstring: that alias's local ssh config,
not this script, decides what host it resolves to). Pass --source local to
instead read copies an agent already pulled with the vast-remote-broker
`fetch_files` tool (matches queue_audit.py's fetch-root convention), for
use from inside an agent session where raw ssh may not be reachable.

No sweep-specific knowledge -- everything comes from the generic
instances.json schema (see monitor_brief.md) plus the queue.txt/
manager.log/launched_idx files vast_pool_manager.sh already writes, so this
works unmodified for any sweep/agent that follows that convention.

A run's status per queue line is read off two files, not one:
  - queue.txt gives the line's tag (same --tag parsing queue_audit.py uses)
    and its position.
  - manager.log's `launched idx=N` gives the pid that ran line N, and
    `finished pid=P rc=R` gives that pid's outcome. A line whose index has
    no launch record is either still queued (if launched_idx confirms it
    hasn't been reached) or a gap in the log itself (reported separately,
    not guessed at).

"Completing soon" is estimated from the trailing average duration of that
instance's own already-finished jobs (last --eta-window of them) minus each
running job's elapsed time -- a rough same-instance rate, not a per-job
prediction, so treat it as an ordering hint more than a precise ETA.

Bucket flags (--running etc.) and --tag-regex narrow what gets printed;
they're applied after the report is built, so the ETA average still reflects
every finished job rather than only the visible ones.

Usage:
    python sweep_status.py --instances instances.json
    python sweep_status.py --instances instances.json --running --failed
    python sweep_status.py --instances instances.json --tag-regex 'sweep18_.*lr3'
    python sweep_status.py --instances instances.json --source local --fetch-root /tmp/vast-remote-broker-fetch
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_health import FINISH_RE, LAUNCH_RE, _parse_ts  # noqa: E402
from queue_audit import (  # noqa: E402
    DEFAULT_FETCH_ROOT,
    DEFAULT_TAG,
    load_instances,
    local_copy_of,
    tag_of,
)

FETCH_FIELDS = ("queue", "launched_idx", "manager_log")

# Display order; also the set of per-bucket --flags parse_args() generates.
BUCKETS = ("running", "failed", "complete", "queued", "unknown")


def _regex(pattern):
    # re.PatternError isn't a ValueError, so argparse won't turn it into a
    # usage error on its own.
    try:
        return re.compile(pattern)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"bad regex {pattern!r}: {e}") from e


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--instances", required=True)
    p.add_argument(
        "--source",
        choices=["ssh", "local"],
        default="ssh",
        help="ssh: fetch live via `ssh ALIAS cat PATH` (default, no agent needed). "
        "local: read copies already fetched by an agent's fetch_files call.",
    )
    p.add_argument(
        "--fetch-root",
        default=DEFAULT_FETCH_ROOT,
        help="only used with --source local",
    )
    p.add_argument("--ssh-timeout", type=float, default=15.0)
    p.add_argument("--default-tag", default=DEFAULT_TAG)
    p.add_argument(
        "--eta-window", type=int, default=10, help="trailing finished jobs to average"
    )
    p.add_argument(
        "--show", type=int, default=8, help="tags to list per bucket before eliding"
    )
    p.add_argument("--json", action="store_true", help="emit the full report as JSON")
    p.add_argument(
        "--no-color", action="store_true", help="disable ANSI color even on a tty"
    )
    g = p.add_argument_group(
        "filters", "narrow what is shown; with none of these, everything is shown"
    )
    for bucket in BUCKETS:
        g.add_argument(f"--{bucket}", action="store_true", help=f"show {bucket} runs")
    g.add_argument(
        "--tag-regex",
        type=_regex,
        help="only show tags matching this regex (unanchored search)",
    )
    return p.parse_args()


def selected_buckets(args):
    chosen = tuple(b for b in BUCKETS if getattr(args, b))
    return chosen or BUCKETS


# ---- fetching: both backends return {field: text_or_None} per instance ----


def fetch_ssh(alias, cfg, timeout):
    out = {}
    for field in FETCH_FIELDS:
        path = cfg.get(field)
        if not path:
            out[field] = None
            continue
        try:
            result = subprocess.run(
                ["ssh", alias, "cat", shlex.quote(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            out[field] = None
            out.setdefault("_errors", []).append(f"{field}: {e}")
            continue
        if result.returncode != 0:
            out[field] = None
            out.setdefault("_errors", []).append(
                f"{field}: ssh {alias} exited {result.returncode}: "
                f"{result.stderr.strip()[:200]}"
            )
        else:
            out[field] = result.stdout
    return out


def fetch_local(alias, cfg, fetch_root):
    out = {}
    for field in FETCH_FIELDS:
        path = cfg.get(field)
        if not path:
            out[field] = None
            continue
        local_path = local_copy_of(fetch_root, alias, path)
        try:
            with open(local_path) as f:
                out[field] = f.read()
        except OSError as e:
            out[field] = None
            out.setdefault("_errors", []).append(f"{field}: {local_path}: {e}")
    return out


# ---- parsing (operates on text, not paths, so both backends share it) ----


def parse_queue_tags(text, default_tag):
    tags = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            tag, _resume = tag_of(line, default_tag)
            tags[lineno] = tag or f"(unparseable line {lineno})"
    return tags


def parse_manager_log(text):
    """idx -> (pid, launch_ts), pid -> (rc, finish_ts)."""
    launches, finishes = {}, {}
    for line in text.splitlines():
        m = LAUNCH_RE.match(line)
        if m:
            launches[int(m["idx"])] = (m["pid"], _parse_ts(m["ts"]))
            continue
        m = FINISH_RE.match(line)
        if m:
            finishes[m["pid"]] = (int(m["rc"]), _parse_ts(m["ts"]))
    return launches, finishes


def instance_report(alias, cfg, fetched, default_tag, eta_window, now):
    errors = fetched.get("_errors", [])
    if fetched.get("queue") is None:
        return {"instance": alias, "errors": errors or ["no queue data"]}
    tags = parse_queue_tags(fetched["queue"], default_tag)

    launched_idx = None
    if fetched.get("launched_idx"):
        try:
            launched_idx = int(fetched["launched_idx"].strip())
        except ValueError:
            pass

    launches, finishes = {}, {}
    if fetched.get("manager_log"):
        launches, finishes = parse_manager_log(fetched["manager_log"])

    running, queued, complete, failed, unknown = [], [], [], [], []
    for lineno in sorted(tags):
        tag = tags[lineno]
        if lineno in launches:
            pid, launch_ts = launches[lineno]
            if pid in finishes:
                rc, finish_ts = finishes[pid]
                entry = {"tag": tag, "rc": rc, "duration_s": finish_ts - launch_ts}
                (complete if rc == 0 else failed).append(entry)
            else:
                running.append({"tag": tag, "running_s": now - launch_ts})
        elif launched_idx is not None and lineno <= launched_idx:
            # Sidecar says this line dispatched, but manager.log has no
            # matching launch record -- e.g. a rotated/truncated log, not
            # something to guess a status for.
            unknown.append({"tag": tag, "lineno": lineno})
        else:
            queued.append({"tag": tag, "lineno": lineno})

    # Trailing avg duration of successful finishes -> rough ETA for running.
    recent_ok = complete[-eta_window:] if complete else []
    avg_duration = (
        sum(e["duration_s"] for e in recent_ok) / len(recent_ok) if recent_ok else None
    )
    for e in running:
        e["eta_s"] = (
            max(avg_duration - e["running_s"], 0.0)
            if avg_duration is not None
            else None
        )
    running.sort(key=lambda e: (e["eta_s"] is None, e["eta_s"]))

    return {
        "instance": alias,
        "avg_duration_s": avg_duration,
        "running": running,
        "queued": queued,
        "complete": complete,
        "failed": failed,
        "unknown": unknown,
        "errors": errors,
    }


def filter_report(report, buckets, tag_re):
    """Drop buckets and tags the caller didn't ask for.

    Applied to a finished report rather than folded into instance_report so
    the ETA average keeps averaging over every finished job.
    """
    if "running" not in report:  # error-only report, nothing to filter
        return report
    out = dict(report)
    for bucket in BUCKETS:
        entries = report[bucket] if bucket in buckets else []
        if tag_re is not None:
            entries = [e for e in entries if tag_re.search(e["tag"])]
        out[bucket] = entries
    return out


CYAN = "\033[36m"
RESET = "\033[0m"


def _fmt_dur(seconds):
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _fmt_running(e):
    eta = f", eta ~{_fmt_dur(e['eta_s'])}" if e["eta_s"] is not None else ""
    return f"{e['tag']} (up {_fmt_dur(e['running_s'])}{eta})"


BUCKET_FMT = {
    "running": _fmt_running,
    "failed": lambda e: f"{e['tag']} (rc={e['rc']})",
    "complete": lambda e: e["tag"],
    "queued": lambda e: e["tag"],
    "unknown": lambda e: e["tag"],
}


def print_human(reports, show, color, buckets):
    heading = (lambda s: f"{CYAN}{s}{RESET}") if color else (lambda s: s)
    totals = {b: 0 for b in buckets}
    for r in reports:
        print(heading(f"=== {r['instance']} ==="))
        for err in r.get("errors") or ():
            print(f"  [error] {err}")
        if "running" not in r:
            continue
        for bucket in buckets:
            fmt = BUCKET_FMT[bucket]
            entries = r[bucket]
            totals[bucket] += len(entries)
            if not entries:
                continue
            print(f"  {bucket} ({len(entries)}):")
            for e in entries[:show]:
                print(f"    {fmt(e)}")
            if len(entries) > show:
                print(f"    ... +{len(entries) - show} more")
    print(heading("=== total ==="))
    print("  " + "  ".join(f"{k}={v}" for k, v in totals.items()))


def main():
    args = parse_args()
    instances = load_instances(args.instances)
    buckets = selected_buckets(args)
    now = time.time()

    reports = []
    for alias, cfg in sorted(instances.items()):
        if args.source == "ssh":
            fetched = fetch_ssh(alias, cfg, args.ssh_timeout)
        else:
            fetched = fetch_local(alias, cfg, args.fetch_root)
        report = instance_report(
            alias, cfg, fetched, args.default_tag, args.eta_window, now
        )
        reports.append(filter_report(report, buckets, args.tag_regex))

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        color = sys.stdout.isatty() and not args.no_color
        print_human(reports, args.show, color, buckets)


if __name__ == "__main__":
    main()
