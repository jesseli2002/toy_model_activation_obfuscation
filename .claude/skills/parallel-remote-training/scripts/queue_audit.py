"""Cross-instance queue audit: catch tag collisions between remote pool queues.

Nothing stops two instances from being handed the same run tag, and nothing
downstream notices: each box has its own runs/ (so the train script's
existing same-tag guard can't see the other box), and every box rsyncs into
one local runs/ with last-writer-wins per file. The result is two runs
quietly interleaving their checkpoints under one tag. This script is the
detector for that.

It reads *locally-fetched copies* of each instance's queue.txt and
launched_idx -- fetch them first with the vast-remote-broker `fetch_files`
tool, whose fixed destination layout this script mirrors. `fetch-request`
prints exactly the argument that tool wants, so the path mapping lives in
one place rather than being hand-assembled per call. A copy that is missing
or older than --max-age-minutes is reported rather than trusted, so a
failed fetch can't read as a clean audit.

Duplicates are classified by whether each occurrence sits at or below that
instance's launched_idx: a collision still in the undispatched tail is
fixable with queue_trim.sh, one already dispatched is a live clobber and
needs a human.

Cross-checks the queues against sweep_pool.py's assignments.json, which is
the intended single registry of tag -> instance. Work that reached a queue
without a row there bypassed that funnel (historically: retries appended
straight through queue_append.sh) and is reported as unregistered.

Usage:
    python queue_audit.py fetch-request --instances instances.json
    python queue_audit.py check --instances instances.json --out-dir SWEEP_POOL_DIR

Exit status is nonzero from `check` if any ERROR-level finding is present,
so a monitor can gate on it. Warnings alone do not fail.
"""

import argparse
import json
import os
import shlex
import sys
import time

# Must match vast_remote_broker.DEFAULT_FETCH_DIR, and the same
# <root>/<instance>/<remote path minus leading slash> layout that tool
# documents. Overridable for a server configured with VAST_REMOTE_FETCH_DIR.
DEFAULT_FETCH_ROOT = "/tmp/vast-remote-broker-fetch"

# train_adversarial_logreg.py's --tag default: a queued command that omits
# --tag still runs under this tag, so two such lines on different boxes are
# a real collision and must not be silently skipped.
DEFAULT_TAG = "adv-logreg"

# Which instances.json fields name a remote file worth pulling. manager_log
# isn't used by this audit, but the monitor needs it locally anyway (for
# pool_health.py) and one fetch serving both beats two.
FETCH_FIELDS = ("queue", "launched_idx", "manager_log")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_req = sub.add_parser(
        "fetch-request",
        help="print the `fetches` argument for the broker's fetch_files tool",
    )
    p_req.add_argument("--instances", required=True)

    p_check = sub.add_parser("check", help="audit the fetched queues")
    p_check.add_argument("--instances", required=True)
    p_check.add_argument(
        "--out-dir",
        default=None,
        help="sweep_pool.py --out-dir holding assignments.json. Omit to check only "
        "for duplicates between queues, skipping every registry cross-check.",
    )
    p_check.add_argument("--fetch-root", default=DEFAULT_FETCH_ROOT)
    p_check.add_argument(
        "--max-age-minutes",
        type=float,
        default=30.0,
        help="a fetched copy older than this is reported as stale rather than "
        "trusted (default 30)",
    )
    p_check.add_argument("--default-tag", default=DEFAULT_TAG)
    p_check.add_argument("--json", action="store_true", help="emit findings as JSON")

    return p.parse_args()


def load_instances(path):
    with open(path) as f:
        instances = json.load(f)
    if not isinstance(instances, dict) or not instances:
        raise SystemExit(
            f"[error] {path} must be a non-empty object of alias -> config"
        )
    return instances


def local_copy_of(fetch_root, instance, remote_path):
    """The path fetch_files lands `remote_path` at -- see that tool's docs."""
    return os.path.join(fetch_root, instance, remote_path.lstrip("/"))


def tag_of(command, default_tag):
    """Extract the run tag a queued command would use, and whether it resumes.

    A resuming line legitimately repeats a tag already on that same instance
    (that is what the retry path emits), so the two have to be told apart.
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return None, False
    tag, resume = default_tag, False
    for i, token in enumerate(argv):
        if token == "--resume":
            resume = True
        elif token == "--tag" and i + 1 < len(argv):
            tag = argv[i + 1]
        elif token.startswith("--tag="):
            tag = token.split("=", 1)[1]
    return tag, resume


def read_queues(instances, fetch_root, max_age_minutes, default_tag, findings):
    """Parse every instance's fetched queue into a list of queued-line records."""
    entries = []
    now = time.time()
    for alias, cfg in sorted(instances.items()):
        queue_remote = cfg.get("queue")
        if not queue_remote:
            findings.append(
                ("ERROR", "config", f"{alias}: instances.json has no 'queue' path")
            )
            continue
        queue_local = local_copy_of(fetch_root, alias, queue_remote)
        if not os.path.isfile(queue_local):
            findings.append(
                (
                    "ERROR",
                    "missing-fetch",
                    f"{alias}: no local copy of {queue_remote} at {queue_local} -- "
                    "fetch it first; this instance was NOT audited",
                )
            )
            continue
        age_min = (now - os.path.getmtime(queue_local)) / 60.0
        if age_min > max_age_minutes:
            findings.append(
                (
                    "WARN",
                    "stale-fetch",
                    f"{alias}: local copy of {queue_remote} is {age_min:.0f} min old "
                    f"(limit {max_age_minutes:.0f}) -- re-fetch before trusting this",
                )
            )

        launched = None
        idx_remote = cfg.get("launched_idx")
        if idx_remote:
            idx_local = local_copy_of(fetch_root, alias, idx_remote)
            try:
                with open(idx_local) as f:
                    launched = int(f.read().strip())
            except (OSError, ValueError):
                findings.append(
                    (
                        "WARN",
                        "missing-fetch",
                        f"{alias}: no readable launched_idx copy at {idx_local} -- "
                        "duplicates can't be classified as dispatched vs. trimmable",
                    )
                )

        with open(queue_local) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                tag, resume = tag_of(line, default_tag)
                if tag is None:
                    findings.append(
                        (
                            "WARN",
                            "unparseable",
                            f"{alias} line {lineno}: could not parse as a shell "
                            f"command: {line[:80]}",
                        )
                    )
                    continue
                entries.append(
                    {
                        "instance": alias,
                        "lineno": lineno,
                        "tag": tag,
                        "resume": resume,
                        "command": line,
                        # None when launched_idx wasn't readable: unknown, not false.
                        "dispatched": None if launched is None else lineno <= launched,
                    }
                )
    return entries


def _where(entry):
    if entry["dispatched"] is None:
        state = "dispatch state unknown"
    elif entry["dispatched"]:
        state = "ALREADY DISPATCHED"
    else:
        state = "not yet dispatched, trimmable"
    resume = ", --resume" if entry["resume"] else ""
    return f"{entry['instance']} line {entry['lineno']} ({state}{resume})"


def check_duplicates(entries, findings):
    by_tag = {}
    for e in entries:
        by_tag.setdefault(e["tag"], []).append(e)

    for tag, group in sorted(by_tag.items()):
        instances = {e["instance"] for e in group}
        if len(instances) > 1:
            live = any(e["dispatched"] for e in group)
            findings.append(
                (
                    "ERROR",
                    "duplicate-across-instances",
                    f"tag {tag!r} is queued on {len(instances)} instances "
                    f"({', '.join(sorted(instances))}) -- "
                    + (
                        "at least one occurrence is already dispatched, so these are "
                        "clobbering each other now; needs a human"
                        if live
                        else "none dispatched yet, so the extra copies can still be "
                        "removed with queue_trim.sh"
                    )
                    + "\n      "
                    + "\n      ".join(_where(e) for e in group),
                )
            )
            continue
        # Sequential resumes of one tag are the normal retry path and any
        # number of them is fine -- but only because each is appended after
        # the previous one was seen to fail, so it is already dispatched.
        # Two occurrences still awaiting dispatch can be launched in the same
        # poll (the manager fills free slots without looking at tags), which
        # puts two live processes on one runs/<tag>.
        undispatched = [e for e in group if e["dispatched"] is False]
        if len(undispatched) > 1:
            findings.append(
                (
                    "ERROR",
                    "concurrent-dispatch-risk",
                    f"tag {tag!r} has {len(undispatched)} occurrences still awaiting "
                    f"dispatch on {group[0]['instance']} -- these can start "
                    "concurrently and clobber each other, even though each is a "
                    "valid resume on its own. Trim all but one.\n      "
                    + "\n      ".join(_where(e) for e in undispatched),
                )
            )

        # Same instance, same tag: fine only if every repeat is a --resume of
        # the first occurrence, which is what the retry path emits.
        if len(group) > 1:
            repeats = [e for e in group[1:] if not e["resume"]]
            if repeats:
                findings.append(
                    (
                        "ERROR",
                        "duplicate-within-instance",
                        f"tag {tag!r} appears {len(group)} times on "
                        f"{group[0]['instance']} with a non-resume repeat -- the "
                        "second run will refuse to start against the existing "
                        "runs/ dir and fail\n      "
                        + "\n      ".join(_where(e) for e in group),
                    )
                )


def check_registry(entries, out_dir, findings):
    assignments_path = os.path.join(out_dir, "assignments.json")
    try:
        with open(assignments_path) as f:
            assignments = json.load(f)
    except FileNotFoundError:
        findings.append(
            (
                "ERROR",
                "config",
                f"no assignments.json at {assignments_path} -- registry cross-checks "
                "skipped (pass no --out-dir if that's intended)",
            )
        )
        return
    registry = {a["tag"]: a["instance"] for a in assignments}

    queued_by_tag = {}
    for e in entries:
        queued_by_tag.setdefault(e["tag"], set()).add(e["instance"])

    for tag, instances in sorted(queued_by_tag.items()):
        expected = registry.get(tag)
        if expected is None:
            findings.append(
                (
                    "ERROR",
                    "unregistered",
                    f"tag {tag!r} is queued on {', '.join(sorted(instances))} but has "
                    "no assignments.json row -- it bypassed sweep_pool.py, so nothing "
                    "was preventing another instance from being given it too. "
                    "Register it with `sweep_pool.py requeue`.",
                )
            )
        elif expected not in instances:
            findings.append(
                (
                    "ERROR",
                    "misplaced",
                    f"tag {tag!r} is registered to {expected} but queued on "
                    f"{', '.join(sorted(instances))} -- record moves with "
                    "`sweep_pool.py reassign` so the registry stays authoritative",
                )
            )

    for tag, instance in sorted(registry.items()):
        if tag not in queued_by_tag:
            findings.append(
                (
                    "WARN",
                    "assigned-not-queued",
                    f"tag {tag!r} is assigned to {instance} but is not in its queue -- "
                    "expected if it already ran and the queue was rotated; "
                    "unexpected otherwise",
                )
            )


def cmd_fetch_request(args):
    instances = load_instances(args.instances)
    fetches = {}
    for alias, cfg in sorted(instances.items()):
        paths = [cfg[f] for f in FETCH_FIELDS if cfg.get(f)]
        if paths:
            fetches[alias] = paths
    print(json.dumps(fetches, indent=2))


def cmd_check(args):
    instances = load_instances(args.instances)
    findings = []
    entries = read_queues(
        instances, args.fetch_root, args.max_age_minutes, args.default_tag, findings
    )
    check_duplicates(entries, findings)
    if args.out_dir:
        check_registry(entries, args.out_dir, findings)

    errors = [f for f in findings if f[0] == "ERROR"]
    if args.json:
        print(
            json.dumps(
                {
                    "queued_lines": len(entries),
                    "instances_audited": sorted({e["instance"] for e in entries}),
                    "findings": [
                        {"level": lv, "kind": k, "message": m} for lv, k, m in findings
                    ],
                },
                indent=2,
            )
        )
    else:
        audited = sorted({e["instance"] for e in entries})
        print(
            f"audited {len(entries)} queued line(s) across "
            f"{len(audited)} instance(s): {', '.join(audited) or '(none)'}"
        )
        for level, kind, message in findings:
            print(f"  [{level}] {kind}: {message}")
        if not findings:
            print("  no findings -- no tag is queued on more than one instance")
    return 1 if errors else 0


def main():
    args = parse_args()
    if args.cmd == "fetch-request":
        cmd_fetch_request(args)
        return 0
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
