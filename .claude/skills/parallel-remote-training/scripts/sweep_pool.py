"""Sweep-agnostic pool bookkeeping: build/status/assign/eta/reassign over a
generic manifest of {tag, command, stage?, weight?} dicts. A sweep-specific
script (e.g. generate_sweep8.py) builds that manifest and either imports
this module or shells out to it -- this file has no knowledge of arches,
lambdas, or any other sweep-specific concept.

Design note: this tool only manages *local* bookkeeping
(pending.json/assignments.json under --out-dir) and reads *locally-fetched
copies* of remote manager.log/launched_idx files for the `eta` command --
it cannot itself reach a remote instance (that needs remote_exec, an
agent-only capability). Actually mutating a remote queue.txt is the
caller's job, via queue_append.sh/queue_trim.sh over remote_exec; `assign`
and `reassign` here only prepare the delta file to push and update the
local audit trail once you've pushed it.

Usage:
    python sweep_pool.py build --out-dir OUT --manifest manifest.json
    python sweep_pool.py status --out-dir OUT
    python sweep_pool.py assign --out-dir OUT --instance i0 --n-runs 20
    python sweep_pool.py eta --out-dir OUT --instance-log i0:manager.log:idx.txt [...]
    python sweep_pool.py reassign --out-dir OUT --from-instance i0 --to-instance i1 --lines-file trimmed.txt

Writes (same shapes generate_sweep8.py already produces, so a future
rewire of that script onto this module is a drop-in):
    {out-dir}/manifest.json     -- full run list, written once by `build`
    {out-dir}/pending.json      -- runs not yet assigned; `assign` pops from front
    {out-dir}/assignments.json  -- audit log of {tag, instance, command}
    {out-dir}/queue_{instance}.txt        -- full accumulated history per instance
    {out-dir}/queue_{instance}.delta.txt  -- newest chunk from the last assign/reassign
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_health import compute_health  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser(
        "build", help="(re)initialize pending pool from a manifest file"
    )
    p_build.add_argument("--out-dir", default="sweep_pool_scratch")
    p_build.add_argument(
        "--manifest",
        required=True,
        help="JSON file: list of {tag, command, stage?, weight?}",
    )

    p_status = sub.add_parser(
        "status", help="show remaining pending pool + per-instance assignment counts"
    )
    p_status.add_argument("--out-dir", default="sweep_pool_scratch")

    p_assign = sub.add_parser(
        "assign", help="pop a chunk off the pool for one instance"
    )
    p_assign.add_argument("--out-dir", default="sweep_pool_scratch")
    p_assign.add_argument("--instance", required=True)
    p_assign.add_argument("--n-runs", type=int, required=True)

    p_eta = sub.add_parser("eta", help="estimate hours to drain the whole pending pool")
    p_eta.add_argument("--out-dir", default="sweep_pool_scratch")
    p_eta.add_argument(
        "--instance-log",
        action="append",
        default=[],
        metavar="NAME:MANAGER_LOG[:LAUNCHED_IDX]",
        help="one per instance to include in the aggregate rate; repeatable. "
        "LAUNCHED_IDX is optional.",
    )
    p_eta.add_argument("--window-minutes", type=float, default=60.0)

    p_reassign = sub.add_parser(
        "reassign",
        help="record that some pending-but-unstarted lines moved to a different instance",
    )
    p_reassign.add_argument("--out-dir", default="sweep_pool_scratch")
    p_reassign.add_argument("--from-instance", required=True)
    p_reassign.add_argument("--to-instance", required=True)
    p_reassign.add_argument(
        "--lines-file",
        required=True,
        help="the lines queue_trim.sh removed from --from-instance's remote queue "
        "(captured via remote_exec) -- matched back to tags by exact command string",
    )

    return p.parse_args()


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _dump_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


class _Lock:
    """flock on {out_dir}/.pool.lock, guarding the pending/assignments
    read-modify-write -- same discipline as vast_pool_manager.sh's queue
    lock, for the same reason: two callers (e.g. a setup agent and a
    monitor agent) must not race a read-decide-write against these files."""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, ".pool.lock")

    def __enter__(self):
        import fcntl

        self._f = open(self.path, "w")
        fcntl.flock(self._f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl

        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()


def cmd_build(args):
    with open(args.manifest) as f:
        runs = json.load(f)
    for r in runs:
        if "tag" not in r or "command" not in r:
            raise SystemExit(f"[error] manifest entry missing tag/command: {r}")
    runs = sorted(runs, key=lambda r: (r.get("stage", 0), -r.get("weight", 0.0)))

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    pending_path = os.path.join(args.out_dir, "pending.json")
    if os.path.exists(manifest_path) or os.path.exists(pending_path):
        raise SystemExit(
            f"[error] {args.out_dir} already has a manifest/pending pool -- "
            "delete it first if you really want to rebuild (discards assignment progress)."
        )
    _dump_json(manifest_path, runs)
    _dump_json(pending_path, runs)
    _dump_json(os.path.join(args.out_dir, "assignments.json"), [])
    print(f"{len(runs)} runs written to {args.out_dir}/")


def cmd_status(args):
    pending = _load_json(os.path.join(args.out_dir, "pending.json"), [])
    assignments = _load_json(os.path.join(args.out_dir, "assignments.json"), [])
    if not pending and not assignments:
        print(f"no pool found at {args.out_dir}/ -- run `build` first")
        return

    print(f"pending: {len(pending)} runs")
    by_instance = {}
    for a in assignments:
        by_instance.setdefault(a["instance"], []).append(a)
    if by_instance:
        print("assigned:")
        for inst, rs in sorted(by_instance.items()):
            print(f"  {inst}: {len(rs)} runs")


def cmd_assign(args):
    pending_path = os.path.join(args.out_dir, "pending.json")
    assignments_path = os.path.join(args.out_dir, "assignments.json")

    with _Lock(args.out_dir):
        pending = _load_json(pending_path, None)
        if pending is None:
            raise SystemExit(
                f"[error] no pending pool at {pending_path} -- run `build` first"
            )
        assignments = _load_json(assignments_path, [])

        chunk, pending = pending[: args.n_runs], pending[args.n_runs :]
        if not chunk:
            print("pending pool is empty -- nothing to assign")
            return

        chunk_lines = [run["command"] + "\n" for run in chunk]
        queue_path = os.path.join(args.out_dir, f"queue_{args.instance}.txt")
        with open(queue_path, "a") as f:
            f.writelines(chunk_lines)
        delta_path = os.path.join(args.out_dir, f"queue_{args.instance}.delta.txt")
        with open(delta_path, "w") as f:
            f.writelines(chunk_lines)

        for run in chunk:
            assignments.append(
                {
                    "tag": run["tag"],
                    "instance": args.instance,
                    "command": run["command"],
                }
            )

        _dump_json(pending_path, pending)
        _dump_json(assignments_path, assignments)

    print(f"assigned {len(chunk)} runs to {args.instance}")
    print(f"  full history -> {queue_path}")
    print(
        f"  new chunk (push this via queue_append.sh over remote_exec) -> {delta_path}"
    )
    print(f"pending pool: {len(pending)} runs remain")


def cmd_eta(args):
    pending = _load_json(os.path.join(args.out_dir, "pending.json"), [])
    if not pending:
        print("pending pool is empty -- nothing left to estimate")
        return

    total_rate = 0.0
    have_rate = False
    for spec in args.instance_log:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(
                f"[error] --instance-log must be NAME:MANAGER_LOG[:LAUNCHED_IDX], got: {spec}"
            )
        name, manager_log = parts[0], parts[1]
        launched_idx = parts[2] if len(parts) == 3 else None
        health = compute_health(
            manager_log, launched_idx=launched_idx, window_minutes=args.window_minutes
        )
        rate = health["jobs_per_hour_trailing"]
        print(
            f"  {name}: {rate:.2f} jobs/hr"
            if rate is not None
            else f"  {name}: no recent data"
        )
        if rate is not None:
            total_rate += rate
            have_rate = True

    if not have_rate or total_rate <= 0:
        print(
            f"pending: {len(pending)} runs -- ETA unknown (no instance has recent throughput data)"
        )
        return

    # ETA is pending / aggregate throughput, NOT avg-duration * pending --
    # the latter ignores concurrency (100 pending * 2h/run is 50h wall-clock
    # at conc=4, not 200h). See plan doc for why this was corrected.
    eta_hours = len(pending) / total_rate
    print(f"pending: {len(pending)} runs, aggregate rate {total_rate:.2f} jobs/hr")
    print(f"ETA: {eta_hours:.1f} hours ({eta_hours / 24:.1f} days)")


def cmd_reassign(args):
    with open(args.lines_file) as f:
        moved_commands = {line.rstrip("\n") for line in f if line.strip()}
    if not moved_commands:
        print("lines file was empty -- nothing to reassign")
        return

    assignments_path = os.path.join(args.out_dir, "assignments.json")
    with _Lock(args.out_dir):
        assignments = _load_json(assignments_path, [])
        moved = 0
        for a in assignments:
            if a["instance"] == args.from_instance and a["command"] in moved_commands:
                a["instance"] = args.to_instance
                moved += 1
        _dump_json(assignments_path, assignments)

        moved_lines = [c + "\n" for c in sorted(moved_commands)]
        delta_path = os.path.join(args.out_dir, f"queue_{args.to_instance}.delta.txt")
        with open(delta_path, "w") as f:
            f.writelines(moved_lines)
        queue_path = os.path.join(args.out_dir, f"queue_{args.to_instance}.txt")
        with open(queue_path, "a") as f:
            f.writelines(moved_lines)

    print(f"reassigned {moved} tags from {args.from_instance} to {args.to_instance}")
    print(f"  push this via queue_append.sh over remote_exec -> {delta_path}")
    if moved != len(moved_commands):
        print(
            f"  [warn] {len(moved_commands) - moved} lines from --lines-file did not match "
            f"any assignment currently on {args.from_instance} -- check for stale bookkeeping"
        )


def main():
    args = parse_args()
    {
        "build": cmd_build,
        "status": cmd_status,
        "assign": cmd_assign,
        "eta": cmd_eta,
        "reassign": cmd_reassign,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
