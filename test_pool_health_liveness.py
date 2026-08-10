"""Verdict-logic tests for pool_health.py's run-liveness check.

The interesting cases are the ones a live healthy sweep can't produce: a
genuinely dead box, a dead sync loop, and the long-quiet-but-healthy run
that made the old time-since-last-log-line flag unusable.
"""

import json
import os
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), ".claude/skills/parallel-remote-training/scripts"
    ),
)
from pool_health import compute_health  # noqa: E402

QUEUE = "python train_adversarial_logreg.py --tag {tag} --max-iters 100\n"


def _fixture(tmp_path, *, log_lines, tags, history_ages, sync_age=5.0, log_age=0.0):
    """Build a manager.log/queue.txt/runs tree with controlled mtimes."""
    now = time.time()
    log = tmp_path / "manager.log"
    log.write_text("".join(log_lines))
    os.utime(log, (now - log_age, now - log_age))

    queue = tmp_path / "queue.txt"
    queue.write_text("".join(QUEUE.format(tag=t) for t in tags))

    runs = tmp_path / "runs"
    runs.mkdir()
    for tag, age in history_ages.items():
        d = runs / tag / "logs"
        d.mkdir(parents=True)
        h = d / "history.jsonl"
        h.write_text(json.dumps({"iter": 100}) + "\n")
        os.utime(h, (now - age, now - age))

    state = tmp_path / "sync_state"
    state.mkdir()
    if sync_age is not None:
        (state / "box.last_sync").write_text(str(now - sync_age))

    return dict(
        manager_log=str(log),
        queue=str(queue),
        runs_dir=str(runs),
        sync_host="box",
        sync_state_dir=str(state),
    )


def _stamp(ago):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - ago))


def test_long_quiet_run_is_not_stalled(tmp_path):
    """The original false positive: hours of manager silence, run advancing."""
    kw = _fixture(
        tmp_path,
        log_lines=[f"{_stamp(7200)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={"t0": 30.0},
    )
    out = compute_health(**kw)
    assert out["manager_quiet"] is True
    assert out["status"] == "ok"
    assert out["running_jobs"][0]["state"] == "advancing"


def test_quiet_manager_and_frozen_run_with_healthy_sync_is_stalled(tmp_path):
    kw = _fixture(
        tmp_path,
        log_lines=[f"{_stamp(7200)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={"t0": 7000.0},
        sync_age=5.0,
    )
    out = compute_health(**kw)
    assert out["status"] == "stalled"
    assert "not making progress" in out["diagnosis"]


def test_dead_sync_loop_is_unknown_not_stalled(tmp_path):
    """Frozen mtimes are ambiguous when the puller itself stopped."""
    kw = _fixture(
        tmp_path,
        log_lines=[f"{_stamp(7200)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={"t0": 7000.0},
        sync_age=6000.0,
    )
    out = compute_health(**kw)
    assert out["status"] == "unknown"
    assert "runs-pull loop is down" in out["diagnosis"]


def test_missing_sync_heartbeat_is_unknown(tmp_path):
    kw = _fixture(
        tmp_path,
        log_lines=[f"{_stamp(7200)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={"t0": 7000.0},
        sync_age=None,
    )
    out = compute_health(**kw)
    assert out["status"] == "unknown"
    assert out["sync_heartbeat_age_seconds"] is None


def test_one_advancing_job_vouches_for_the_box(tmp_path):
    """Per-job staleness is expected at the tail of a run; only an all-stale
    box is a stall."""
    kw = _fixture(
        tmp_path,
        log_lines=[
            f"{_stamp(7200)} launched pid=1 idx=1/2: cmd...\n",
            f"{_stamp(7200)} launched pid=2 idx=2/2: cmd...\n",
        ],
        tags=["t0", "t1"],
        history_ages={"t0": 7000.0, "t1": 20.0},
    )
    out = compute_health(**kw)
    assert out["status"] == "ok"
    assert {j["state"] for j in out["running_jobs"]} == {"stale", "advancing"}


def test_quiet_manager_with_no_running_jobs_is_stalled(tmp_path):
    """Not even idle-logging, nothing running: the dispatcher itself is gone."""
    kw = _fixture(
        tmp_path,
        log_lines=[
            f"{_stamp(7200)} launched pid=1 idx=1/1: cmd...\n",
            f"{_stamp(7100)} finished pid=1 job1 rc=0\n",
        ],
        tags=["t0"],
        history_ages={"t0": 7000.0},
    )
    out = compute_health(**kw)
    assert out["status"] == "stalled"
    assert "likely died" in out["diagnosis"]


def test_startup_grace_covers_a_job_with_no_history_yet(tmp_path):
    kw = _fixture(
        tmp_path,
        log_lines=[f"{_stamp(60)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={},
    )
    out = compute_health(**kw)
    assert out["running_jobs"][0]["state"] == "starting"
    assert out["status"] == "ok"


def test_no_history_long_after_launch_is_never_started(tmp_path):
    kw = _fixture(
        tmp_path,
        log_lines=[f"{_stamp(7200)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={},
    )
    out = compute_health(**kw)
    assert out["running_jobs"][0]["state"] == "never-started"
    assert out["status"] == "stalled"


def test_without_runs_dir_a_quiet_manager_is_unknown_never_stalled(tmp_path):
    """The key regression guard: no liveness input must not resurrect the
    old false positive."""
    kw = _fixture(
        tmp_path,
        log_lines=[f"{_stamp(7200)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={"t0": 7000.0},
    )
    kw.pop("queue")
    out = compute_health(**kw)
    assert out["status"] == "unknown"
    assert out["running_jobs"] == []


def test_stale_snapshot_cannot_produce_a_stall(tmp_path):
    """A fetched copy's age must not be charged to the manager as silence."""
    kw = _fixture(
        tmp_path,
        # last event 60s before the fetch, but the fetch was 40min ago
        log_lines=[f"{_stamp(2460)} launched pid=1 idx=1/1: cmd...\n"],
        tags=["t0"],
        history_ages={"t0": 7000.0},
        log_age=2400.0,
    )
    out = compute_health(**kw)
    assert out["last_event_age_seconds"] < 120  # not ~2460
    assert out["manager_quiet"] is False
    assert out["status"] != "stalled"
    assert "re-fetch" in out["diagnosis"]
