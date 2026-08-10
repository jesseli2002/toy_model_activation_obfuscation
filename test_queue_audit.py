"""Tests for the cross-instance tag-collision guard.

Two halves, tested together because they are two halves of one guarantee:
sweep_pool.py's registry is what makes a tag collision impossible while every
queue mutation goes through it, and queue_audit.py is what catches the ones
that got onto a remote queue some other way.

Both are driven as subprocesses, the way an agent calls them -- the exit
status is part of the contract (a monitor gates on it), not an afterthought.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent / ".claude/skills/parallel-remote-training/scripts"
POOL = SCRIPTS / "sweep_pool.py"
AUDIT = SCRIPTS / "queue_audit.py"

# Both instances deliberately use identical remote paths: that is what a real
# pool looks like, and it is exactly the case a per-instance local layout has
# to keep separate.
REMOTE_QUEUE = "/home/agent/sweep_scratch/queue.txt"
REMOTE_IDX = "/home/agent/sweep_scratch/queue.txt.launched_idx"
INSTANCES = {
    "vtao": {"queue": REMOTE_QUEUE, "launched_idx": REMOTE_IDX},
    "vtao1": {"queue": REMOTE_QUEUE, "launched_idx": REMOTE_IDX},
}


class Sweep:
    """A pool directory plus the local copies a fetch_files call would leave."""

    def __init__(self, root):
        self.out_dir = root / "pool"
        self.fetch_root = root / "fetchroot"
        self.instances_json = root / "instances.json"
        self.instances_json.write_text(json.dumps(INSTANCES))

    def pool(self, *args, expect_rc=0):
        result = subprocess.run(
            [sys.executable, str(POOL), *args, "--out-dir", str(self.out_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == expect_rc, result.stdout + result.stderr
        return result

    def build(self, tags):
        manifest = self.out_dir.parent / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                [
                    {
                        "tag": t,
                        "command": f"python train_adversarial_logreg.py --tag {t}",
                    }
                    for t in tags
                ]
            )
        )
        self.pool("build", "--manifest", str(manifest))

    def place(self, instance, field, content):
        """Stand in for what fetch_files would have written locally."""
        dest = self.fetch_root / instance / INSTANCES[instance][field].lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        return dest

    def queue_of(self, instance):
        return (self.out_dir / f"queue_{instance}.txt").read_text()

    def delta_of(self, instance):
        return (self.out_dir / f"queue_{instance}.delta.txt").read_text()

    def audit(self, *extra, with_registry=True, expect_rc=0):
        argv = [
            sys.executable,
            str(AUDIT),
            "check",
            "--instances",
            str(self.instances_json),
            "--fetch-root",
            str(self.fetch_root),
            *extra,
        ]
        if with_registry:
            argv += ["--out-dir", str(self.out_dir)]
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == expect_rc, result.stdout + result.stderr
        return result.stdout


@pytest.fixture
def sweep(tmp_path):
    """Four runs, split two-and-two across two instances, nothing dispatched."""
    s = Sweep(tmp_path)
    s.build(["t0", "t1", "t2", "t3"])
    s.pool("assign", "--instance", "vtao", "--n-runs", "2")
    s.pool("assign", "--instance", "vtao1", "--n-runs", "2")
    s.place("vtao", "queue", s.queue_of("vtao"))
    s.place("vtao1", "queue", s.queue_of("vtao1"))
    s.place("vtao", "launched_idx", "0\n")
    s.place("vtao1", "launched_idx", "0\n")
    return s


def test_a_properly_assigned_sweep_audits_clean(sweep):
    assert "no findings" in sweep.audit()


def test_fetch_request_names_every_file_the_audit_needs(sweep, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "fetch-request",
            "--instances",
            str(sweep.instances_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    request = json.loads(result.stdout)
    assert set(request) == {"vtao", "vtao1"}
    assert REMOTE_QUEUE in request["vtao"] and REMOTE_IDX in request["vtao"]


def test_undispatched_duplicate_is_flagged_as_still_trimmable(sweep):
    sweep.place(
        "vtao1",
        "queue",
        sweep.queue_of("vtao1") + "python train_adversarial_logreg.py --tag t0\n",
    )
    out = sweep.audit(expect_rc=1)

    assert "duplicate-across-instances" in out
    assert "queue_trim.sh" in out


def test_dispatched_duplicate_is_escalated_rather_than_called_fixable(sweep):
    # The distinction that matters: past launched_idx the two runs are already
    # writing into one runs/<tag>, and trimming cannot undo that.
    sweep.place(
        "vtao1",
        "queue",
        sweep.queue_of("vtao1") + "python train_adversarial_logreg.py --tag t0\n",
    )
    sweep.place("vtao1", "launched_idx", "3\n")
    out = sweep.audit(expect_rc=1)

    assert "clobbering each other now" in out
    assert "queue_trim.sh" not in out


def test_tag_queued_only_on_a_non_owning_instance_is_misplaced(sweep):
    sweep.place("vtao", "queue", "".join(sweep.queue_of("vtao").splitlines(True)[1:]))
    sweep.place(
        "vtao1",
        "queue",
        sweep.queue_of("vtao1") + "python train_adversarial_logreg.py --tag t0\n",
    )
    assert "misplaced" in sweep.audit(expect_rc=1)


def test_queued_tag_with_no_registry_row_is_flagged(sweep):
    # The historical bypass: a retry appended straight through queue_append.sh.
    sweep.place(
        "vtao",
        "queue",
        sweep.queue_of("vtao") + "python train_adversarial_logreg.py --tag t0_retry1\n",
    )
    assert "unregistered" in sweep.audit(expect_rc=1)


def test_a_registered_retry_audits_clean(sweep):
    sweep.pool("requeue", "--instance", "vtao", "--retry-tag", "t0")
    delta = sweep.delta_of("vtao")
    assert "--tag t0_retry1" in delta

    sweep.place("vtao", "queue", sweep.queue_of("vtao"))
    assert "no findings" in sweep.audit()


def test_retry_tags_increment_rather_than_colliding(sweep):
    sweep.pool("requeue", "--instance", "vtao", "--retry-tag", "t0")
    sweep.pool("requeue", "--instance", "vtao", "--retry-tag", "t0")
    tags = [
        a["tag"] for a in json.loads((sweep.out_dir / "assignments.json").read_text())
    ]

    assert "t0_retry1" in tags and "t0_retry2" in tags


def test_resume_is_refused_on_an_instance_without_the_checkpoint(sweep):
    before = (sweep.out_dir / "assignments.json").read_text()
    result = sweep.pool(
        "requeue", "--instance", "vtao1", "--resume-tag", "t0", expect_rc=1
    )

    assert "checkpoint" in result.stderr
    assert (sweep.out_dir / "assignments.json").read_text() == before


def test_resume_on_the_owning_instance_adds_no_duplicate_row(sweep):
    before = json.loads((sweep.out_dir / "assignments.json").read_text())
    sweep.pool("requeue", "--instance", "vtao", "--resume-tag", "t0")
    after = json.loads((sweep.out_dir / "assignments.json").read_text())

    assert before == after
    assert sweep.delta_of("vtao").strip().endswith("--resume")


def test_requeue_refuses_a_tag_that_was_never_registered(sweep):
    result = sweep.pool(
        "requeue", "--instance", "vtao", "--retry-tag", "ghost", expect_rc=1
    )
    assert "no assignment row" in result.stderr


def test_a_resume_repeat_in_a_queue_is_not_a_duplicate(sweep):
    # The retry path legitimately puts a tag on its instance's queue twice.
    first = sweep.queue_of("vtao").splitlines()[0]
    sweep.place("vtao", "queue", sweep.queue_of("vtao") + first + " --resume\n")
    assert "no findings" in sweep.audit()


def test_a_bare_repeat_in_a_queue_is_a_duplicate(sweep):
    first = sweep.queue_of("vtao").splitlines()[0]
    sweep.place("vtao", "queue", sweep.queue_of("vtao") + first + "\n")
    assert "duplicate-within-instance" in sweep.audit(expect_rc=1)


def test_a_missing_local_copy_fails_rather_than_auditing_an_empty_queue(sweep):
    (sweep.fetch_root / "vtao1" / REMOTE_QUEUE.lstrip("/")).unlink()
    out = sweep.audit(expect_rc=1)

    assert "missing-fetch" in out
    assert "NOT audited" in out


def test_a_stale_local_copy_is_reported(sweep):
    import os

    os.utime(sweep.fetch_root / "vtao1" / REMOTE_QUEUE.lstrip("/"), (0, 0))
    assert "stale-fetch" in sweep.audit("--max-age-minutes", "30")


def test_commands_without_an_explicit_tag_still_collide(sweep):
    # A command omitting --tag runs under the train script's default tag, so
    # two of them on different boxes are a real collision, not an unparsed line.
    sweep.place("vtao", "queue", "python train_adversarial_logreg.py\n")
    sweep.place("vtao1", "queue", "python train_adversarial_logreg.py\n")
    out = sweep.audit(with_registry=False, expect_rc=1)

    assert "duplicate-across-instances" in out
    assert "adv-logreg" in out
