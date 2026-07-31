"""End-to-end tests for the git push broker MCP server.

Drives the server through its JSON-RPC surface against a real local bare
repo standing in for the GitHub remote, so the guards are exercised as a
client would hit them rather than by calling the helpers directly. No
network and no credentials involved: the file:// remote never invokes the
credential helper.
"""

import io
import json
import subprocess

import pytest

from project_utils.mcp import git_push_broker


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A bare 'remote' plus a working clone with `master` and `feature`."""
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "master", str(work)], check=True)
    for key, value in (
        ("GIT_AUTHOR_NAME", "t"),
        ("GIT_AUTHOR_EMAIL", "t@example.com"),
        ("GIT_COMMITTER_NAME", "t"),
        ("GIT_COMMITTER_EMAIL", "t@example.com"),
    ):
        monkeypatch.setenv(key, value)

    def git(*args, cwd=work):
        result = subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True
        )
        assert result.returncode == 0, (args, result.stderr)
        return result.stdout.strip()

    (work / "a").write_text("1")
    git("add", "-A")
    git("commit", "-qm", "c1")
    git("checkout", "-qb", "feature")
    (work / "b").write_text("1")
    git("add", "-A")
    git("commit", "-qm", "c2")

    monkeypatch.setenv("GIT_PUSH_BROKER_TOKEN", "unused")
    monkeypatch.setenv("GIT_PUSH_BROKER_REMOTE", str(remote))
    monkeypatch.setenv("GIT_PUSH_BROKER_REPO_ROOT", str(tmp_path))

    class Repo:
        config = git_push_broker.load_config_from_env()
        work_path = str(work)
        remote_path = remote
        run_git = staticmethod(git)

        @staticmethod
        def call(name, arguments):
            """Returns (is_error, text) from a tools/call round trip."""
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
            out = io.StringIO()
            git_push_broker.serve(
                Repo.config, io.StringIO(json.dumps(request) + "\n"), out
            )
            result = json.loads(out.getvalue())["result"]
            return result.get("isError", False), result["content"][0]["text"]

        @staticmethod
        def remote_sha(branch):
            return git("rev-parse", branch, cwd=remote)

    return Repo


def test_force_tool_declares_requires_user_interaction():
    """The force tool's approval gate is a server-side declaration.

    It must reach the client as a literal JSON `true`; any other value is
    ignored and the tool would silently fall back to normal permissions.
    """
    tools = {tool["name"]: tool for tool in git_push_broker.TOOLS}
    assert set(tools) == set(git_push_broker.TOOL_HANDLERS)

    assert "_meta" not in tools["git_push"]
    assert tools["git_push_force"]["_meta"] == {
        "anthropic/requiresUserInteraction": True
    }
    assert '"anthropic/requiresUserInteraction": true' in json.dumps(
        tools["git_push_force"]
    )


def test_force_tool_requires_expected_remote_sha():
    schema = next(
        tool for tool in git_push_broker.TOOLS if tool["name"] == "git_push_force"
    )["inputSchema"]
    assert "expected_remote_sha" in schema["required"]


def test_push_then_force_push_rewrites_history(repo):
    is_error, text = repo.call(
        "git_push", {"worktree_path": repo.work_path, "branch": "feature"}
    )
    assert not is_error, text
    original = repo.run_git("rev-parse", "feature")
    assert repo.remote_sha("feature") == original

    repo.run_git("commit", "-q", "--amend", "-m", "c2-amended")
    rewritten = repo.run_git("rev-parse", "feature")

    # The non-force tool must not be able to rewrite remote history.
    is_error, text = repo.call(
        "git_push", {"worktree_path": repo.work_path, "branch": "feature"}
    )
    assert is_error and "git push failed" in text
    assert repo.remote_sha("feature") == original

    is_error, text = repo.call(
        "git_push_force",
        {
            "worktree_path": repo.work_path,
            "branch": "feature",
            "expected_remote_sha": original,
        },
    )
    assert not is_error, text
    assert repo.remote_sha("feature") == rewritten
    # The clobbered commit must be recoverable from the tool's own output.
    assert original in text and rewritten in text


def test_force_push_refuses_stale_expected_sha(repo):
    repo.call("git_push", {"worktree_path": repo.work_path, "branch": "feature"})
    original = repo.remote_sha("feature")
    repo.run_git("commit", "-q", "--amend", "-m", "c2-amended")

    is_error, text = repo.call(
        "git_push_force",
        {
            "worktree_path": repo.work_path,
            "branch": "feature",
            "expected_remote_sha": "0" * 40,
        },
    )
    assert is_error and "does not match actual remote tip" in text
    assert repo.remote_sha("feature") == original


@pytest.mark.parametrize(
    "arguments, expected_message",
    [
        ({"branch": "feature"}, "expected_remote_sha is required"),
        (
            {"branch": "feature", "expected_remote_sha": ""},
            "expected_remote_sha is required",
        ),
        (
            {"branch": "master", "expected_remote_sha": "0" * 40},
            "protected branch",
        ),
        (
            {"branch": "--upload-pack=evil", "expected_remote_sha": "0" * 40},
            "does not match the allowed pattern",
        ),
        (
            {"branch": "absent", "expected_remote_sha": "0" * 40},
            "not found",
        ),
    ],
)
def test_force_push_guards(repo, arguments, expected_message):
    is_error, text = repo.call(
        "git_push_force", {"worktree_path": repo.work_path, **arguments}
    )
    assert is_error, text
    assert expected_message in text


def test_force_push_refuses_worktree_outside_repo_root(repo, tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    is_error, text = repo.call(
        "git_push_force",
        {
            "worktree_path": str(outside),
            "branch": "feature",
            "expected_remote_sha": "0" * 40,
        },
    )
    assert is_error and "outside the allowed repo root" in text


def test_force_push_refuses_branch_absent_on_remote(repo):
    repo.run_git("branch", "unpushed")
    is_error, text = repo.call(
        "git_push_force",
        {
            "worktree_path": repo.work_path,
            "branch": "unpushed",
            "expected_remote_sha": "0" * 40,
        },
    )
    assert is_error and "does not exist on the remote" in text


def test_unknown_tool_is_rejected(repo):
    is_error, text = repo.call("git_push_nuke", {})
    assert is_error and "unknown tool" in text
