"""MCP stdio server exposing a single, narrow `git_push` tool.

Runs as a host process launched from .mcp.json -- outside the sandboxed
Bash-tool process tree entirely -- so it is the one place a GitHub PAT
can live where a sandboxed agent has no path or syscall to reach it.
The agent supplies only a local worktree path and a branch name; it
never sees the token, the remote URL, or raw git argv.

The actual credential handoff to git happens through an inline
`credential.helper` shell function that reads GIT_PUSH_BROKER_TOKEN
from this process's own environment. Git talks to credential helpers
over its own private pipe -- never the parent's stdout/stderr -- so
the token cannot leak into a tool result even on auth failure. This
module's own Python code never reads the token's value at all, only
checks that the env var is present; only the one-line helper subshell
ever touches it.

No third-party dependencies (this sandbox has no PyPI access), so the
MCP JSON-RPC handshake (initialize / tools/list / tools/call) is
hand-rolled here instead of using the `mcp` SDK.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import IO, Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "git-push-broker"
SERVER_VERSION = "0.1.0"

# Conservative allow-list for branch names: no leading '-' (flag
# injection), no '..' (git ref traversal), no whitespace/control
# chars, no trailing '/'. Deliberately stricter than what git itself
# permits.
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_FORBIDDEN_BRANCH_SUBSTRINGS = ("..", "//", "@{")
_FORBIDDEN_BRANCHES = {"main", "master", "HEAD"}

GIT_TOOL_TIMEOUT_S = 30


@dataclass(frozen=True)
class Config:
    token_env_var: str
    remote_url: str
    repo_root: str


class BrokerError(Exception):
    """User-facing error -- safe to return as tool-call error text."""


def load_config_from_env() -> Config:
    missing = [
        name
        for name in (
            "GIT_PUSH_BROKER_TOKEN",
            "GIT_PUSH_BROKER_REMOTE",
            "GIT_PUSH_BROKER_REPO_ROOT",
        )
        if name not in os.environ
    ]
    if missing:
        raise RuntimeError(
            f"git_push_broker: missing required env var(s): {', '.join(missing)}"
        )
    return Config(
        token_env_var="GIT_PUSH_BROKER_TOKEN",
        remote_url=os.environ["GIT_PUSH_BROKER_REMOTE"],
        repo_root=os.path.realpath(os.environ["GIT_PUSH_BROKER_REPO_ROOT"]),
    )


def _validate_branch(branch: str) -> None:
    if branch in _FORBIDDEN_BRANCHES:
        raise BrokerError(f"refusing to push to protected branch {branch!r}")
    if not _BRANCH_RE.match(branch):
        raise BrokerError(f"branch name {branch!r} does not match the allowed pattern")
    if any(bad in branch for bad in _FORBIDDEN_BRANCH_SUBSTRINGS):
        raise BrokerError(f"branch name {branch!r} contains a forbidden substring")


def _validate_worktree_path(worktree_path: str, repo_root: str) -> str:
    resolved = os.path.realpath(worktree_path)
    if resolved != repo_root and not resolved.startswith(repo_root + os.sep):
        raise BrokerError(
            f"worktree_path {worktree_path!r} resolves outside the allowed repo root {repo_root!r}"
        )
    if not os.path.isdir(resolved):
        raise BrokerError(f"worktree_path {worktree_path!r} is not a directory")
    check = subprocess.run(
        ["git", "-C", resolved, "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        timeout=GIT_TOOL_TIMEOUT_S,
    )
    if check.returncode != 0 or check.stdout.strip() != "true":
        raise BrokerError(f"worktree_path {worktree_path!r} is not a git working tree")
    return resolved


def _credential_helper_arg() -> str:
    # Token value is never interpolated by Python -- the literal string
    # "$GIT_PUSH_BROKER_TOKEN" is what lands in argv; the subshell git
    # spawns for the credential helper resolves it from its inherited
    # environment at call time, off of any pipe this process reads.
    return "!f() { echo username=x-access-token; echo password=$GIT_PUSH_BROKER_TOKEN; }; f"


def _run_git(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=GIT_TOOL_TIMEOUT_S,
        env=env,
    )


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def git_push(config: Config, arguments: dict[str, Any]) -> str:
    worktree_path = arguments.get("worktree_path")
    branch = arguments.get("branch")
    expected_remote_sha = arguments.get("expected_remote_sha")

    if not isinstance(worktree_path, str) or not worktree_path:
        raise BrokerError("worktree_path is required and must be a non-empty string")
    if not isinstance(branch, str) or not branch:
        raise BrokerError("branch is required and must be a non-empty string")
    if expected_remote_sha is not None and not isinstance(expected_remote_sha, str):
        raise BrokerError("expected_remote_sha must be a string if given")

    _validate_branch(branch)
    resolved_worktree = _validate_worktree_path(worktree_path, config.repo_root)

    env = _git_env()
    cred_helper = _credential_helper_arg()

    local_ref = subprocess.run(
        [
            "git",
            "-C",
            resolved_worktree,
            "rev-parse",
            "--verify",
            f"refs/heads/{branch}",
        ],
        capture_output=True,
        text=True,
        timeout=GIT_TOOL_TIMEOUT_S,
    )
    if local_ref.returncode != 0:
        raise BrokerError(
            f"local branch {branch!r} not found in {worktree_path!r}: {local_ref.stderr.strip()}"
        )
    local_sha = local_ref.stdout.strip()

    ls_remote = _run_git(
        [
            "git",
            "-c",
            f"credential.helper={cred_helper}",
            "ls-remote",
            config.remote_url,
            f"refs/heads/{branch}",
        ],
        env=env,
    )
    if ls_remote.returncode != 0:
        raise BrokerError(
            f"failed to query remote tip for {branch!r}: {ls_remote.stderr.strip()}"
        )
    remote_sha_before = (
        ls_remote.stdout.split()[0] if ls_remote.stdout.strip() else None
    )

    if expected_remote_sha is not None and remote_sha_before != expected_remote_sha:
        raise BrokerError(
            f"expected_remote_sha {expected_remote_sha!r} does not match actual remote tip "
            f"{remote_sha_before!r} for {branch!r} -- refusing to push on a stale assumption"
        )

    push = _run_git(
        [
            "git",
            "-C",
            resolved_worktree,
            "-c",
            f"credential.helper={cred_helper}",
            "push",
            config.remote_url,
            f"{local_sha}:refs/heads/{branch}",
        ],
        env=env,
    )
    if push.returncode != 0:
        raise BrokerError(f"git push failed: {push.stderr.strip()}")

    return (
        f"pushed {branch} ({worktree_path}) -> {local_sha}\n"
        f"remote tip before: {remote_sha_before or '(branch did not exist)'}\n"
        f"remote tip after: {local_sha}\n"
        f"{push.stderr.strip()}"
    ).strip()


TOOLS = [
    {
        "name": "git_push",
        "description": (
            "Push a local branch to the configured GitHub remote using a real `git push` "
            "(not an API-synthesized commit, so the pushed SHA matches local history exactly). "
            "Refuses main/master, requires the worktree to be under the configured repo root, "
            "and never force-pushes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "worktree_path": {
                    "type": "string",
                    "description": "Local path to the git working tree containing the commit to push.",
                },
                "branch": {
                    "type": "string",
                    "description": "Branch name to push (same name used locally and on the remote).",
                },
                "expected_remote_sha": {
                    "type": "string",
                    "description": (
                        "Optional: the remote branch tip this push assumes as its base. "
                        "If given and it doesn't match, the push is refused."
                    ),
                },
            },
            "required": ["worktree_path", "branch"],
        },
    }
]


def _write_message(out: IO[str], obj: dict[str, Any]) -> None:
    out.write(json.dumps(obj) + "\n")
    out.flush()


def _handle_request(config: Config, msg: dict[str, Any]) -> dict[str, Any] | None:
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name != "git_push":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"unknown tool {name!r}"}],
                    "isError": True,
                },
            }
        try:
            text = git_push(config, arguments)
            is_error = False
        except BrokerError as exc:
            text = str(exc)
            is_error = True
        except subprocess.TimeoutExpired:
            text = "git command timed out"
            is_error = True
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }

    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return None


def serve(config: Config, in_stream: IO[str], out_stream: IO[str]) -> None:
    for line in in_stream:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle_request(config, msg)
        if response is not None:
            _write_message(out_stream, response)


def main() -> None:
    config = load_config_from_env()
    serve(config, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
