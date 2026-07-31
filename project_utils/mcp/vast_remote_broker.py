"""MCP stdio server exposing shell access to the vast.ai GPU instance.

Runs as a host process launched from .mcp.json -- outside the sandboxed
Bash-tool process tree -- so it can reach the network and the SSH keys
that a sandboxed agent has no path or syscall to touch. The agent
supplies a command string; the SSH host, key material and remote layout
all stay on this side.

This is a deliberately wide tool: the remote is a stateless GPU executor
holding nothing durable except runs/, so arbitrary command execution on
it is an acceptable capability to hand an agent. It is NOT a security
boundary -- anything reachable from the remote's shell is reachable
through this server. The only guard is an anti-footgun check on commands
that look like they destroy runs/ or the workspace root, which callers
can override per call.

Commands run through a small wrapper that cd's to the remote project dir
and activates the remote venv, because the remote's ~/.bashrc bails out
early for non-interactive sessions and so does neither on its own.

No third-party dependencies (this sandbox has no PyPI access), so the
MCP JSON-RPC handshake (initialize / tools/list / tools/call) is
hand-rolled here instead of using the `mcp` SDK.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import IO, Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vast-remote-broker"
SERVER_VERSION = "0.1.0"

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 3600

# Output cap, split head/tail so both the start of a build log and the
# error that ended it survive truncation.
MAX_OUTPUT_CHARS = 40_000

# Anti-footgun only, not a security boundary: catches the obvious
# `rm -rf runs` typo class, not a determined caller. See PR for rationale.
_DESTRUCTIVE_VERB_RE = re.compile(r"(?<![\w/-])(rm|rmdir|shred|mkfs|dd)(?![\w-])")
_PROTECTED_TARGET_RE = re.compile(r"(?<![\w.-])(runs|/workspace)(?![\w-])")


@dataclass(frozen=True)
class Config:
    ssh_command: list[str]
    ssh_host: str
    workdir: str
    venv: str


class BrokerError(Exception):
    """User-facing error -- safe to return as tool-call error text."""


def load_config_from_env() -> Config:
    ssh_command = shlex.split(os.environ.get("VAST_REMOTE_SSH_COMMAND", "ssh"))
    if not ssh_command:
        raise RuntimeError("vast_remote_broker: VAST_REMOTE_SSH_COMMAND is empty")
    ssh_host = os.environ.get("VAST_REMOTE_SSH_HOST")
    if not ssh_host:
        raise RuntimeError("vast_remote_broker: VAST_REMOTE_SSH_HOST is required")
    return Config(
        ssh_command=ssh_command,
        ssh_host=ssh_host,
        workdir=os.environ.get("VAST_REMOTE_WORKDIR", "/workspace/toy_probe_hiding"),
        venv=os.environ.get("VAST_REMOTE_VENV", "/workspace/.venv"),
    )


def _check_destructive(command: str, confirmed: bool) -> None:
    if confirmed:
        return
    if _DESTRUCTIVE_VERB_RE.search(command) and _PROTECTED_TARGET_RE.search(command):
        raise BrokerError(
            "command looks like it destroys runs/ or the workspace root, which hold "
            "the only state on the remote worth keeping. Re-send with "
            "confirm_destructive=true if that is genuinely intended."
        )


def _remote_script(
    config: Config, command: str, cwd: str | None, use_venv: bool
) -> str:
    """Wrap the caller's command with the remote setup it can't assume."""
    lines = ["set -o pipefail"]
    target_dir = cwd if cwd is not None else config.workdir
    lines.append(f"cd {shlex.quote(target_dir)} || exit 1")
    if use_venv:
        activate = shlex.quote(f"{config.venv}/bin/activate")
        lines.append(f"if [ -f {activate} ]; then . {activate}; fi")
    lines.append(command)
    return "\n".join(lines)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... [{omitted} chars omitted] ...\n{text[-half:]}"


def _section(name: str, text: str) -> str:
    text = text.rstrip()
    return f"--- {name} ---\n{_truncate(text)}" if text else f"--- {name} --- (empty)"


def remote_exec(config: Config, arguments: dict[str, Any]) -> str:
    command = arguments.get("command")
    cwd = arguments.get("cwd")
    timeout_s = arguments.get("timeout_s", DEFAULT_TIMEOUT_S)
    use_venv = arguments.get("use_venv", True)
    confirm_destructive = arguments.get("confirm_destructive", False)

    if not isinstance(command, str) or not command.strip():
        raise BrokerError("command is required and must be a non-empty string")
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        raise BrokerError("cwd must be a non-empty string if given")
    if not isinstance(use_venv, bool):
        raise BrokerError("use_venv must be a boolean if given")
    if not isinstance(confirm_destructive, bool):
        raise BrokerError("confirm_destructive must be a boolean if given")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise BrokerError("timeout_s must be a number if given")
    if not 1 <= timeout_s <= MAX_TIMEOUT_S:
        raise BrokerError(f"timeout_s must be between 1 and {MAX_TIMEOUT_S}")

    _check_destructive(command, confirm_destructive)

    script = _remote_script(config, command, cwd, use_venv)
    argv = [
        *config.ssh_command,
        "-o",
        "BatchMode=yes",
        config.ssh_host,
        # One shlex.quote survives exactly one round of remote shell parsing,
        # which is what `ssh host bash -c <script>` performs.
        f"bash -c {shlex.quote(script)}",
    ]

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            # Never inherit the server's stdin: that pipe carries JSON-RPC, and
            # a remote command reading from it would eat the protocol stream.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise BrokerError(
            f"command timed out after {timeout_s}s and the ssh connection was closed. "
            "The remote process may still be running; re-connect to check. For work "
            "that outlives one call, launch it detached (see the tool description)."
        )

    status = (
        "exit code: 0 (success)"
        if result.returncode == 0
        else f"exit code: {result.returncode}"
    )
    return "\n".join(
        [
            f"{config.ssh_host}:{cwd or config.workdir}$ {command}",
            status,
            _section("stdout", result.stdout),
            _section("stderr", result.stderr),
        ]
    )


TOOLS = [
    {
        "name": "remote_exec",
        "description": (
            "Run a shell command on the remote vast.ai GPU instance over SSH and return "
            "its exit code, stdout and stderr. The command runs under bash in the remote "
            "project directory with the remote venv activated, so `python train_*.py ...` "
            "and `nvidia-smi` work directly. Arbitrary commands are allowed.\n"
            "stdin is closed, so nothing interactive will work. The call blocks until the "
            "command finishes or timeout_s elapses, and a timeout closes the connection "
            "without necessarily killing the remote process -- for anything long-running "
            "(training), launch it detached instead, e.g. "
            "`setsid nohup python train.py > logs/x.log 2>&1 < /dev/null &`, then poll with "
            "further remote_exec calls that tail the log.\n"
            "Output is truncated in the middle if very long; prefer piping through "
            "head/tail/grep on the remote.\n"
            "The remote is a stateless executor: source is synced from this machine and "
            "runs/ syncs back, so remote-side source edits get overwritten and are not "
            "the way to change code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run on the remote (bash syntax; may contain pipes, &&, etc).",
                },
                "cwd": {
                    "type": "string",
                    "description": "Absolute remote directory to run in. Defaults to the remote project directory.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": f"Seconds to wait before giving up (default {DEFAULT_TIMEOUT_S}, max {MAX_TIMEOUT_S}).",
                },
                "use_venv": {
                    "type": "boolean",
                    "description": "Activate the remote venv first (default true). Set false to use the system Python.",
                },
                "confirm_destructive": {
                    "type": "boolean",
                    "description": (
                        "Set true to allow a command that appears to delete runs/ or the "
                        "workspace root. Such commands are refused by default because runs/ "
                        "holds the only remote state worth keeping."
                    ),
                },
            },
            "required": ["command"],
        },
    },
]

TOOL_HANDLERS = {"remote_exec": remote_exec}


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
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"unknown tool {name!r}"}],
                    "isError": True,
                },
            }
        try:
            text = handler(config, arguments)
            is_error = False
        except BrokerError as exc:
            text = str(exc)
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
