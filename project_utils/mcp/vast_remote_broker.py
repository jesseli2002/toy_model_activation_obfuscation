"""MCP stdio server exposing shell access to one or more vast.ai GPU instances.

Also exposes the local source<->remote sync as a tool. It runs on this
side but reaches the remote over the network, so a sandboxed agent
cannot flush it directly and would otherwise have to guess whether its
local source edits had landed before launching a run.

fetch_files exists for the same reason in the opposite direction: only
`runs/` flows remote -> local, so operational files that live outside it
(a pool manager's queue.txt/manager.log in the remote home dir) have no
transport back. remote_exec can print them, but that routes the bytes
through the agent's context and leaves nothing on disk for the local
analysis tools to read -- a redirection in a remote_exec command string
writes on the *remote*, which is an easy thing to get backwards.

Multi-instance: callers pass an `instance` argument (the base SSH alias a
rental was created under, e.g. "vtao" or "vtao1" -- see vast_setup/
create_instance.py's --index) rather than the server being pinned to one
host at startup. remote_exec connects to "<instance>-agent" (the locked-down
account); sync_flush passes `--host <instance>` through to sync_vastai.py.
Nothing here queries vast.ai to resolve that alias -- ssh's own config
Include (~/.ssh/toy_probe_hiding_vastai/*.sshconfig) already does that
translation, so an unknown instance just fails as a normal ssh error. Use
list_instances for live status (which rentals exist, running/stopped) --
that's the only tool here that calls out to `vastai`.

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

Each call reuses one multiplexed SSH connection (ControlMaster/
ControlPersist) instead of renegotiating a fresh TCP+auth session per
call. If the shared connection is stale (e.g. the instance was
restarted), ssh exits 255 for connection-level failures; we tear down
the dead control socket and retry once on a fresh connection.

No third-party dependencies (this sandbox has no PyPI access), so the
MCP JSON-RPC handshake (initialize / tools/list / tools/call) is
hand-rolled here instead of using the `mcp` SDK.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from typing import IO, Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vast-remote-broker"
SERVER_VERSION = "0.1.0"

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 3600

# Flushing blocks until both sessions reach a steady state, which after a
# large edit (or a first sync) takes appreciably longer than a shell call.
DEFAULT_SYNC_TIMEOUT_S = 300

# The sync script lives in a directory that is deliberately excluded from
# this repo's git history, so it is absent from worktree checkouts; the
# server is launched from the main checkout, where this resolves.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_SYNC_SCRIPT = os.path.join(_REPO_ROOT, "vast_setup", "sync_vastai.py")

# Output cap, split head/tail so both the start of a build log and the
# error that ended it survive truncation.
MAX_OUTPUT_CHARS = 40_000

# Where fetch_files lands remote files. Deliberately NOT caller-supplied:
# this server runs outside the agent's sandbox, so an agent-chosen
# destination would be a write primitive anywhere on the host filesystem.
# Fixed per-server instead, and readable from inside the sandbox.
DEFAULT_FETCH_DIR = "/tmp/vast-remote-broker-fetch"
# Refused rather than truncated -- a half-file that still parses is worse
# than a clear error, and nothing this is meant for (queues, manager logs)
# comes close to this.
MAX_FETCH_BYTES = 64 * 1024 * 1024

# Anti-footgun only, not a security boundary: catches the obvious
# `rm -rf runs` typo class, not a determined caller. See PR for rationale.
_DESTRUCTIVE_VERB_RE = re.compile(r"(?<![\w/-])(rm|rmdir|shred|mkfs|dd)(?![\w-])")
_PROTECTED_TARGET_RE = re.compile(r"(?<![\w.-])(runs|/workspace)(?![\w-])")
# Runs tagged debug_* are declared-disposable scratch, excluded from every sync
# and backup, so an agent may delete them without asking. Matches a whole path
# component only, so nothing that walks back out of the run dir slips through.
# "/" is allowed immediately before "runs/" (unlike other path chars) so this
# still matches when "runs/debug_x" is reached via an absolute path.
_DISPOSABLE_RUN_RE = re.compile(r"(?<![\w.-])runs/debug_[\w.*?-]+/?(?![\w./-])")

# An instance name becomes part of an ssh argv entry (as "<instance>-agent")
# and a sync_vastai.py --host value, never shell-interpolated -- but a
# leading '-' would still be read as a flag by ssh/argparse, so restrict to
# what create_instance.py actually generates (vtao, vtao1, vtao2, ...).
_VALID_INSTANCE_RE = re.compile(r"^[A-Za-z0-9._]+$")
# vast.ai labels this project's instances "vtao-<index>" (see
# create_instance.py); list_instances filters on this by default.
DEFAULT_LABEL_PREFIX = "vtao-"


@dataclass(frozen=True)
class Config:
    ssh_command: list[str]
    default_instance: str
    workdir: str
    venv: str
    control_path: str
    control_persist: str
    sync_script: str
    vastai_command: list[str]
    fetch_dir: str


class BrokerError(Exception):
    """User-facing error -- safe to return as tool-call error text."""


def load_config_from_env() -> Config:
    ssh_command = shlex.split(os.environ.get("VAST_REMOTE_SSH_COMMAND", "ssh"))
    if not ssh_command:
        raise RuntimeError("vast_remote_broker: VAST_REMOTE_SSH_COMMAND is empty")
    vastai_command = shlex.split(os.environ.get("VAST_REMOTE_VASTAI_COMMAND", "vastai"))
    if not vastai_command:
        raise RuntimeError("vast_remote_broker: VAST_REMOTE_VASTAI_COMMAND is empty")
    fetch_dir = os.environ.get("VAST_REMOTE_FETCH_DIR", DEFAULT_FETCH_DIR)
    if not os.path.isabs(fetch_dir):
        raise RuntimeError(
            f"vast_remote_broker: VAST_REMOTE_FETCH_DIR must be absolute, got {fetch_dir!r}"
        )
    return Config(
        ssh_command=ssh_command,
        # Matches create_instance.py's --index 0 default alias.
        default_instance=os.environ.get("VAST_REMOTE_DEFAULT_INSTANCE", "vtao"),
        workdir=os.environ.get("VAST_REMOTE_WORKDIR", "/workspace/toy_probe_hiding"),
        venv=os.environ.get("VAST_REMOTE_VENV", "/workspace/.venv"),
        # %C is ssh's own hash of (host, port, user); avoids a fixed name
        # colliding across servers and avoids the ~104 char socket path
        # limit that a literal host name could hit. Distinct instances hash
        # to distinct paths automatically, so this stays a single template.
        control_path=os.environ.get(
            "VAST_REMOTE_CONTROL_PATH", "/tmp/vast-remote-broker-%C"
        ),
        control_persist=os.environ.get("VAST_REMOTE_CONTROL_PERSIST", "5m"),
        sync_script=os.environ.get("VAST_REMOTE_SYNC_SCRIPT", DEFAULT_SYNC_SCRIPT),
        vastai_command=vastai_command,
        fetch_dir=fetch_dir,
    )


def _check_instance(instance: Any) -> None:
    if not isinstance(instance, str) or not instance:
        raise BrokerError("instance must be a non-empty string")
    if not _VALID_INSTANCE_RE.match(instance):
        raise BrokerError(
            f"invalid instance {instance!r}: must match {_VALID_INSTANCE_RE.pattern!r} "
            "(this is create_instance.py's alias naming, e.g. 'vtao' or 'vtao1')"
        )


def _check_destructive(command: str, confirmed: bool) -> None:
    if confirmed:
        return
    # Drop disposable targets first, so a command that only names those has
    # nothing protected left to trip on.
    command = _DISPOSABLE_RUN_RE.sub(" ", command)
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


def _control_opts(config: Config) -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={config.control_path}",
        "-o",
        f"ControlPersist={config.control_persist}",
    ]


def _teardown_master(config: Config, ssh_host: str) -> None:
    """Best-effort: kill a stale multiplexed connection so the next call starts fresh."""
    try:
        subprocess.run(
            [
                *config.ssh_command,
                *_control_opts(config),
                "-O",
                "exit",
                ssh_host,
            ],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _run_ssh(argv: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        # Never inherit the server's stdin: that pipe carries JSON-RPC, and
        # a remote command reading from it would eat the protocol stream.
        stdin=subprocess.DEVNULL,
    )


def remote_exec(config: Config, arguments: dict[str, Any]) -> str:
    command = arguments.get("command")
    cwd = arguments.get("cwd")
    timeout_s = arguments.get("timeout_s", DEFAULT_TIMEOUT_S)
    use_venv = arguments.get("use_venv", True)
    confirm_destructive = arguments.get("confirm_destructive", False)
    instance = arguments.get("instance", config.default_instance)

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
    _check_instance(instance)

    _check_destructive(command, confirm_destructive)

    # The locked-down agent account, not the root alias create_instance.py
    # also writes -- see that script's write_ssh_config().
    ssh_host = f"{instance}-agent"
    script = _remote_script(config, command, cwd, use_venv)
    argv = [
        *config.ssh_command,
        *_control_opts(config),
        ssh_host,
        # One shlex.quote survives exactly one round of remote shell parsing,
        # which is what `ssh host bash -c <script>` performs.
        f"bash -c {shlex.quote(script)}",
    ]

    def run_or_raise() -> subprocess.CompletedProcess[str]:
        try:
            return _run_ssh(argv, timeout_s)
        except subprocess.TimeoutExpired:
            raise BrokerError(
                f"command timed out after {timeout_s}s and the ssh connection was "
                "closed. The remote process may still be running; re-connect to "
                "check. For work that outlives one call, launch it detached (see "
                "the tool description)."
            )

    result = run_or_raise()
    if result.returncode == 255:
        # 255 is ssh's own exit code for connection-level failures (as opposed to
        # the remote command's exit code, which passes through unchanged) -- the
        # shared connection is dead, e.g. the instance was restarted. Tear down
        # the stale control socket and retry once on a fresh connection.
        _teardown_master(config, ssh_host)
        result = run_or_raise()

    status = (
        "exit code: 0 (success)"
        if result.returncode == 0
        else f"exit code: {result.returncode}"
    )
    return "\n".join(
        [
            f"{ssh_host}:{cwd or config.workdir}$ {command}",
            status,
            _section("stdout", result.stdout),
            _section("stderr", result.stderr),
        ]
    )


def sync_flush(config: Config, arguments: dict[str, Any]) -> str:
    timeout_s = arguments.get("timeout_s", DEFAULT_SYNC_TIMEOUT_S)
    instance = arguments.get("instance", config.default_instance)
    _check_instance(instance)
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise BrokerError("timeout_s must be a number if given")
    if not 1 <= timeout_s <= MAX_TIMEOUT_S:
        raise BrokerError(f"timeout_s must be between 1 and {MAX_TIMEOUT_S}")

    if not os.path.isfile(config.sync_script):
        raise BrokerError(
            f"sync script not found at {config.sync_script}. Set "
            "VAST_REMOTE_SYNC_SCRIPT to its absolute path."
        )

    argv = [sys.executable, config.sync_script, "flush", "--host", instance]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise BrokerError(
            f"sync flush did not settle within {timeout_s}s. A large first sync can "
            "genuinely take longer, but a session halted on a conflict will never "
            "settle -- check the session list before retrying with a larger timeout_s."
        )
    except OSError as exc:
        raise BrokerError(f"could not run {config.sync_script}: {exc}")

    status = (
        "exit code: 0 (both sessions flushed)"
        if result.returncode == 0
        else (
            f"exit code: {result.returncode} -- flush failed. A halted session (sync "
            "stops rather than clobbering either side on conflict) must be resolved "
            "locally; until then the remote source is stale."
        )
    )
    return "\n".join(
        [
            f"$ {' '.join(argv)}",
            status,
            _section("stdout", result.stdout),
            _section("stderr", result.stderr),
        ]
    )


def _check_fetch_path(path: Any) -> str:
    """Validate one remote path and return it relative to '/'.

    Rejecting '..' here is not about escaping the *remote* (remote_exec
    already grants arbitrary shell there) -- it's so the local
    fetch_dir-relative destination derived from this path can't climb out
    of fetch_dir.
    """
    if not isinstance(path, str) or not path:
        raise BrokerError("each fetch path must be a non-empty string")
    if not path.startswith("/"):
        raise BrokerError(f"fetch paths must be absolute, got {path!r}")
    if "\n" in path or "\0" in path:
        raise BrokerError(f"fetch path contains a newline or NUL byte: {path!r}")
    rel = path.lstrip("/")
    if any(part in ("", ".", "..") for part in rel.split("/")):
        raise BrokerError(
            "fetch path must be normalized (no empty, '.' or '..' components), "
            f"got {path!r}"
        )
    return rel


def _fetch_instance(
    config: Config,
    instance: str,
    paths: list[str],
    rels: list[str],
    timeout_s: float,
) -> list[dict[str, Any]]:
    quoted = " ".join(shlex.quote(r) for r in rels)
    # One tar per instance rather than one cat per file: a single round trip,
    # and tar's own member paths give the local layout for free. tar exits
    # nonzero both for a missing member and for a file that grew while being
    # read (manager.log always is), so per-file success is decided locally by
    # what actually landed, not by that exit code.
    script = (
        "set -u\n"
        "cd / || exit 1\n"
        "tot=0\n"
        f"for f in {quoted}; do\n"
        '  if [ -f "$f" ]; then tot=$((tot + $(stat -c %s "$f"))); fi\n'
        "done\n"
        f'if [ "$tot" -gt {MAX_FETCH_BYTES} ]; then\n'
        f'  echo "requested files total $tot bytes, over the '
        f'{MAX_FETCH_BYTES}-byte fetch limit" >&2\n'
        "  exit 9\n"
        "fi\n"
        f"tar -cf - -C / -- {quoted}\n"
    )
    ssh_host = f"{instance}-agent"
    argv = [
        *config.ssh_command,
        *_control_opts(config),
        ssh_host,
        f"bash -c {shlex.quote(script)}",
    ]

    def run_or_raise() -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                argv, capture_output=True, timeout=timeout_s, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired:
            raise BrokerError(f"fetch from {instance} timed out after {timeout_s}s")

    result = run_or_raise()
    if result.returncode == 255:
        _teardown_master(config, ssh_host)
        result = run_or_raise()
    stderr = result.stderr.decode("utf-8", "replace").strip()
    if result.returncode == 9:
        raise BrokerError(f"{instance}: {stderr}")

    dest_root = os.path.join(config.fetch_dir, instance)
    landed: dict[str, int] = {}
    os.makedirs(dest_root, exist_ok=True)
    staging = tempfile.mkdtemp(dir=config.fetch_dir, prefix=".staging-")
    try:
        wanted = set(rels)
        if result.stdout:
            try:
                with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r|") as tf:
                    for member in tf:
                        # Only members we asked for, and the destination is
                        # built from our own validated list rather than from
                        # the member name, so a hostile archive has no say in
                        # where anything is written.
                        if member.name not in wanted or not member.isfile():
                            continue
                        src = tf.extractfile(member)
                        if src is None:
                            continue
                        staged = os.path.join(staging, member.name.replace("/", "_"))
                        with open(staged, "wb") as out:
                            shutil.copyfileobj(src, out)
                        dest = os.path.join(dest_root, member.name)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        os.replace(staged, dest)
                        landed[member.name] = os.path.getsize(dest)
            except tarfile.TarError as exc:
                raise BrokerError(
                    f"{instance}: could not read the fetched archive ({exc}). "
                    f"ssh stderr: {stderr or '(empty)'}"
                )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    records = []
    for rel, path in zip(rels, paths):
        dest = os.path.join(dest_root, rel)
        if rel in landed:
            records.append(
                {
                    "instance": instance,
                    "remote": path,
                    "local": dest,
                    "bytes": landed[rel],
                    "ok": True,
                }
            )
        else:
            # Drop any previous copy: a stale file left behind here would let
            # an unreachable or reconfigured instance keep reading as healthy
            # to whatever consumes these locally.
            try:
                os.remove(dest)
                note = "not fetched; removed the stale local copy"
            except FileNotFoundError:
                note = "not fetched"
            records.append(
                {
                    "instance": instance,
                    "remote": path,
                    "local": dest,
                    "ok": False,
                    "note": f"{note} ({stderr or 'no error output'})",
                }
            )
    return records


def fetch_files(config: Config, arguments: dict[str, Any]) -> str:
    fetches = arguments.get("fetches")
    timeout_s = arguments.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not isinstance(fetches, dict) or not fetches:
        raise BrokerError(
            "fetches is required and must be a non-empty object mapping instance "
            'name -> list of absolute remote paths, e.g. {"vtao": ["/home/agent/q.txt"]}'
        )
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise BrokerError("timeout_s must be a number if given")
    if not 1 <= timeout_s <= MAX_TIMEOUT_S:
        raise BrokerError(f"timeout_s must be between 1 and {MAX_TIMEOUT_S}")
    # Validate every instance and path before touching the network, so bad
    # caller input fails the whole call outright instead of surfacing as a
    # per-instance fetch failure -- those two mean very different things to a
    # caller deciding whether an instance is reachable.
    plan: list[tuple[str, list[str], list[str]]] = []
    for instance, paths in fetches.items():
        _check_instance(instance)
        if not isinstance(paths, list) or not paths:
            raise BrokerError(
                f"fetches[{instance!r}] must be a non-empty list of paths"
            )
        plan.append((instance, paths, [_check_fetch_path(p) for p in paths]))

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for instance, paths, rels in plan:
        # One unreachable instance shouldn't discard the others' results --
        # a partial fetch is exactly the situation a caller needs to see.
        try:
            records.extend(_fetch_instance(config, instance, paths, rels, timeout_s))
        except BrokerError as exc:
            errors.append(str(exc))

    lines = [f"fetch root: {config.fetch_dir}"]
    for rec in records:
        if rec["ok"]:
            lines.append(
                f"  [ok] {rec['instance']}:{rec['remote']} -> {rec['local']} "
                f"({rec['bytes']} bytes)"
            )
        else:
            lines.append(f"  [fail] {rec['instance']}:{rec['remote']} -- {rec['note']}")
    lines.extend(f"  [error] {e}" for e in errors)
    n_ok = sum(1 for r in records if r["ok"])
    lines.append(f"{n_ok}/{len(records)} file(s) fetched")
    if errors or n_ok != len(records):
        lines.append(
            "Local copies are stamped with their fetch time, and anything that "
            "failed has no local copy at all -- so a consumer that checks file "
            "age will not mistake a stale copy for a fresh one."
        )
    return "\n".join(lines)


def list_instances(config: Config, arguments: dict[str, Any]) -> str:
    label_prefix = arguments.get("label_prefix", DEFAULT_LABEL_PREFIX)
    if not isinstance(label_prefix, str):
        raise BrokerError("label_prefix must be a string if given")

    argv = [*config.vastai_command, "show", "instances", "--raw"]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired:
        raise BrokerError("`vastai show instances` timed out after 30s.")
    except OSError as exc:
        raise BrokerError(f"could not run {config.vastai_command[0]!r}: {exc}")

    if result.returncode != 0:
        raise BrokerError(
            f"`vastai show instances` failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    try:
        instances = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise BrokerError(
            "could not parse `vastai show instances --raw` output as JSON"
        )

    # This is vast.ai's own IP/port for the rental -- informational only,
    # not what remote_exec connects through (that goes via the ssh alias's
    # Include'd config, see the module docstring).
    matches = [
        i for i in instances if str(i.get("label") or "").startswith(label_prefix)
    ]
    if not matches:
        return f"No instances found with label prefix {label_prefix!r}."
    lines = [f"{len(matches)} instance(s) labeled {label_prefix!r}*:"]
    for inst in sorted(matches, key=lambda i: str(i.get("label") or "")):
        lines.append(
            f"  label={inst.get('label')} id={inst.get('id')} "
            f"status={inst.get('actual_status')} "
            f"vastai_ssh={inst.get('ssh_host')}:{inst.get('ssh_port')} "
            f"{inst.get('status_msg') or ''}".rstrip()
        )
    return "\n".join(lines)


TOOLS = [
    {
        "name": "remote_exec",
        "description": (
            "Run a shell command on the remote vast.ai GPU instance over SSH and return "
            "its exit code, stdout and stderr. Runs under bash in the remote project "
            "directory with the remote venv activated, so `python train_*.py ...` and "
            "`nvidia-smi` work directly. Arbitrary commands are allowed.\n"
            "Stateless executor: source syncs this-machine -> remote and runs/ syncs "
            "remote -> this-machine on its own schedule (see sync_flush); "
            "remote-side source edits get overwritten, so edit source locally instead. "
            "Tag scratch/debug runs `debug_*` -- excluded from sync and backup, so "
            "delete or leave them freely.\n"
            "Leave cwd at its default (the remote project directory, not a worktree "
            "checkout) for anything touching runs/: paths.py resolves runs/ off CWD, "
            "and only that top-level runs/ is synced back. To run a script that lives "
            "in a worktree, invoke it by absolute path rather than cd'ing into it.\n"
            "stdin is closed, so nothing interactive will work. The call blocks until "
            "the command finishes or timeout_s elapses, and a timeout closes the "
            "connection without necessarily killing the remote process -- for anything "
            "long-running (training), launch it detached instead, e.g. "
            "`setsid nohup python train.py > logs/x.log 2>&1 < /dev/null &`, then poll "
            "with further remote_exec calls that tail the log.\n"
            "Output is truncated in the middle if very long; prefer piping through "
            "head/tail/grep on the remote.\n"
            "See /etc/vast-agents-guide.md on the remote for more. The in-container "
            "vastai CLI is instance-scoped (cannot create new instances); instance "
            "lifecycle (create/destroy) and sync management are handled by vast_setup/ "
            "locally, per CLAUDE.md."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run on the remote (bash syntax; may contain pipes, &&, etc).",
                },
                "instance": {
                    "type": "string",
                    "description": (
                        "Which rental to target, by its base SSH alias (e.g. 'vtao', "
                        "'vtao1' -- see create_instance.py --index). Defaults to the "
                        "server's configured default. Use list_instances to see what's "
                        "currently up."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": "Absolute remote directory to run in. Defaults to the remote project directory -- see the tool description before pointing this at a worktree.",
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
                        "holds the only remote state worth keeping. Not needed to delete a "
                        "runs/debug_* directory: those are disposable by convention and are "
                        "never synced or backed up."
                    ),
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "sync_flush",
        "description": (
            "Force the local sync sessions to settle now, and report their "
            "status. Source (*.py + configs/) flows local -> remote and runs/ flows "
            "remote -> local, normally on their own schedule; this blocks until "
            "both are in a steady state.\n"
            "Call this after editing source locally and before launching anything on "
            "the remote that depends on the edit, otherwise the remote may still be "
            "running the previous version. Also call it before reading a finished "
            "run's outputs locally, to pull runs/ back.\n"
            "A non-zero exit usually means a session has HALTED on a conflict: the "
            "sessions are one-way-safe, so sync stops rather than overwriting "
            "either side. That needs resolving locally -- until it is, the remote "
            "source stays stale and remote_exec will keep running old code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": (
                        "Which rental's sync sessions to flush, by its base SSH alias "
                        "(e.g. 'vtao', 'vtao1'). Defaults to the server's configured "
                        "default. Must match what create_instance.py --index wrote."
                    ),
                },
                "timeout_s": {
                    "type": "number",
                    "description": f"Seconds to wait for the sync to settle (default {DEFAULT_SYNC_TIMEOUT_S}, max {MAX_TIMEOUT_S}).",
                },
            },
        },
    },
    {
        "name": "fetch_files",
        "description": (
            "Copy files from one or more remote instances onto THIS machine's "
            "filesystem, where local tools can read them. Use this for remote "
            "operational files that no sync covers -- a pool manager's queue.txt, "
            "manager.log, launched_idx -- so local analysis scripts can read them "
            "directly.\n"
            "This is the only way to get a remote file onto the local disk. A "
            "redirection inside a remote_exec command string ('cat x > y') writes "
            "on the REMOTE, not here; piping the bytes through your own context and "
            "re-writing them locally is lossy and expensive.\n"
            "Destination is fixed by this server and cannot be chosen by the "
            "caller: each file lands at <fetch_root>/<instance>/<remote path minus "
            "its leading slash>, e.g. /home/agent/q.txt on 'vtao' -> "
            f"{DEFAULT_FETCH_DIR}/vtao/home/agent/q.txt. The reported fetch root is "
            "authoritative if it was configured differently.\n"
            "Each fetched file's local mtime is its fetch time, and a file that "
            "could not be fetched leaves NO local copy (any previous one is "
            "deleted), so an unreachable instance cannot masquerade as a fresh "
            "one. Reading these back is a snapshot, not a live view -- re-fetch "
            "rather than trusting an earlier copy.\n"
            "Read-only on the remote. Regular files only; directories and globs "
            "are not expanded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fetches": {
                    "type": "object",
                    "description": (
                        "Map of instance base SSH alias (e.g. 'vtao', 'vtao1') -> "
                        "list of absolute, normalized remote paths to fetch from "
                        'that instance. Example: {"vtao": ["/home/agent/sweep_scratch/'
                        'queue.txt"], "vtao1": ["/home/agent/sweep_scratch/queue.txt"]}. '
                        "Instances are fetched independently: one unreachable "
                        "instance still returns the others' results."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "timeout_s": {
                    "type": "number",
                    "description": f"Per-instance seconds to wait (default {DEFAULT_TIMEOUT_S}, max {MAX_TIMEOUT_S}).",
                },
            },
            "required": ["fetches"],
        },
    },
    {
        "name": "list_instances",
        "description": (
            "List this project's vast.ai rentals (filtered by label prefix, default "
            "'vtao-') and their live status: id, actual_status (running/stopped/...), "
            "and vast.ai's own ssh_host:ssh_port for reference. This is status/"
            "discovery only -- it does NOT determine what remote_exec/sync_flush "
            "connect to; that goes through the ssh alias's local config (see the "
            "module docstring), independent of this tool. Use this to see what "
            "rentals currently exist before picking an `instance` value."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label_prefix": {
                    "type": "string",
                    "description": f"Filter to labels starting with this (default {DEFAULT_LABEL_PREFIX!r}).",
                },
            },
        },
    },
]

TOOL_HANDLERS = {
    "remote_exec": remote_exec,
    "sync_flush": sync_flush,
    "fetch_files": fetch_files,
    "list_instances": list_instances,
}


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
