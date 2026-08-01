"""End-to-end tests for the vast.ai remote-exec MCP server.

Drives the server through its JSON-RPC surface with a stub standing in
for the `ssh` binary: it discards the options and host, then runs the
remaining command string through a local shell exactly as a remote login
shell would. That makes the quoting path -- the part most likely to
break -- real, without needing a remote machine.
"""

import io
import json
import os
import stat

import pytest

from project_utils.mcp import vast_remote_broker

# Mimics ssh's own behaviour: everything after the host is joined and
# handed to a shell, so the server's quoting has to survive one round of
# shell parsing.
FAKE_SSH = """#!/usr/bin/env bash
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -o) shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
echo "FAKE_SSH_HOST=${args[0]}" >&2
exec /bin/sh -c "${args[*]:1}"
"""

VENV_ACTIVATE = "export VENV_ACTIVE=1\n"


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A configured server whose 'remote' is this machine, in tmp_path."""
    ssh = tmp_path / "fake_ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(VENV_ACTIVATE)

    monkeypatch.setenv("VAST_REMOTE_SSH_COMMAND", str(ssh))
    monkeypatch.setenv("VAST_REMOTE_SSH_HOST", "vtao-agent")
    monkeypatch.setenv("VAST_REMOTE_WORKDIR", str(workdir))
    monkeypatch.setenv("VAST_REMOTE_VENV", str(venv))

    class Remote:
        config = vast_remote_broker.load_config_from_env()
        workdir_path = workdir

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
            vast_remote_broker.serve(
                Remote.config, io.StringIO(json.dumps(request) + "\n"), out
            )
            result = json.loads(out.getvalue())["result"]
            return result.get("isError", False), result["content"][0]["text"]

        @staticmethod
        def exec(command, **kwargs):
            return Remote.call("remote_exec", {"command": command, **kwargs})

    return Remote


def test_handshake_and_tool_listing(remote):
    out = io.StringIO()
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    vast_remote_broker.serve(
        remote.config,
        io.StringIO("".join(json.dumps(r) + "\n" for r in requests)),
        out,
    )
    responses = [json.loads(line) for line in out.getvalue().splitlines()]

    # The notification must not draw a response, or the client desyncs.
    assert [r["id"] for r in responses] == [1, 2]
    assert responses[0]["result"]["protocolVersion"] == (
        vast_remote_broker.PROTOCOL_VERSION
    )
    tools = responses[1]["result"]["tools"]
    assert {tool["name"] for tool in tools} == set(vast_remote_broker.TOOL_HANDLERS)
    assert tools[0]["inputSchema"]["required"] == ["command"]


def test_reports_stdout_stderr_and_exit_code(remote):
    is_error, text = remote.exec("echo out; echo err >&2; exit 3")
    assert not is_error, text
    assert "exit code: 3" in text
    assert "--- stdout ---\nout" in text
    assert "err" in text.split("--- stderr ---")[1]


def test_success_is_reported_as_success(remote):
    is_error, text = remote.exec("true")
    assert not is_error
    assert "exit code: 0 (success)" in text


@pytest.mark.parametrize(
    "payload",
    [
        "it's got a quote",
        'double "quotes" too',
        "$HOME ${NOPE} `backticks` $(echo hi)",
        "semi; colon && and | pipe",
        "back\\slash",
    ],
)
def test_command_text_survives_shell_parsing_verbatim(remote, payload):
    """Quoting bugs would let the remote shell expand or split these."""
    is_error, text = remote.exec(f"cat <<'EOF'\n{payload}\nEOF")
    assert not is_error, text
    assert payload in text


def test_runs_in_workdir_by_default_and_honours_cwd(remote, tmp_path):
    is_error, text = remote.exec("pwd")
    assert not is_error, text
    assert str(remote.workdir_path) in text

    other = tmp_path / "elsewhere"
    other.mkdir()
    is_error, text = remote.exec("pwd", cwd=str(other))
    assert not is_error, text
    assert str(other) in text


def test_missing_cwd_fails_rather_than_running_elsewhere(remote, tmp_path):
    """A bad cwd must not silently degrade into running in $HOME."""
    is_error, text = remote.exec("pwd", cwd=str(tmp_path / "nope"))
    assert not is_error, text
    assert "exit code: 1" in text
    assert "--- stdout --- (empty)" in text
    assert "No such file or directory" in text


def test_venv_is_activated_by_default_and_can_be_skipped(remote):
    is_error, text = remote.exec("echo venv=${VENV_ACTIVE:-none}")
    assert not is_error, text
    assert "venv=1" in text

    is_error, text = remote.exec("echo venv=${VENV_ACTIVE:-none}", use_venv=False)
    assert not is_error, text
    assert "venv=none" in text


def test_missing_venv_is_not_fatal(remote, monkeypatch):
    monkeypatch.setenv("VAST_REMOTE_VENV", "/definitely/not/here")
    config = vast_remote_broker.load_config_from_env()
    text = vast_remote_broker.remote_exec(config, {"command": "echo alive"})
    assert "exit code: 0" in text
    assert "alive" in text


def test_stdin_is_closed_not_wired_to_the_protocol_stream(remote):
    """A remote command reading stdin would otherwise eat JSON-RPC traffic."""
    is_error, text = remote.exec("cat")
    assert not is_error, text
    assert "exit code: 0" in text
    assert "--- stdout --- (empty)" in text


def test_timeout_is_reported_as_an_error(remote):
    is_error, text = remote.exec("sleep 5", timeout_s=1)
    assert is_error
    assert "timed out after 1s" in text


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf runs",
        "rm -rf /workspace",
        "rm -rf ~/runs/",
        "dd if=/dev/zero of=/workspace/x",
    ],
)
def test_destructive_commands_are_refused_by_default(remote, command):
    is_error, text = remote.exec(command)
    assert is_error
    assert "confirm_destructive" in text


def test_destructive_commands_run_when_confirmed(remote):
    target = remote.workdir_path / "runs"
    target.mkdir()
    is_error, text = remote.exec("rm -rf runs", confirm_destructive=True)
    assert not is_error, text
    assert not target.exists()


@pytest.mark.parametrize(
    "command",
    [
        "ls runs",
        "du -sh runs/",
        "python plot_curves.py --tag runs_v2",
        "rm -rf /tmp/scratch",
        "grep -r norm runs",
    ],
)
def test_guard_does_not_block_ordinary_commands(remote, command):
    """The guard must not be so broad that it makes the tool annoying."""
    vast_remote_broker._check_destructive(command, confirmed=False)


def test_long_output_is_truncated_in_the_middle(remote):
    line_count = vast_remote_broker.MAX_OUTPUT_CHARS
    is_error, text = remote.exec(f"seq 1 {line_count}")
    assert not is_error, text
    assert "chars omitted" in text
    # Both ends survive: the head and the tail of the log are what matter.
    assert "\n1\n" in text
    assert f"\n{line_count}\n" in text


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"command": ""},
        {"command": "   "},
        {"command": "ls", "timeout_s": 0},
        {"command": "ls", "timeout_s": vast_remote_broker.MAX_TIMEOUT_S + 1},
        {"command": "ls", "timeout_s": "30"},
        {"command": "ls", "cwd": ""},
        {"command": "ls", "use_venv": "yes"},
        {"command": "ls", "confirm_destructive": "yes"},
    ],
)
def test_invalid_arguments_are_rejected(remote, arguments):
    is_error, _ = remote.call("remote_exec", arguments)
    assert is_error


def test_unknown_tool_is_an_error_not_a_crash(remote):
    is_error, text = remote.call("remote_shell", {})
    assert is_error
    assert "unknown tool" in text


def test_ssh_host_is_required(monkeypatch):
    monkeypatch.delenv("VAST_REMOTE_SSH_HOST", raising=False)
    with pytest.raises(RuntimeError, match="VAST_REMOTE_SSH_HOST"):
        vast_remote_broker.load_config_from_env()


def test_defaults_cover_the_optional_settings(monkeypatch):
    monkeypatch.setenv("VAST_REMOTE_SSH_HOST", "vtao-agent")
    for name in ("VAST_REMOTE_SSH_COMMAND", "VAST_REMOTE_WORKDIR", "VAST_REMOTE_VENV"):
        monkeypatch.delenv(name, raising=False)
    config = vast_remote_broker.load_config_from_env()
    assert config.ssh_command == ["ssh"]
    assert os.path.isabs(config.workdir) and os.path.isabs(config.venv)
    assert config.control_path
    assert config.control_persist


def test_stale_connection_is_retried_once_on_a_fresh_one(tmp_path, monkeypatch):
    """A 255 exit (ssh's own connection-failure code) should trigger one retry,
    not be reported straight to the caller as a command failure."""
    marker = tmp_path / "seen_teardown"
    ssh = tmp_path / "fake_ssh"
    ssh.write_text(f"""#!/usr/bin/env bash
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -o) shift 2 ;;
        -O) marker_seen=1; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
if [ -n "$marker_seen" ]; then
    touch {marker}
    exit 0
fi
if [ ! -f {marker} ]; then
    exit 255
fi
exec /bin/sh -c "${{args[*]:1}}"
""")
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.setenv("VAST_REMOTE_SSH_COMMAND", str(ssh))
    monkeypatch.setenv("VAST_REMOTE_SSH_HOST", "vtao-agent")
    monkeypatch.setenv("VAST_REMOTE_WORKDIR", str(workdir))
    monkeypatch.delenv("VAST_REMOTE_VENV", raising=False)
    config = vast_remote_broker.load_config_from_env()

    text = vast_remote_broker.remote_exec(config, {"command": "echo alive"})
    assert "exit code: 0" in text
    assert "alive" in text
    assert marker.exists()
