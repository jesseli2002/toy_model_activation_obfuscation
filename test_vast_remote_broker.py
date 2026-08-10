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
# shell parsing. Also echoes which host it was pointed at, so tests can
# verify instance -> ssh_host selection.
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
def sync_script(tmp_path, monkeypatch):
    """Stub standing in for sync_vastai.py. Returns a writer that sets its body,
    so a test can make the flush succeed, fail, or hang. Written in Python since
    the broker always invokes the sync script via `sys.executable`."""
    script = tmp_path / "fake_sync.py"

    def write_script(body):
        script.write_text("#!/usr/bin/env python3\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return script

    write_script('import sys\nprint(f"flushed: {sys.argv[1]}")\n')
    monkeypatch.setenv("VAST_REMOTE_SYNC_SCRIPT", str(script))
    return write_script


@pytest.fixture
def fake_vastai(tmp_path, monkeypatch):
    """Stub standing in for the vastai CLI, for list_instances tests. Ignores
    its argv and prints whatever JSON body the test configures."""
    script = tmp_path / "fake_vastai.py"

    def write_script(body):
        script.write_text("#!/usr/bin/env python3\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return script

    write_script("print('[]')\n")
    monkeypatch.setenv("VAST_REMOTE_VASTAI_COMMAND", str(script))
    return write_script


@pytest.fixture
def remote(tmp_path, monkeypatch, sync_script):
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
    # "vtao" is the default instance, so remote_exec's default ssh_host
    # ("<instance>-agent") comes out as "vtao-agent" -- same alias the old
    # fixed-host env var used, so FAKE_SSH_HOST assertions stay meaningful.
    monkeypatch.setenv("VAST_REMOTE_DEFAULT_INSTANCE", "vtao")
    monkeypatch.setenv("VAST_REMOTE_WORKDIR", str(workdir))
    monkeypatch.setenv("VAST_REMOTE_VENV", str(venv))
    fetch_dir = tmp_path / "fetched"
    monkeypatch.setenv("VAST_REMOTE_FETCH_DIR", str(fetch_dir))

    class Remote:
        config = vast_remote_broker.load_config_from_env()
        workdir_path = workdir
        fetch_root = fetch_dir

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

        @staticmethod
        def flush(**kwargs):
            return Remote.call("sync_flush", kwargs)

        @staticmethod
        def fetch(fetches, **kwargs):
            return Remote.call("fetch_files", {"fetches": fetches, **kwargs})

        @staticmethod
        def fetched_path(instance, remote_path):
            """Where fetch_files is documented to land a file."""
            return fetch_dir / instance / str(remote_path).lstrip("/")

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
    tools = {tool["name"]: tool for tool in responses[1]["result"]["tools"]}
    assert set(tools) == set(vast_remote_broker.TOOL_HANDLERS)
    assert tools["remote_exec"]["inputSchema"]["required"] == ["command"]
    # Flushing and listing take no required arguments -- they act on the
    # configured default instance / all labeled instances.
    assert not tools["sync_flush"]["inputSchema"].get("required")
    assert not tools["list_instances"]["inputSchema"].get("required")


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
        # A disposable run named alongside a protected target is still refused.
        "rm -rf runs/debug_a runs/keepme",
        # Escaping the run directory forfeits the carve-out.
        "rm -rf runs/debug_a/../..",
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
        # debug_* runs are disposable by convention: deleting one needs no
        # confirmation, since nothing syncs or backs them up.
        "rm -rf runs/debug_smoke",
        "rm -rf runs/debug_smoke/",
        "rm -rf runs/debug_*",
        "rm -rf runs/debug_a runs/debug_b",
    ],
)
def test_guard_does_not_block_ordinary_commands(remote, command):
    """The guard must not be so broad that it makes the tool annoying."""
    vast_remote_broker._check_destructive(command, confirmed=False)


def test_guard_does_not_block_debug_run_deletion_by_absolute_path(remote):
    # remote_exec's cwd is caller-controlled, so a debug_* target named by
    # absolute path is just as common as one named relative to workdir.
    command = f"rm -rf {remote.workdir_path}/runs/debug_smoke"
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


def test_flush_invokes_the_sync_script_and_reports_its_output(remote):
    is_error, text = remote.flush()
    assert not is_error, text
    # The subcommand matters: `start` or `stop` would create or tear down
    # sessions rather than settling the existing ones.
    assert "flushed: flush" in text
    assert "exit code: 0" in text


def test_flush_reports_a_failed_flush_without_hiding_it_as_success(remote, sync_script):
    """A halted session is the failure that matters: the remote silently keeps
    running stale source, so the caller must not read this as a clean sync."""
    sync_script(
        "import sys\n"
        'print("unable to flush: session is halted on conflict", file=sys.stderr)\n'
        "sys.exit(1)\n"
    )
    is_error, text = remote.flush()
    assert "exit code: 1" in text
    assert "halted" in text.lower()
    assert "session is halted on conflict" in text


def test_flush_that_never_settles_is_reported_as_an_error(remote, sync_script):
    sync_script("import time\ntime.sleep(30)\n")
    is_error, text = remote.flush(timeout_s=1)
    assert is_error
    assert "did not settle" in text


def test_missing_sync_script_names_the_override(remote, monkeypatch, tmp_path):
    monkeypatch.setenv("VAST_REMOTE_SYNC_SCRIPT", str(tmp_path / "absent.sh"))
    remote.config = vast_remote_broker.load_config_from_env()
    is_error, text = remote.flush()
    assert is_error
    assert "VAST_REMOTE_SYNC_SCRIPT" in text


@pytest.mark.parametrize("timeout_s", [0, -1, "soon", True])
def test_flush_rejects_a_bad_timeout(remote, timeout_s):
    is_error, text = remote.flush(timeout_s=timeout_s)
    assert is_error
    assert "timeout_s" in text


def test_defaults_cover_the_optional_settings(monkeypatch):
    for name in (
        "VAST_REMOTE_SSH_COMMAND",
        "VAST_REMOTE_DEFAULT_INSTANCE",
        "VAST_REMOTE_WORKDIR",
        "VAST_REMOTE_VENV",
        "VAST_REMOTE_VASTAI_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    config = vast_remote_broker.load_config_from_env()
    assert config.ssh_command == ["ssh"]
    # Matches create_instance.py's --index 0 default alias.
    assert config.default_instance == "vtao"
    assert config.vastai_command == ["vastai"]
    assert os.path.isabs(config.workdir) and os.path.isabs(config.venv)
    assert config.control_path
    assert config.control_persist
    assert os.path.isabs(config.sync_script)


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
    monkeypatch.setenv("VAST_REMOTE_DEFAULT_INSTANCE", "vtao")
    monkeypatch.setenv("VAST_REMOTE_WORKDIR", str(workdir))
    monkeypatch.delenv("VAST_REMOTE_VENV", raising=False)
    config = vast_remote_broker.load_config_from_env()

    text = vast_remote_broker.remote_exec(config, {"command": "echo alive"})
    assert "exit code: 0" in text
    assert "alive" in text
    assert marker.exists()


# --- multi-instance: instance argument selection & validation ---------------


def test_instance_argument_selects_which_rental_remote_exec_targets(remote):
    is_error, text = remote.exec("true", instance="vtao1")
    assert not is_error, text
    assert "FAKE_SSH_HOST=vtao1-agent" in text


def test_default_instance_is_used_when_none_given(remote):
    is_error, text = remote.exec("true")
    assert not is_error, text
    assert "FAKE_SSH_HOST=vtao-agent" in text


def test_flush_passes_instance_as_host_to_sync_script(remote, sync_script):
    sync_script("import sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    is_error, text = remote.flush(instance="vtao1")
    assert not is_error, text
    assert json.dumps(["flush", "--host", "vtao1"]) in text


def test_flush_defaults_to_the_configured_default_instance(remote, sync_script):
    sync_script("import sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    is_error, text = remote.flush()
    assert not is_error, text
    assert json.dumps(["flush", "--host", "vtao"]) in text


@pytest.mark.parametrize(
    "instance",
    [
        "",
        "-oProxyCommand=curl evil.example",
        "vtao/../etc",
        "vtao;rm -rf /",
        "vtao agent",
        "vtao$(whoami)",
    ],
)
def test_invalid_instance_is_rejected_by_remote_exec(remote, instance):
    """A caller-controlled instance string lands in ssh's argv; a leading '-'
    in particular must not be readable as a flag (e.g. ProxyCommand)."""
    is_error, text = remote.exec("true", instance=instance)
    assert is_error
    assert "instance" in text.lower()


@pytest.mark.parametrize("instance", ["", "vtao/etc", "vtao;x"])
def test_invalid_instance_is_rejected_by_sync_flush(remote, instance):
    is_error, text = remote.flush(instance=instance)
    assert is_error
    assert "instance" in text.lower()


# --- list_instances -----------------------------------------------------


def test_list_instances_filters_by_label_prefix(remote, fake_vastai):
    fake_vastai(
        "import json\n"
        "print(json.dumps([\n"
        "    {'label': 'vtao-0', 'id': 1, 'actual_status': 'running',\n"
        "     'ssh_host': '1.2.3.4', 'ssh_port': 22, 'status_msg': ''},\n"
        "    {'label': 'other-project', 'id': 2, 'actual_status': 'running',\n"
        "     'ssh_host': '5.6.7.8', 'ssh_port': 22, 'status_msg': ''},\n"
        "]))\n"
    )
    remote.config = vast_remote_broker.load_config_from_env()
    is_error, text = remote.call("list_instances", {})
    assert not is_error, text
    assert "vtao-0" in text
    assert "id=1" in text
    assert "other-project" not in text


def test_list_instances_no_matches_is_not_an_error(remote, fake_vastai):
    fake_vastai("print('[]')\n")
    remote.config = vast_remote_broker.load_config_from_env()
    is_error, text = remote.call("list_instances", {})
    assert not is_error, text
    assert "no instances" in text.lower()


def test_list_instances_custom_label_prefix(remote, fake_vastai):
    fake_vastai(
        "import json\n"
        "print(json.dumps([\n"
        "    {'label': 'vtao-0', 'id': 1, 'actual_status': 'running'},\n"
        "    {'label': 'other-9', 'id': 2, 'actual_status': 'running'},\n"
        "]))\n"
    )
    remote.config = vast_remote_broker.load_config_from_env()
    is_error, text = remote.call("list_instances", {"label_prefix": "other-"})
    assert not is_error, text
    assert "other-9" in text
    assert "vtao-0" not in text


def test_list_instances_reports_vastai_failure(remote, fake_vastai):
    fake_vastai("import sys\nprint('bad key', file=sys.stderr)\nsys.exit(1)\n")
    remote.config = vast_remote_broker.load_config_from_env()
    is_error, text = remote.call("list_instances", {})
    assert is_error
    assert "bad key" in text


def test_list_instances_reports_unparseable_output(remote, fake_vastai):
    fake_vastai("print('not json')\n")
    remote.config = vast_remote_broker.load_config_from_env()
    is_error, text = remote.call("list_instances", {})
    assert is_error
    assert "json" in text.lower()


def test_list_instances_rejects_non_string_label_prefix(remote):
    is_error, text = remote.call("list_instances", {"label_prefix": 5})
    assert is_error
    assert "label_prefix" in text


# --- fetch_files -------------------------------------------------------------
#
# fetch_files is the only path that puts a remote file on the local disk, and
# its consumers (queue_audit.py, pool_health.py) decide real things from what
# they find there. A silently stale or missing copy is worse than a loud
# failure, so the destination layout and the no-stale-leftovers rule are what
# these pin down.


@pytest.fixture
def remote_file(tmp_path):
    """Writes a file on the 'remote' (this machine) and returns its path."""

    def write(name, content):
        path = tmp_path / "remote_files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    return write


def test_fetch_lands_files_at_the_documented_instance_scoped_path(remote, remote_file):
    queue = remote_file("queue.txt", "python train.py --tag a\n")
    is_error, text = remote.fetch({"vtao": [str(queue)]})

    landed = remote.fetched_path("vtao", queue)
    assert not is_error, text
    assert landed.read_text() == queue.read_text()
    assert str(landed) in text
    assert "1/1 file(s) fetched" in text


def test_fetch_keeps_instances_in_separate_trees(remote, remote_file):
    # Every rental uses the same remote scratch layout, so identical paths on
    # two instances must not collide into one local file.
    queue = remote_file("queue.txt", "shared path\n")
    is_error, _ = remote.fetch({"vtao": [str(queue)], "vtao1": [str(queue)]})

    assert not is_error
    assert remote.fetched_path("vtao", queue).exists()
    assert remote.fetched_path("vtao1", queue).exists()


def test_fetch_reports_a_missing_file_without_discarding_the_others(
    remote, remote_file
):
    queue = remote_file("queue.txt", "real\n")
    absent = queue.parent / "absent.txt"
    is_error, text = remote.fetch({"vtao": [str(queue), str(absent)]})

    assert not is_error, text
    assert remote.fetched_path("vtao", queue).exists()
    assert not remote.fetched_path("vtao", absent).exists()
    assert "[fail]" in text and "1/2 file(s) fetched" in text


def test_failed_refetch_deletes_the_previous_local_copy(remote, remote_file):
    # A leftover copy would let an unreachable instance keep reading as
    # healthy to whatever parses these files later.
    queue = remote_file("queue.txt", "first\n")
    remote.fetch({"vtao": [str(queue)]})
    assert remote.fetched_path("vtao", queue).exists()

    queue.unlink()
    is_error, text = remote.fetch({"vtao": [str(queue)]})

    assert not is_error
    assert not remote.fetched_path("vtao", queue).exists()
    assert "stale local copy" in text


def test_fetched_copy_is_stamped_with_its_fetch_time(remote, remote_file):
    # queue_audit.py's staleness check reads this mtime, so it has to mean
    # "when we fetched", not the remote file's own (possibly ancient) mtime.
    queue = remote_file("queue.txt", "x\n")
    os.utime(queue, (0, 0))
    remote.fetch({"vtao": [str(queue)]})

    assert remote.fetched_path("vtao", queue).stat().st_mtime > 0


@pytest.mark.parametrize(
    "path",
    ["relative/path", "/climbs/../../out", "/has/a\nnewline", "", 5],
)
def test_fetch_rejects_a_bad_path_outright(remote, path):
    # Hard error, not a per-instance [fail] line: bad caller input and an
    # unreachable instance need to stay distinguishable.
    is_error, text = remote.fetch({"vtao": [path]})
    assert is_error, text


def test_a_bad_path_rejects_the_whole_call_before_fetching_anything(
    remote, remote_file
):
    queue = remote_file("queue.txt", "x\n")
    is_error, _ = remote.fetch({"vtao": [str(queue)], "vtao1": ["relative"]})

    assert is_error
    assert not remote.fetched_path("vtao", queue).exists()


@pytest.mark.parametrize(
    "fetches", [{}, {"vtao": []}, {"vtao": "not-a-list"}, {"-bad": ["/x"]}, "nope"]
)
def test_fetch_rejects_a_malformed_fetches_argument(remote, fetches):
    is_error, text = remote.fetch(fetches)
    assert is_error, text


def test_fetch_refuses_an_oversize_request(remote, remote_file, monkeypatch):
    big = remote_file("big.txt", "x" * 4096)
    monkeypatch.setattr(vast_remote_broker, "MAX_FETCH_BYTES", 100)
    is_error, text = remote.fetch({"vtao": [str(big)]})

    assert "fetch limit" in text
    assert not remote.fetched_path("vtao", big).exists()


@pytest.mark.parametrize("timeout_s", [0, -1, "soon", True])
def test_fetch_rejects_a_bad_timeout(remote, timeout_s):
    is_error, text = remote.fetch({"vtao": ["/x"]}, timeout_s=timeout_s)
    assert is_error
    assert "timeout_s" in text
