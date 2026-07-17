## Style
Most scripts have some heavy imports e.g. (torch), which can take some time. This is annoying if you only call --help on a command-line script. So, preferably parse_args should be defined & called before all the imports are completed. This is not a hard rule; if argument parsing necessarily relies on some heavy libraries, don't sacrifice code readability or command line ease-of-use for this one specific problem.

## Workflow
- Multiple agents and the user may be coding simultaneously - use git worktrees to isolate your changes.
    - Make sure to exit worktrees once done, so that the branches get cleared out. If you're unsure if you're done (e.g. waiting on approval review), remind the user to let you know when approval is ready so that you can exit the worktree.
- By default, do not set GitHub PRs as draft.

## Autonomous engineering
Due to machine resource limitations, realistically at most one agent should be running training code at a time. If tasking subagents to complete work, this should be considered for task allocation.

## Background processes
Each Bash call runs in its own PID namespace, so a sandboxed `ps` sees only its own invocation, and plain `&`/`nohup`/`setsid` processes are killed when the invocation ends.
- Never use `ps` to check whether a process you launched is still alive; empty output means "not visible from here," not "dead."
- For long-running work, launch it with `run_in_background`, monitor via its output file and completion notification, and stop it with the TaskStop tool (which reaps the whole subprocess tree via the PID namespace).
- To inspect or kill a host process, or anything the harness isn't tracking, hand the user a host `ps`/`kill` command rather than disabling the sandbox.
- Above all, be cautious - there's still quirks in the sandbox environment setup. If a process (especially a resource-intensive one) should be running but you can't find it, don't assume it died unexpectedly; check if your harnesses can tell the difference between a dead and invisible process, and don't be afraid to ask the user for help diagnosing issues.

## Notes on sandbox environment
This environment is in a sandbox. Writes and sensitive reads outside this directory are denied at the OS level, and network access is highly restricted. Moreover, shell commands which don't match the deterministic allowlist pass through a classifier which will raise permission prompts for complex commands. To reduce the number of permission prompts:
- Commands with shell variable expansion that can't be statically verified (e.g. `$VAR`, `$$`, `for i in "$@"; do ...$i...; done`) raise a permission prompt even when the command is safe, because the sandbox can't confirm what the expansion will resolve to. Solutions:
    - Substitute the known value directly instead of using a variable or loop.
    - To find environment variables, use `printenv ENV_VAR` instead of `echo $ENV_VAR`
- The safety classifier favors simple, single-purpose calls over multi-command bundles. Avoid needing the classifier by construction:
    - For read tasks use native tools (Read/Grep/Glob). Write-capabale tools like `sed` are not automatically approved, even if individual calls are read-only.
    - `black` is allow-listed only via its absolute path `/home/jesse/v/bin/black` (guards against a shadowed `black` on PATH); invoke it that way. `black --check <file>` doubles as a read-only syntax check and is preferred over `py_compile`.
- If you run into permissions issues, prefer trying to solve the cause (and ask the user to help debug permissions), rather than working around the symptoms and trying a bunch of techniques to get past them.
- Use the Github MCP servers to push features, instead of Bash git/gh commands.
    - There are two identified servers (github-readonly and github-write): One dedicated for reading (highly permissive), and one dedicated for writing (highly restricted). Make sure to use the right server.
