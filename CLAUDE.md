## Project structure
The two main entry points are:
- train_adversarial_logreg.py (training models)
- adversarial_report.py (reporting on results)

### Key directories
- vast_setup/
	- Scripts to setup and manage a VastAI instance. Gitignored here and tracked as a separate repo; to work in it, user must /cd into vast_setup/; otherwise your worktree changes will not be reflected.
- runs/
	- Training run checkpoints
- plot/
	- Output plots from adversarial_report.py
- project_utils/
	- Local MCP servers & utility scripts
Other directories are generally not relevant unless the user specifically mentions them.

## Style
- Most scripts have some heavy imports e.g. (torch), which can take some time. This is annoying if you only call --help on a command-line script. Preferably structure CLI scripts like this:
```python
import argparse # and other light imports

def parse_args():
    ...

if __name__ == "__main__":
    args = parse_args() # ArgumentParser early exits on --help

import torch # and other heavy imports
... # rest of code
```
This is not a hard rule; if argument parsing necessarily relies on some heavy libraries, don't sacrifice code readability or command line ease-of-use for this one specific problem.
- Indicate tensor/array shapes/sizes with jaxtyping on functions and methods
- Write concise docs; avoid brittle docs.
    - Cover high level concepts, not implementation (which would be redundant and brittle to implementation changes).
    - Don't list specific callers/users; instead explain use cases, to be robust to changes in where a method gets used.
    - Don't give unimportant history (contrast important history: examples of past pitfalls/bugs of a more obvious-seeming solution; even in these cases, keep it concise and link to external issue reasoning)
    - If a subtle flaw was identified, don't pollute the code with a long comment about it - often it's both too short to explain all the nuance and so long that it interrupts the flow. Instead, use one or two line comments, and refer to a PR# where more details can be found.

## Workflow
- Multiple agents and the user may be coding simultaneously - use git worktrees to isolate your changes.
- Don't use draft GitHub PRs unless told.
- *.tmp.py files are throwaway scripts - don't worry about code quality when reading/writing them, and when the user asks for a throwaway script, use a .tmp.py suffix.
- Runs tagged `debug_*` are throwaway: they are excluded from every sync and backup, so use that prefix for scratch/debugging runs and treat them as disposable.
- Commits should ideally be small and self-contained to help with reviewing.
- If during testing, you encounter warnings - don't ignore them, unless you tell the user and have a very good reason to think it's a false positive (e.g. you're intentionally trying to trigger it to test it).

## Autonomous engineering
- Due to machine resource limitations, realistically at most one agent should be running training code at a time. If tasking subagents to complete work, this should be considered for task allocation.
- For pure code-location tasks (Explore agent, "quick"/"medium" breadth), use model: "haiku". Reserve Sonnet/Opus for exploration that requires judging ambiguous matches or synthesizing findings.
- There are some unit tests but they only cover specific parts of the codebase - unit tests passing DOES NOT mean that the codebase is working.

## Background processes
Each Bash call runs in its own PID namespace, so a sandboxed `ps` sees only its own invocation, and plain `&`/`nohup`/`setsid` processes are killed when the invocation ends.
- Never use `ps` to check whether a process you launched is still alive; empty output means "not visible from here," not "dead."
- For long-running work, launch it with `run_in_background`, monitor via its output file and completion notification, and stop it with the TaskStop tool (which reaps the whole subprocess tree via the PID namespace).
- To inspect or kill a host process, or anything the harness isn't tracking, hand the user a host `ps`/`kill` command rather than disabling the sandbox.
- Above all, be cautious - there's still quirks in the sandbox environment setup. If a process (especially a resource-intensive one) should be running but you can't find it, don't assume it died unexpectedly; check if your harnesses can tell the difference between a dead and invisible process, and don't be afraid to ask the user for help diagnosing issues.

## Notes on sandbox environment
This environment is in a sandbox. Writes, sensitive reads, and network access are highly restricted at the OS level. Moreover, shell commands which don't match the deterministic allowlist pass through a classifier which will raise permission prompts for complex commands. To reduce the number of permission prompts:
- Commands with shell variable expansion that can't be statically verified (e.g. `$VAR`, `$$`, `for i in "$@"; do ...$i...; done`) raise a permission prompt even when the command is safe, because the sandbox can't confirm what the expansion will resolve to. Solutions:
    - Substitute the known value directly instead of using a variable or loop.
    - To find environment variables, use `printenv ENV_VAR` instead of `echo $ENV_VAR`
- The safety classifier favors simple, single-purpose calls over multi-command bundles. Avoid needing the classifier by construction:
    - For read tasks use native tools (Read/Grep/Glob). Write-capable tools like `sed` are not automatically approved, even if individual calls are read-only.
    - `black` is automatically run as a hook; no need to manually run it. `black --check <file>` doubles as a read-only syntax check.
- For non-trivial Python, write it to a temporary file and run `python tmp.py` rather than `python -c "…"`. Inline `-c` trips the command-safety classifier and forces a permission prompt — specifically a newline-then-`#` comment inside the quoted arg, or an embedded deny-listed path. Reserve `-c` for short, comment-free, single-line snippets.
- If you run into permissions issues, prefer trying to solve the cause (and ask the user to help debug permissions), rather than working around the symptoms and trying a bunch of techniques to get past them.
- Use the Github MCP servers to push features, instead of Bash git/gh commands.
    - There are two identified servers (github-readonly and github-write): One dedicated for reading (highly permissive), and one dedicated for writing (highly restricted). USE THE READONLY SERVER IF ONLY READING; OTHERWISE THE COMMAND MAY GET REJECTED
    - For git push, a custom git-push-broker MCP is used; the one from GitHub doesn't preserve commit history properly.
- If you get this error or something like it on all Bash calls: `apply-seccomp: write /proc/self/setgroups (nested userns is capability-restricted; caller must provide CAP_SYS_ADMIN): Permission denied` - it's a known issue that occurs occasionally when the machine is updated. Stop what you're doing and tell the user; there's a known fix which needs user intervention
- If asking user to rm files - give a trash command instead since user disabled rm in bashrc.
