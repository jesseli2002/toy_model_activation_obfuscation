## Project structure
The two main entry points are:
- train_adversarial_logreg.py (training models)
- adversarial_report.py (reporting on results)

As of writing, there is active development on sweep_*.py (various reporting scripts for sweeps) - while the scripts themselves haven't stabilized, they should still be considered when refactoring.

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

## Style and code standards
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
    - Don't give unimportant history like a change in design preference (contrast important history: examples of past pitfalls/bugs of a more obvious-seeming solution; even in these cases, keep it concise and link to external issue reasoning)
    - If a subtle flaw was identified, don't pollute the code with a long comment about it - often it's both too short to explain all the nuance and so long that it interrupts the flow. Instead, use one or two line comments, and refer to a PR# where more details can be found.
    - Rule of thumb: If you're removing features but somehow the codebase got longer, something has probably gone wrong.
- Demand elegance (Balanced)
    - For non-trivial changes, ask: “Is there a more elegant solution?”
    - If a fix feels hacky, ask: “Knowing everything I know now, implement the elegant solution.”
    - Skip this for simple fixes — don’t over-engineer
- Keep commits small and self-contained.
    - The size of the diff for a single commit should be inversely proportional to its logical complexity. A simple rename of a method might touch a bunch of files for every place it's used - that's fine since it's not complex. Meanwhile, a refactoring of a god function might involve moving a lot of code around - that's also fine. But both shouldn't happen in the same commit.
- Don't line-break long strings (even if they otherwise violate the max column width) so their contents stay searchable.

## Workflow
- Multiple agents and the user may be coding simultaneously - use git worktrees to isolate your changes.
- By default, make a PR for changes to tracked files. Undrafted GitHub PRs, not draft ones.
- *.tmp.py files are throwaway scripts - don't worry about code quality when reading/writing them, and when the user asks for a throwaway script, use a .tmp.py suffix.
- Runs tagged `debug_*` are throwaway: they are excluded from every sync and backup, so use that prefix for scratch/debugging runs and treat them as disposable.
- Commits should ideally be small and self-contained to help with reviewing.
- If during testing, you encounter warnings - don't ignore them, unless you tell the user and have a very good reason to think it's a false positive (e.g. you're intentionally trying to trigger it to test it).

## Autonomous engineering
- Due to machine resource limitations, realistically at most one agent should be running training code at a time. If tasking subagents to complete work, this should be considered for task allocation.
    - This only applies for local code. For remote work on a vastai instance, it's a case-by-case situation; ask the user if you're unsure. Generally, only use the remote vastai box if the user directs you to.
- For pure code-location tasks (Explore agent, "quick"/"medium" breadth), use model: "haiku". Reserve Sonnet/Opus for exploration that requires judging ambiguous matches or synthesizing findings.
- There are some unit tests but they only cover specific parts of the codebase - unit tests passing DOES NOT mean that the codebase is working.

## Background processes
Each Bash call runs in its own PID namespace, so a sandboxed `ps` sees only its own invocation, and plain `&`/`nohup`/`setsid` processes are killed when the invocation ends.
- Never use `ps` to check whether a process you launched is still alive; empty output means "not visible from here," not "dead."
- For long-running work, launch it with `run_in_background`, monitor via its output file and completion notification, and stop it with the TaskStop tool (which reaps the whole subprocess tree via the PID namespace).
- To inspect or kill a host process, or anything the harness isn't tracking, hand the user a host `ps`/`kill` command rather than disabling the sandbox.
- Above all, be cautious - there's still quirks in the sandbox environment setup. If a process (especially a resource-intensive one) should be running but you can't find it, don't assume it died unexpectedly; check if your harnesses can tell the difference between a dead and invisible process, and don't be afraid to ask the user for help diagnosing issues.

## Notes on sandbox environment
This environment is in a sandbox. Writes, sensitive reads, and network access are highly restricted at the OS level. As a result, the following processes are preferred:
- Use the Github MCP server to push features, instead of Bash git/gh commands.
    - For git push, a custom git-push-broker MCP is used; the GitHub one doesn't preserve commit history properly.
- If you get something like this error on all Bash calls: `apply-seccomp: write /proc/self/setgroups (nested userns is capability-restricted; caller must provide CAP_SYS_ADMIN): Permission denied` - it's a known issue that occurs after machine reboot. Stop what you're doing and tell the user; there's a known fix which needs user intervention
- If asking user to rm files - give a trash command instead since user disabled rm in bashrc.
- GPU access is not available in sandbox. Never run a script that does non-trivial calculation locally — training, checkpoint loading/inference, probe refits, torch/numpy-heavy metric recomputation — not even once "just to check it runs." --help/argparse-only invocations are fine.Anything past that: hand the script to the user to run/validate. Exceptions include:
    - `pytest`; unit tests are still reasonably fast on CPU.
    - Scripts where heavy computations are cached. If you're unsure if the results are cached for the specific arguments you're using, set a timeout of ~20s.
