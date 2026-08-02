"""Delete all linked git worktrees (and their branches) after confirmation.

Lists every worktree except the main one, asks for a single y/N confirmation,
then unlocks and force-removes each, deletes the branch it was on, and prunes
stale remote-tracking refs.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "repo_dir",
        nargs="?",
        default=".",
        help="Path inside the git repo to operate on (default: cwd)",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    return p.parse_args()


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def list_worktrees(repo_dir: Path) -> list[dict]:
    """Parse `git worktree list --porcelain` into a list of entries in order.

    The first entry is always the main worktree.
    """
    result = run(["git", "worktree", "list", "--porcelain"], repo_dir)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    worktrees = []
    current = {}
    for line in result.stdout.splitlines():
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in ("bare", "detached", "locked", "prunable"):
            current[key] = value or True
    if current:
        worktrees.append(current)
    return worktrees


def main():
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()

    worktrees = list_worktrees(repo_dir)
    if len(worktrees) <= 1:
        print("No linked worktrees to clean up.")
        return

    main_worktree, *linked = worktrees

    print(f"Main worktree (kept): {main_worktree['path']}")
    print(f"\nThe following {len(linked)} worktree(s) will be DELETED,")
    print("along with their branches:\n")
    for wt in linked:
        branch = wt.get("branch", "<detached>")
        print(f"  {wt['path']}  [{branch}]")

    if not args.yes:
        reply = input("\nDelete all of the above? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return

    for wt in linked:
        path = wt["path"]
        branch = wt.get("branch")

        run(["git", "worktree", "unlock", path], repo_dir)

        result = run(["git", "worktree", "remove", "--force", path], repo_dir)
        if result.returncode != 0:
            print(
                f"Failed to remove worktree {path}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            continue
        print(f"Removed worktree {path}")

        if branch:
            result = run(["git", "branch", "-D", branch], repo_dir)
            if result.returncode != 0:
                print(
                    f"Failed to delete branch {branch}: {result.stderr.strip()}",
                    file=sys.stderr,
                )
            else:
                print(f"Deleted branch {branch}")

    result = run(["git", "remote", "prune", "origin"], repo_dir)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)


if __name__ == "__main__":
    main()
