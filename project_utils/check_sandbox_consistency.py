"""Audit a Claude Code .claude/settings.local.json for sandbox self-consistency.

Two layers protect the filesystem: `permissions.allow/deny` (the prompt-suppression
layer, Read()/Edit() rules) and `sandbox.filesystem.{allow,deny}{Read,Write}` (the
OS-enforced boundary). They should be kept in sync by hand; this script flags where
they've drifted apart, where a declared path uses the wrong absolute/relative form
for its git-tracked status, and reports which category (private / read-only / fully
writable) each declared path falls into, based on the filesystem layer alone.

See /work/ml/claude_sandbox_notes/the_notes/SANDBOX.md for the design this checks
against, in particular the "Implementation" section, whose relative/absolute path
rules (tied to git-tracked status) are the main thing this script checks beyond
raw allow/deny drift between the two layers.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "project_dir",
        nargs="?",
        default=".",
        help="Project root containing .claude/settings.local.json (default: cwd)",
    )
    return p.parse_args()


def load_settings(project_root: Path) -> dict:
    settings_path = project_root / ".claude" / "settings.local.json"
    with open(settings_path) as f:
        return json.load(f)


def strip_trailing_glob(path: str) -> str:
    """Drop a trailing full-wildcard component (`/**`, `/*`, or a trailing slash)
    to get a base directory. Deliberately does NOT strip a partial file-level glob
    like `/*.py` or `/*.extension` -- those are distinct single-file declarations,
    not directory-level ones, and stripping them down to their parent directory
    would make different globs (`*.py` vs `*.extension`) collide onto the same
    base and falsely register as "the same path" elsewhere in this script.
    """
    parts = path.split("/")
    while parts and parts[-1] in ("", "**", "*"):
        parts.pop()
    return "/".join(parts) or "/"


def normalize_path(raw: str, project_root: Path) -> tuple[str, bool]:
    """Return (absolute base directory, had_multilevel_glob)."""
    s = raw
    had_starstar = "**" in s
    s = re.sub(r"^\*\*/", "", s)  # leading **/ (e.g. worktree .git pattern)
    if s.startswith("//"):
        # Claude Code's escaped-absolute form: //home/... -> /home/...
        abs_path = "/" + s.lstrip("/")
    elif s.startswith("~"):
        abs_path = str(Path(s).expanduser())
    elif s.startswith("./"):
        abs_path = str(project_root / s[2:])
    elif s.startswith("/"):
        abs_path = s
    else:
        abs_path = str(project_root / s)
    base = strip_trailing_glob(abs_path)
    return base, had_starstar


def is_absolute_form(raw: str) -> bool:
    """True if the raw entry is written as an absolute or home-relative path
    (`/...`, `//...`, `~...`) rather than a `./`-relative one."""
    return raw.startswith("/") or raw.startswith("~")


def is_relative_form(raw: str) -> bool:
    return raw.startswith("./")


PERM_RE = re.compile(r"^(Read|Edit)\((.+)\)$")


def extract_permission_entries(entries: list[str], project_root: Path) -> dict:
    """tool -> list of (base_dir, had_starstar, raw)."""
    out = {"Read": [], "Edit": []}
    for e in entries:
        m = PERM_RE.match(e)
        if not m:
            continue
        tool, raw_path = m.groups()
        base, starstar = normalize_path(raw_path, project_root)
        out[tool].append((base, starstar, e))
    return out


def extract_fs_entries(
    paths: list[str], project_root: Path
) -> list[tuple[str, bool, str]]:
    return [(*normalize_path(p, project_root), p) for p in paths]


def find_match(base_dir: str, candidates: list[tuple[str, bool, str]]):
    """Find a candidate covering base_dir. Three tiers, in order:

    1. Exact base-dir equality.
    2. Ancestor coverage: a candidate whose pattern had `**` (e.g.
       `Read(//project/**)`) and whose base dir is an ancestor-or-equal of
       base_dir -- a genuine "everything under this dir" declaration, safe to
       treat as covering, unlike a bare directory name or a single-level file
       glob (`*.py`) that merely glob-stripped down to some base dir by
       coincidence.
    3. Basename-only match for patterns that used `**` in a way that doesn't
       resolve to one fixed absolute directory (e.g. `**/.git/**`).

    Deliberately NOT a general parent/child (prefix) match: two paths merely
    sharing an ancestor (e.g. both living under the project root) are not "the
    same resource" -- only an explicit `**` marks broad, intentional coverage.

    Returns (raw, match_kind), match_kind in {None, "exact", "covered", "basename"}.
    """
    for cand_base, cand_starstar, raw in candidates:
        if cand_base == base_dir:
            return raw, "exact"
    for cand_base, cand_starstar, raw in candidates:
        if cand_starstar and (
            base_dir == cand_base or base_dir.startswith(cand_base.rstrip("/") + "/")
        ):
            return raw, "covered"
    target_name = Path(base_dir).name
    for cand_base, cand_starstar, raw in candidates:
        if cand_starstar and Path(cand_base).name == target_name:
            return raw, "basename"
    return None, None


def audit_consistency(settings: dict, project_root: Path) -> list[str]:
    perms = settings.get("permissions", {})
    fs = settings.get("sandbox", {}).get("filesystem", {})

    perm_allow = extract_permission_entries(perms.get("allow", []), project_root)
    perm_deny = extract_permission_entries(perms.get("deny", []), project_root)

    fs_allow_read = extract_fs_entries(fs.get("allowRead", []), project_root)
    fs_deny_read = extract_fs_entries(fs.get("denyRead", []), project_root)
    fs_allow_write = extract_fs_entries(fs.get("allowWrite", []), project_root)
    fs_deny_write = extract_fs_entries(fs.get("denyWrite", []), project_root)

    findings = []

    def check_direction(
        label,
        fs_entries,
        perm_same_dir,
        perm_opposite_dir,
        opposite_label,
        is_deny_direction,
    ):
        def fuzzy_note(match_kind, matched_raw):
            if match_kind == "covered":
                return (
                    f"FUZZY-OK  {label}: sandbox `{raw}` is covered by permissions "
                    f"`{matched_raw}`, a broader `**` grant over an ancestor directory"
                )
            return (
                f"FUZZY-OK  {label}: sandbox `{raw}` matched permissions `{matched_raw}` "
                f"only by basename (multi-level ** pattern)"
            )

        for base, starstar, raw in fs_entries:
            same_match, same_kind = find_match(base, perm_same_dir)
            if is_deny_direction and same_match is not None:
                # permissions.deny always wins outright, regardless of how broad a
                # competing allow elsewhere is -- a same-direction deny match (at
                # any tier) already justifies the sandbox's denial, full stop.
                if same_kind != "exact":
                    findings.append(fuzzy_note(same_kind, same_match))
                continue
            opp_match, _ = find_match(base, perm_opposite_dir)
            if opp_match is not None:
                findings.append(
                    f"CONFLICT  {label}: sandbox `{raw}` but permissions has "
                    f"{opposite_label} entry `{opp_match}` for the same path"
                )
            elif same_match is None:
                findings.append(
                    f"MISSING   {label}: sandbox `{raw}` has no matching permissions entry"
                )
            elif same_kind != "exact":
                findings.append(fuzzy_note(same_kind, same_match))

    check_direction(
        "allowRead",
        fs_allow_read,
        perm_allow["Read"],
        perm_deny["Read"],
        "deny Read",
        False,
    )
    check_direction(
        "denyRead",
        fs_deny_read,
        perm_deny["Read"],
        perm_allow["Read"],
        "allow Read",
        True,
    )
    check_direction(
        "allowWrite",
        fs_allow_write,
        perm_allow["Edit"],
        perm_deny["Edit"],
        "deny Edit",
        False,
    )
    check_direction(
        "denyWrite",
        fs_deny_write,
        perm_deny["Edit"],
        perm_allow["Edit"],
        "allow Edit",
        True,
    )

    # Reverse direction: permissions entries under the project root with no fs counterpart at all.
    project_root_str = str(project_root)
    all_fs = fs_allow_read + fs_deny_read + fs_allow_write + fs_deny_write

    def check_reverse(label, perm_entries, fs_same_kind):
        for base, starstar, raw in perm_entries:
            if not is_broad_deny(raw):
                continue  # file-level glob (*.tmp.py, *.extension) -- sandbox.filesystem
                # can only declare directories, so these have no possible fs counterpart
            if base == project_root_str:
                continue  # whole-root grant; sandbox defaults (read-allow everywhere,
                # write-allow inside the root) already cover it, no fs entry expected
            if not base.startswith(project_root_str) and not any(
                base == c[0] or base.startswith(c[0] + "/") for c in all_fs
            ):
                continue  # outside project root and not near any declared fs path; skip (venv, notes dir etc. covered above)
            match, _ = find_match(base, fs_same_kind)
            if match is None:
                findings.append(
                    f"ORPHAN    permissions {label} `{raw}` has no matching sandbox.filesystem entry"
                )

    check_reverse("allow Read", perm_allow["Read"], fs_allow_read + fs_deny_read)
    check_reverse("deny Read", perm_deny["Read"], fs_allow_read + fs_deny_read)
    check_reverse("allow Edit", perm_allow["Edit"], fs_allow_write + fs_deny_write)
    check_reverse("deny Edit", perm_deny["Edit"], fs_allow_write + fs_deny_write)

    return findings


def check_claude_dir_requirement(settings: dict, project_root: Path) -> list[str]:
    """.claude/ must have a literal absolute-path entry in BOTH allowRead and allowWrite.

    Relative paths in sandbox.filesystem resolve per-worktree, not to one fixed
    project root (see SANDBOX.md gotchas) -- a `./.claude` entry doesn't reliably
    cover .claude/ access when running from inside a worktree. Only an absolute
    entry does, and Bash needs it in worktrees.
    """
    fs = settings.get("sandbox", {}).get("filesystem", {})
    required = str(project_root / ".claude")
    findings = []
    if required not in fs.get("allowRead", []):
        findings.append(
            f"MISSING   sandbox.filesystem.allowRead lacks the absolute entry `{required}`"
        )
    if required not in fs.get("allowWrite", []):
        findings.append(
            f"MISSING   sandbox.filesystem.allowWrite lacks the absolute entry `{required}`"
        )
    return findings


def categorize(settings: dict, project_root: Path) -> dict:
    """Bucket every path declared in sandbox.filesystem into one of three categories,
    resolved using that block alone: allow beats deny (see SANDBOX.md's
    'act as if allow trumps deny' operating rule); reads default-allow everywhere;
    writes default-allow only inside the project root.
    """
    fs = settings.get("sandbox", {}).get("filesystem", {})
    allow_read = {normalize_path(p, project_root)[0] for p in fs.get("allowRead", [])}
    deny_read = {normalize_path(p, project_root)[0] for p in fs.get("denyRead", [])}
    allow_write = {normalize_path(p, project_root)[0] for p in fs.get("allowWrite", [])}
    deny_write = {normalize_path(p, project_root)[0] for p in fs.get("denyWrite", [])}

    all_paths = allow_read | deny_read | allow_write | deny_write
    project_root_str = str(project_root)

    categories = {"1_private": [], "2_read_only": [], "3_read_write": []}
    for path in sorted(all_paths):
        if path in allow_read:
            read = "allow"
        elif path in deny_read:
            read = "deny"
        else:
            read = "allow"  # default

        if path in allow_write:
            write = "allow"
        elif path in deny_write:
            write = "deny"
        else:
            write = "allow" if path.startswith(project_root_str) else "deny"  # default

        if read == "deny":
            categories["1_private"].append(path)
        elif write == "deny":
            categories["2_read_only"].append(path)
        else:
            categories["3_read_write"].append(path)
    return categories


def is_broad_deny(raw: str) -> bool:
    """True if `raw` plausibly covers a whole subtree (bare dir, or trailing `/**`),
    rather than a filename glob -- either single-level (`*.py`, `*.extension`) or
    recursive (`**/*.tmp.py`) -- which only matches specific filenames, not
    everything beneath a directory. Judged by the *last* path segment: `**/*.tmp.py`
    ends in a filename pattern despite containing `**` earlier in the path, so it's
    narrow, not broad.
    """
    last_segment = raw.rstrip("/").split("/")[-1]
    if last_segment == "**":
        return True
    return "*" not in last_segment


def check_permission_deny_shadowing(settings: dict, project_root: Path) -> list[str]:
    """permissions.{allow,deny} use deny-always-wins semantics (unlike
    sandbox.filesystem, where allow trumps deny per SANDBOX.md's operating rule).
    A broad `deny` entry (e.g. `Read(~/**)`) silently blocks Read/Edit tool calls
    to any path underneath it, even if a narrower `allow` entry also matches and
    even if sandbox.filesystem explicitly allows it at the OS level. Flag every
    sandbox.filesystem allowRead/allowWrite path that falls under (or equals) a
    permissions.deny Read/Edit entry's directory.

    Only entries that plausibly cover a whole subtree count as "shadowing" here --
    single-level file globs like `Read(./*.extension)` are excluded (best effort;
    false negatives on those are fine, false positives on unrelated directories
    are not).
    """
    perms = settings.get("permissions", {})
    fs = settings.get("sandbox", {}).get("filesystem", {})
    perm_deny = extract_permission_entries(perms.get("deny", []), project_root)
    perm_deny = {
        tool: [e for e in entries if is_broad_deny(e[2])]
        for tool, entries in perm_deny.items()
    }
    fs_allow_read = extract_fs_entries(fs.get("allowRead", []), project_root)
    fs_allow_write = extract_fs_entries(fs.get("allowWrite", []), project_root)

    findings = []

    def check(label, fs_entries, deny_entries, tool):
        for base, _, raw in fs_entries:
            for deny_base, _, deny_raw in deny_entries:
                if base == deny_base or base.startswith(deny_base.rstrip("/") + "/"):
                    findings.append(
                        f"SHADOWED  sandbox.{label} `{raw}` is covered by permissions "
                        f"deny `{deny_raw}` -- {tool} tool calls will be refused here "
                        f"even though sandbox.filesystem allows it"
                    )

    check("allowRead", fs_allow_read, perm_deny["Read"], "Read")
    check("allowWrite", fs_allow_write, perm_deny["Edit"], "Edit")
    return findings


def check_broad_deny_sweeps_harness_dirs(
    settings: dict, project_root: Path
) -> list[str]:
    """A broad denyWrite/deny-Edit glob that sweeps in .claude/** or .git/** breaks
    bwrap and kills bash for the session (SANDBOX.md gotchas) -- these paths are
    harness-virtualized, unlike ordinary repo content. Flag any such entry; a
    narrow (non-`is_broad_deny`) glob is fine and not flagged.
    """
    perms = settings.get("permissions", {})
    fs = settings.get("sandbox", {}).get("filesystem", {})
    protected = {str(project_root / ".claude"), str(project_root / ".git")}

    findings = []

    def check(label, raw_entries, extractor):
        for raw in raw_entries:
            if not is_broad_deny(raw):
                continue
            base = extractor(raw)
            for p in protected:
                if base == p or p.startswith(base.rstrip("/") + "/"):
                    findings.append(
                        f"DANGEROUS {label} `{raw}` is broad enough to sweep in `{p}` "
                        f"-- this is harness-virtualized and a broad deny here can break "
                        f"bash for the whole session"
                    )

    check(
        "sandbox.denyWrite",
        fs.get("denyWrite", []),
        lambda raw: normalize_path(raw, project_root)[0],
    )
    deny_edit_raw = [
        m.group(2)
        for e in perms.get("deny", [])
        if (m := PERM_RE.match(e)) and m.group(1) == "Edit"
    ]
    check(
        "permissions.deny Edit",
        deny_edit_raw,
        lambda raw: normalize_path(raw, project_root)[0],
    )
    return findings


def check_no_broad_project_root_allow(settings: dict, project_root: Path) -> list[str]:
    """'Do not use a global allowRead on the project directory' (SANDBOX.md gotchas)
    -- a project-root-wide allow overrides narrower denies underneath it, since
    `sandbox.filesystem`'s operating rule is allow-trumps-deny.

    This is specific to `sandbox.filesystem`: in the `permissions` layer, deny
    always wins outright regardless of how broad a competing allow is (see
    SANDBOX.md's "Available sandboxing tooling" section), so a project-root-wide
    `permissions.allow` entry (e.g. the intended `Read(project/**)` granting
    broad repo read) can never actually override a `permissions.deny` -- it is
    not the hazard this gotcha warns about, and isn't checked here.

    Only entries broad enough to plausibly mean "this whole subtree" are
    checked (`is_broad_deny`) -- a single-level file glob like `*.py` that
    happens to glob-strip down to the project root isn't a project-root-wide
    grant.
    """
    fs = settings.get("sandbox", {}).get("filesystem", {})
    project_root_str = str(project_root)
    findings = []

    def check(label, raw_entries):
        for raw in raw_entries:
            if not is_broad_deny(raw):
                continue
            base = normalize_path(raw, project_root)[0]
            if base == project_root_str:
                findings.append(
                    f"BROAD-ALLOW {label} `{raw}` resolves to the whole project root -- "
                    f"this overrides every narrower deny underneath it (allow trumps deny)"
                )

    check("sandbox.allowRead", fs.get("allowRead", []))
    check("sandbox.allowWrite", fs.get("allowWrite", []))
    return findings


def check_bare_bash_allow(settings: dict) -> list[str]:
    """'Bare Bash/Bash(*) don't suppress prompts' (SANDBOX.md gotchas) -- Claude Code
    special-cases and skips them. Flag their presence as dead weight / a
    misunderstanding of what's actually suppressing prompts.
    """
    perms = settings.get("permissions", {})
    findings = []
    for e in perms.get("allow", []):
        if e in ("Bash", "Bash(*)"):
            findings.append(
                f"DEAD      permissions.allow `{e}` -- bare Bash/Bash(*) is special-cased "
                f"and skipped, it does not suppress prompts"
            )
    return findings


def git_status_of(path: Path, project_root: Path) -> str:
    """Classify `path` (need not exist on disk) relative to the project's git repo:
    'outside_repo', 'tracked' (path itself or something under it is tracked),
    'gitignored', or 'untracked' (inside repo, not tracked, not gitignored --
    e.g. a newly-created dir that hasn't been git-added or .gitignore'd yet).
    """
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        return "outside_repo"
    rel_str = str(rel) if str(rel) != "." else "."
    if rel_str == ".":
        return "tracked"  # the project root itself

    def run(args):
        return subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True,
            text=True,
        )

    if run(["check-ignore", "-q", rel_str]).returncode == 0:
        return "gitignored"
    if run(["ls-files", "--error-unmatch", rel_str]).returncode == 0:
        return "tracked"
    listing = run(["ls-files", rel_str])
    if listing.stdout.strip():
        return "tracked"
    return "untracked"


def check_path_form(settings: dict, project_root: Path) -> list[str]:
    """SANDBOX.md's Implementation section ties path *form* (absolute vs `./`-relative)
    to a directory's git-tracked status, separately for deny-write (category 2) and
    allow-write (category 3) entries:

    - denyWrite / deny-Edit on a git-tracked dir -> relative (`./dir`) form.
    - denyWrite / deny-Edit on a gitignored/untracked in-repo dir -> absolute form
      (relative paths resolve per-worktree, not to one fixed project root).
    - denyWrite / deny-Edit on a path outside the repo -> redundant, writes there
      are already denied by default.
    - allowWrite / allow-Edit on a git-tracked in-root dir -> redundant, already
      read/write by default inside the project root.
    - allowWrite / allow-Edit on a gitignored/untracked-in-repo/outside-repo dir ->
      absolute form required.

    This is deliberately per-raw-entry (not per resolved category from `categorize`):
    the design ties the rule to which list an entry was placed in, not to the
    net-resolved permission.

    Skips: single-level file globs (`*.py`) that merely glob-strip down to some
    base dir by coincidence, not a real directory-level declaration
    (`is_broad_deny`); and `.claude`/`.git`, which are harness-virtualized and
    have their own dedicated requirements (`check_claude_dir_requirement`,
    `check_broad_deny_sweeps_harness_dirs`) rather than the ordinary
    tracked/gitignored rule.
    """
    perms = settings.get("permissions", {})
    fs = settings.get("sandbox", {}).get("filesystem", {})
    special_dirs = {str(project_root / ".claude"), str(project_root / ".git")}
    findings = []

    def check_deny_write(label, raw_entries, extractor):
        for raw in raw_entries:
            if not is_broad_deny(raw):
                continue
            base = extractor(raw)
            if base in special_dirs:
                continue
            status = git_status_of(Path(base), project_root)
            if status == "tracked":
                if not is_relative_form(raw):
                    findings.append(
                        f"WRONGFORM {label} `{raw}` targets a git-tracked directory -- "
                        f"should use a relative (`./`) path, not absolute"
                    )
            elif status in ("gitignored", "untracked"):
                if not is_absolute_form(raw):
                    findings.append(
                        f"WRONGFORM {label} `{raw}` targets a gitignored/untracked "
                        f"directory -- should use an absolute path (relative paths "
                        f"resolve per-worktree), not `./`-relative"
                    )
            elif status == "outside_repo":
                findings.append(
                    f"REDUNDANT {label} `{raw}` targets a path outside the repo -- "
                    f"writes there are already denied by default"
                )

    def check_allow_write(label, raw_entries, extractor):
        for raw in raw_entries:
            if not is_broad_deny(raw):
                continue
            base = extractor(raw)
            if base in special_dirs:
                continue
            status = git_status_of(Path(base), project_root)
            if status == "tracked":
                findings.append(
                    f"REDUNDANT {label} `{raw}` targets a git-tracked directory inside "
                    f"the project root -- already read/write by default, no entry needed"
                )
            elif status in ("gitignored", "untracked", "outside_repo"):
                if not is_absolute_form(raw):
                    findings.append(
                        f"WRONGFORM {label} `{raw}` targets a gitignored/untracked/"
                        f"outside-repo directory -- should use an absolute path, not "
                        f"`./`-relative"
                    )

    deny_edit_raw = [
        m.group(2)
        for e in perms.get("deny", [])
        if (m := PERM_RE.match(e)) and m.group(1) == "Edit"
    ]
    allow_edit_raw = [
        m.group(2)
        for e in perms.get("allow", [])
        if (m := PERM_RE.match(e)) and m.group(1) == "Edit"
    ]

    check_deny_write(
        "sandbox.denyWrite",
        fs.get("denyWrite", []),
        lambda raw: normalize_path(raw, project_root)[0],
    )
    check_deny_write(
        "permissions.deny Edit",
        deny_edit_raw,
        lambda raw: normalize_path(raw, project_root)[0],
    )
    check_allow_write(
        "sandbox.allowWrite",
        fs.get("allowWrite", []),
        lambda raw: normalize_path(raw, project_root)[0],
    )
    check_allow_write(
        "permissions.allow Edit",
        allow_edit_raw,
        lambda raw: normalize_path(raw, project_root)[0],
    )
    return findings


def check_global_deny_rules(settings: dict, project_root: Path) -> list[str]:
    """SANDBOX.md: for all projects, `~/` and `/var/log` need deny-read set via
    BOTH `sandbox.filesystem.denyRead` AND `permissions.deny` Read *and* Edit rules,
    with absolute paths. Edit is easy to forget since it's not needed for the
    sandbox.filesystem layer (denyRead alone blocks writes there too, per the
    denyRead/denyWrite mechanics table) but permissions.deny has no such
    read-implies-write shortcut.
    """
    perms = settings.get("permissions", {})
    fs = settings.get("sandbox", {}).get("filesystem", {})
    perm_deny = extract_permission_entries(perms.get("deny", []), project_root)
    fs_deny_read = extract_fs_entries(fs.get("denyRead", []), project_root)

    findings = []
    for label, target in [
        ("~/", str(Path("~").expanduser())),
        ("/var/log", "/var/log"),
    ]:
        if find_match(target, fs_deny_read)[0] is None:
            findings.append(
                f"MISSING   sandbox.filesystem.denyRead lacks an entry for {label}"
            )
        if find_match(target, perm_deny["Read"])[0] is None:
            findings.append(
                f"MISSING   permissions.deny lacks a Read(...) entry for {label}"
            )
        if find_match(target, perm_deny["Edit"])[0] is None:
            findings.append(
                f"MISSING   permissions.deny lacks an Edit(...) entry for {label} "
                f"(Read-only doesn't block Write/NotebookEdit -- see SANDBOX.md)"
            )
    return findings


def print_section(title: str, findings: list[str]):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    if not findings:
        print("\nOK.")
    else:
        print(f"\n{len(findings)} finding(s):\n")
        for f in findings:
            print(f"  {f}")


def main():
    args = parse_args()
    project_root = Path(args.project_dir).resolve()
    settings = load_settings(project_root)

    print(f"Auditing {project_root}/.claude/settings.local.json\n")

    print("=" * 70)
    print("CATEGORIES (resolved from sandbox.filesystem only)")
    print("=" * 70)
    categories = categorize(settings, project_root)
    labels = {
        "1_private": "1. Private to user (no agent read/write)",
        "2_read_only": "2. Agent-readable, not writable",
        "3_read_write": "3. Fully agent read/write",
    }
    for key, label in labels.items():
        print(f"\n{label}:")
        for path in categories[key]:
            print(f"  {path}")
        if not categories[key]:
            print("  (none)")

    print_section(
        "CONSISTENCY CHECK (permissions.{allow,deny} vs sandbox.filesystem)",
        audit_consistency(settings, project_root),
    )
    print_section(
        "REQUIRED ABSOLUTE ENTRIES (.claude/ worktree access)",
        check_claude_dir_requirement(settings, project_root),
    )
    print_section(
        "GLOBAL DENY-READ RULES (~/ and /var/log, both layers, Read+Edit)",
        check_global_deny_rules(settings, project_root),
    )
    print_section(
        "PATH FORM (absolute vs ./relative, tied to git-tracked status)",
        check_path_form(settings, project_root),
    )
    print_section(
        "PERMISSION-LAYER SHADOWING (deny wins outright, unlike sandbox.filesystem)",
        check_permission_deny_shadowing(settings, project_root),
    )
    print_section(
        "BROAD DENY SWEEPING .claude/ OR .git/ (harness-virtualized, breaks bash)",
        check_broad_deny_sweeps_harness_dirs(settings, project_root),
    )
    print_section(
        "BROAD ALLOW ON PROJECT ROOT (overrides every narrower deny beneath it)",
        check_no_broad_project_root_allow(settings, project_root),
    )
    print_section(
        "DEAD ENTRIES (don't do what they look like they do)",
        check_bare_bash_allow(settings),
    )


if __name__ == "__main__":
    main()
