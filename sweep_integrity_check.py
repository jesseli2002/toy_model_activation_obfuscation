"""Integrity checker for sweep-style training run directories.

Verifies that a run directory looks fully complete and internally
consistent: training reached the final iteration, checkpoints are all
present and correctly linked, and configs match a shared template except
for the per-run swept knobs (lam, seed). Intended for auditing runs
restored from backup, where every directory should either be complete or
be an empty stub (e.g. a run tag that was allocated but never started).
Stubs are reported separately from real failures so they can be reviewed
by hand instead of treated as corruption.
"""

import argparse
import fnmatch
import json
import os
import re


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument(
        "--pattern", default="sweep7_*", help="glob for run tags under runs-dir"
    )
    p.add_argument("--final-iter", type=int, default=99999)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    p.add_argument(
        "--tag-re",
        default=r"^sweep7_lam(?P<lam>[0-9.eE+-]+)_tr(?P<seed>\d+)$",
        help="regex with named groups 'lam' and 'seed', used to check config.json/"
        "input_config.json against the values encoded in the run's tag",
    )
    return p.parse_args()


def is_stub(run_dir: str) -> bool:
    """A stub is a run directory with no files in it at all."""
    if not os.path.isdir(run_dir):
        return True
    for _root, _dirs, files in os.walk(run_dir):
        if files:
            return False
    return True


def check_history(run_dir: str, final_iter: int, errors: list[str]) -> None:
    hist_path = os.path.join(run_dir, "logs", "history.jsonl")
    if not os.path.isfile(hist_path):
        errors.append("logs/history.jsonl missing")
        return
    last = None
    with open(hist_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        errors.append("logs/history.jsonl is empty")
        return
    try:
        rec = json.loads(last)
    except json.JSONDecodeError as e:
        errors.append(f"logs/history.jsonl last line is not valid JSON: {e}")
        return
    if rec.get("iter") != final_iter:
        errors.append(
            f"logs/history.jsonl last iter is {rec.get('iter')!r}, expected {final_iter}"
        )


def check_checkpoints(
    run_dir: str, final_iter: int, ckpt_interval: int, errors: list[str]
) -> None:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        errors.append("checkpoints/ missing")
        return

    expected_target = f"iter_{final_iter}.pt"
    last_link = os.path.join(ckpt_dir, "last.pt")
    if not os.path.islink(last_link):
        errors.append("checkpoints/last.pt missing or not a symlink")
    else:
        target = os.readlink(last_link)
        if target != expected_target:
            errors.append(
                f"checkpoints/last.pt -> {target!r}, expected {expected_target!r}"
            )

    final_ckpt = os.path.join(ckpt_dir, expected_target)
    if not os.path.isfile(final_ckpt):
        errors.append(f"checkpoints/{expected_target} missing")

    expected_iters = list(range(ckpt_interval, final_iter, ckpt_interval)) + [
        final_iter
    ]
    missing = [
        i
        for i in expected_iters
        if not os.path.isfile(os.path.join(ckpt_dir, f"iter_{i}.pt"))
    ]
    if missing:
        errors.append(f"missing checkpoint(s) for iter(s): {missing}")


def _diff_against_template(
    actual: dict, template: dict, path: str, ignore: set[str], errors: list[str]
) -> None:
    """Recursively compares `actual` to `template`, reporting any
    difference not covered by `ignore` (dotted paths relative to the
    top-level call, e.g. "adversarial.lam")."""
    for key, tval in template.items():
        rel = key if path == "" else f"{path}.{key}"
        if rel in ignore:
            continue
        if key not in actual:
            errors.append(f"{path or '<root>'} missing key {key!r}")
        elif isinstance(tval, dict):
            _diff_against_template(actual[key], tval, rel, ignore, errors)
        elif actual[key] != tval:
            errors.append(f"{rel} = {actual[key]!r}, expected {tval!r} (template)")
    for key in actual:
        if (
            key not in template
            and (key if path == "" else f"{path}.{key}") not in ignore
        ):
            errors.append(f"{path or '<root>'} has unexpected key {key!r}")


def check_configs(
    run_dir: str,
    tag: str,
    tag_re: re.Pattern,
    template_config: dict,
    template_input_config: dict,
    errors: list[str],
) -> None:
    cfg_path = os.path.join(run_dir, "config.json")
    input_path = os.path.join(run_dir, "input_config.json")
    cfg = input_cfg = None
    if not os.path.isfile(cfg_path):
        errors.append("config.json missing")
    else:
        cfg = json.load(open(cfg_path))
    if not os.path.isfile(input_path):
        errors.append("input_config.json missing")
    else:
        input_cfg = json.load(open(input_path))
    if cfg is None or input_cfg is None:
        return

    m = tag_re.match(tag)
    if m is None:
        errors.append(f"tag {tag!r} doesn't match --tag-re; skipping lam/seed check")
    else:
        expected_lam = float(m.group("lam"))
        expected_seed = int(m.group("seed"))
        if cfg.get("adversarial", {}).get("lam") != expected_lam:
            errors.append(
                f"config.json adversarial.lam={cfg.get('adversarial', {}).get('lam')!r}, expected {expected_lam!r} from tag"
            )
        if cfg.get("seed") != expected_seed:
            errors.append(
                f"config.json seed={cfg.get('seed')!r}, expected {expected_seed!r} from tag"
            )
        if input_cfg.get("lam") != expected_lam:
            errors.append(
                f"input_config.json lam={input_cfg.get('lam')!r}, expected {expected_lam!r} from tag"
            )

    _diff_against_template(
        cfg, template_config, "", {"seed", "adversarial.lam"}, errors
    )
    _diff_against_template(input_cfg, template_input_config, "", {"lam"}, errors)


def check_run(
    run_dir: str,
    tag: str,
    args: argparse.Namespace,
    tag_re: re.Pattern,
    template_config: dict,
    template_input_config: dict,
) -> list[str]:
    errors: list[str] = []
    check_history(run_dir, args.final_iter, errors)
    check_checkpoints(run_dir, args.final_iter, args.ckpt_interval, errors)
    check_configs(run_dir, tag, tag_re, template_config, template_input_config, errors)
    return errors


def main() -> int:
    args = parse_args()
    tag_re = re.compile(args.tag_re)

    all_tags = sorted(
        d for d in os.listdir(args.runs_dir) if fnmatch.fnmatch(d, args.pattern)
    )
    if not all_tags:
        print(f"No run directories under {args.runs_dir!r} matching {args.pattern!r}")
        return 1

    stub_tags = [t for t in all_tags if is_stub(os.path.join(args.runs_dir, t))]
    real_tags = [t for t in all_tags if t not in stub_tags]

    if not real_tags:
        print("Every matched directory is a stub; nothing to use as a config template.")
        return 1
    template_tag = real_tags[0]
    template_config = json.load(
        open(os.path.join(args.runs_dir, template_tag, "config.json"))
    )
    template_input_config = json.load(
        open(os.path.join(args.runs_dir, template_tag, "input_config.json"))
    )
    print(f"Using {template_tag!r} as the config template.\n")

    n_ok = 0
    failures = {}
    for tag in real_tags:
        run_dir = os.path.join(args.runs_dir, tag)
        errors = check_run(
            run_dir, tag, args, tag_re, template_config, template_input_config
        )
        if errors:
            failures[tag] = errors
        else:
            n_ok += 1

    for tag, errors in failures.items():
        print(f"FAIL {tag}")
        for e in errors:
            print(f"    - {e}")

    print()
    print(f"{len(all_tags)} run(s) matched {args.pattern!r}")
    print(f"  {n_ok} OK")
    print(f"  {len(failures)} FAILED (see above)")
    if stub_tags:
        print(
            f"  {len(stub_tags)} STUB (empty dir, flagged for manual review, not treated as failure):"
        )
        for tag in stub_tags:
            print(f"    - {tag}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
