# `adversarial_report.py` terminology cleanup: "diagnostic(s)" -> "analysis"

## Context

`adversarial_report_docs_plan.md` (archived) rewrote this module's framing
around "does it hide c or erase it?" and settled on "diagnostic(s)" as the
vocabulary throughout (module docstring, `parse_args`'s `description=`,
`_build_report`'s header line, and assorted comments). That hidden-vs-erased
dichotomy is itself now retired: the project's actual success condition is
simply *task learned + logreg probe fails* (no separate hidden/erased
taxonomy), per the current project-state understanding.

Re-examined on its own merits (not just for internal consistency with text
that turns out to already be stale): "diagnostic" implies a single targeted
pass/fail test. This script instead produces a per-layer, per-metric,
per-held-out-pair breakdown (task fidelity, DoM/logreg/LDA probe gap, held-out
c recovery, linear-y reconstruction) meant to explain *what actually
happened* in a run -- broader than a yes/no check. "Analysis" fits that
better.

This plan covers the terminology only. It does not touch the retired
hidden-vs-erased framing's substance (already tracked as stale in project
memory, not this codebase) beyond removing the phrase where it appears as
scaffolding for the term "diagnostic."

## Items

1. Module docstring (`adversarial_report.py` lines 1-22): reframe opening
   line away from "Diagnostics for ... does it hide c or erase it?" to
   describe what the script does (task fidelity / probe-strength gap /
   held-out c recovery / linear-y reconstruction) without asserting a
   hidden-vs-erased dichotomy as the goal. Update "Two optional deep-dive
   diagnostics" similarly.
2. `parse_args`'s `description="Adversarial diagnostics: hidden vs. erased
   c."` -> drop the hidden-vs-erased framing, rename to analysis.
3. `_build_report`'s header line `f"# Adversarial diagnostics — tag=..."` ->
   `f"# Adversarial analysis — tag=..."`.
4. `DiagnosticsResult` dataclass (introduced by
   `adversarial_report_main_decomposition_plan.md`) -> `AnalysisResult`,
   including its use sites in `_run_diagnostics`/`_make_plots`/`main`.
   Consider whether `_run_diagnostics` itself should rename too (e.g.
   `_run_analysis`) for consistency, though the function name is less load
   bearing than the report-facing text in items 1-3.
5. Sweep remaining "diagnostic"/"diagnostics" occurrences in comments (e.g.
   `_linear_y_reconstruction`'s docstring: "this diagnostic asks whether...")
   and reword case by case.

## Steps

1. Grep `adversarial_report.py` for `[Dd]iagnos` to enumerate every
   occurrence before editing (don't rely on the list in Items above being
   exhaustive).
2. Apply the renames from Items 1-5.
3. Check `plans/`, `CLAUDE.md`, and other `.md` docs for stale references to
   "diagnostics" describing this script, in case any need updating too.

## Verification

- `black --check adversarial_report.py`.
- Re-read the rewritten module docstring and `parse_args` description standing
  alone (not next to this plan) to confirm they read as accurate framing, not
  just "search-and-replace of one word."
- No behavior change expected; a smoke run (`--tag <existing run> --ckpt
  last`) is optional but not required to catch anything -- this is a pure
  text/naming change.

## Risks / caveats

- Purely cosmetic/naming; the main risk is an incomplete sweep leaving mixed
  terminology in one file. Step 1's grep-first approach is there to catch
  that.
