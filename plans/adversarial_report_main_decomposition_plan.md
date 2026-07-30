# Decompose `adversarial_report.py`'s `main()` into phases

## Context

`main()` is ~170 lines doing five things in sequence with interleaved
conditionals: setup (device/checkpoint/config derivation/RNG/backend), phase
1 compute (task fidelity, probe-gap metrics, optionally held-out-pair
metrics and linear-y R²), phase 2 report assembly + file write (already
factored into `_build_report`, just not phase-isolated at the `main()`
level), and phase 3 plotting (~10 plot calls gated by `args.detailed`/
`args.steer`). This plan splits phases 1 and 3 out into their own functions,
so `main()` becomes a short, linear sequence of named steps.

**Land this one last** among the `adversarial_report_*_plan.md` set --
`adversarial_report_save_plot_plan.md`,
`adversarial_report_dom_probe_refactor_plan.md`, and
`adversarial_report_checkpoint_access_plan.md` all touch code this plan
moves wholesale; landing them first means this plan moves already-cleaned
code instead of needing rework after the fact.

**Explicitly preserved, not changed:** the visible `b_dom = -pi["midpoint"]`
sign-flip inline at its `LinearBoundary(pi["w_dom"], b_dom)` call site (see
`adversarial_report_dom_probe_refactor_plan.md`'s Context for why it's
deliberately placed there) -- it moves into `_make_plots` verbatim, still
inline, still visible next to the un-flipped `w_probe`/`b_probe` pair.

## Design

```python
@dataclasses.dataclass
class DiagnosticsResult:
    """Everything computed in phase 1, needed by both the report (phase 2)
    and the plots (phase 3)."""
    me: float
    gap: dict
    gap_plot_inputs: dict
    heldout: dict
    linear_y_r2: dict | None


def _run_diagnostics(
    model, hidden_layers, num_x, num_blocks, args, g, device, probe_backend_name,
    eval_noise_std,
) -> DiagnosticsResult:
    """Phase 1: all data generation, no plotting or printing."""
    me = eval_max_err(model, g, device=device)
    gap, gap_plot_inputs = _binary_probe_metrics_all_layers(...)
    heldout = {}
    if args.detailed:
        ...  # unchanged loop body
    linear_y_r2 = None
    if args.detailed:
        linear_y_r2 = _linear_y_reconstruction(...)
    return DiagnosticsResult(me, gap, gap_plot_inputs, heldout, linear_y_r2)


def _make_plots(
    args, model, ck, num_blocks, penalty_layers, hidden_layers, class_threshold,
    result: DiagnosticsResult, plot_dir, device,
):
    """Phase 3: everything currently after the report is written in main()."""
    ...  # unchanged bodies, reading result.gap / result.heldout / etc.
    #     in place of the bare local variables they replace


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.tag, args.ckpt, device)
    num_x, num_blocks = model.num_x, model.num_blocks
    penalty_layers, hidden_layers, eval_noise_std, class_threshold = _resolve_run_config(
        ck, num_blocks, args
    )
    if args.steer:
        _validate_steer_layers(args.steer, hidden_layers)
    plot_dir = get_plot_dir(args.tag)
    os.makedirs(plot_dir, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(args.seed)
    probe_backend_name = resolve_probe_backend(args.probe_backend, device)

    result = _run_diagnostics(
        model, hidden_layers, num_x, num_blocks, args, g, device,
        probe_backend_name, eval_noise_std,
    )

    lines = _build_report(
        args, num_x, model, ck, num_blocks, penalty_layers, hidden_layers,
        result.me, result.gap, result.heldout, result.linear_y_r2,
    )
    print("\n".join(lines))
    out_log = log_dir(args.tag)
    os.makedirs(out_log, exist_ok=True)
    report_path = os.path.join(out_log, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {report_path}")

    _make_plots(
        args, model, ck, num_blocks, penalty_layers, hidden_layers,
        class_threshold, result, plot_dir, device,
    )
```

`_resolve_run_config` and `_validate_steer_layers` are small extractions of
setup logic already sitting in `main()` today (the `adv_ck`-derived
fallbacks, and the `assert lyr in hidden_layers` loop for `--steer`) -- pulled
out purely so `main()` reads as a flat sequence of named steps, not because
either is reused elsewhere. `_build_report`'s own signature is unchanged by
this plan (still takes the individual fields, not `result`) -- collapsing it
to take `result` directly is a reasonable follow-up but out of scope here to
keep this diff to "extract, don't also reshape the extracted-from function."

## Steps

1. Add `DiagnosticsResult` dataclass.
2. Extract `_run_diagnostics` (phase 1 body, unchanged internals, wrapped).
3. Extract `_resolve_run_config` and `_validate_steer_layers` from `main()`'s
   setup section.
4. Extract `_make_plots` (phase 3 body, unchanged internals, wrapped; reads
   `result.<field>` in place of the bare locals it replaces).
5. Rewrite `main()` to the sequence in Design.
6. Diff every extracted function's body against `main()`'s pre-refactor
   source line-by-line to confirm nothing changed beyond the wrapping (this
   plan is pure extraction, not an opportunity to also fix other things --
   those are the other plans in this set).

## Verification

- Smoke run: `python adversarial_report.py --tag <existing run> --ckpt last`
  and diff `report.md` + the full set of `*.png` filenames/sizes against a
  pre-refactor run on the same tag/seed -- must match.
- Repeat with `--detailed --steer <valid layer> --show` to exercise every
  branch (`args.detailed`, `args.steer`, `linear_y_r2 is not None`) through
  both `_run_diagnostics` and `_make_plots`.
- `black --check adversarial_report.py`.

## Risks / caveats

- This is the largest diff of the five plans in this set, purely because
  it's a mechanical reshuffle of a large function -- keep it a pure
  extraction (step 6's line-by-line diff check) so review can focus on "did
  anything move wrong" rather than "did anything change."
- `_run_diagnostics`'s and `_make_plots`'s parameter lists are still fairly
  long (setup-derived values threaded through). That's accepted here rather
  than solved -- introducing a second config/context object on top of
  `DiagnosticsResult` is more surface area than this plan's scope justifies;
  revisit only if these signatures keep growing.
