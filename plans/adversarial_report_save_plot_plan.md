# Shared `_save_plot` helper for `adversarial_report.py` / `probe_lib.py`

## Context

Every plot-producing function across `adversarial_report.py` (7 functions:
`plot_learned_curves`, `_plot_training_traces`, `_plot_probe_gap`,
`_plot_heldout_gap`, `_plot_layer_distributions`, `_plot_steer_comparison`,
`_plot_linear_y_reconstruction`) and `probe_lib.py` (`plot_probe`,
`plot_probe_pca`) repeats the same four-line tail:

```python
path = os.path.join(plot_dir, f"{tag}_..._....png")
fig.savefig(path, dpi=120)
plt.close(fig)
print(f"[plot] wrote {path}")
```

`probe_lib.py`'s two functions are missing the `plt.close(fig)` call -- a
real figure leak in the sense that these figures are never explicitly freed.
But note what this leak currently *does*: `main()`'s `--show` flag
(`plt.show()`, called after all plotting) only ever displays figures that
are still open at that point. Every other plot function already closes
immediately after saving, so **today, `--show` only pops up windows for the
`plot_probe`/`plot_probe_pca` figures** -- the "leak" is, accidentally, the
only thing currently making `--show` do anything. A naive dedup that closes
every figure immediately (matching the other 7 functions) would silently
turn `--show` into a no-op. This plan closes that gap deliberately rather
than reproducing it.

Independent of every other `adversarial_report_*_plan.md` from this review
except that it touches the same lines `adversarial_report_main_decomposition_plan.md`
will later move into `_make_plots` -- land this one first so that
decomposition inherits the already-deduplicated call sites.

## Design

Add to `probe_lib.py` (the lower-level, dependency-free module of the two --
`adversarial_report.py` already imports from it). Saving and closing are
split: every plot function saves+logs immediately (as today), but closing is
deferred to one explicit point at the end of `main()`, after the optional
`plt.show()` -- this makes `--show` actually show every plot (not just the
two that happened to leak before), while the common case (no `--show`) still
frees each figure's memory no later than process exit, same as today.

```python
def save_plot(fig, plot_dir: str, filename: str) -> str:
    """Save and log a finished figure; does not close it (see `main`'s
    show/close-all at the end of the plotting phase)."""
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")
    return path
```

Every plot function's tail becomes `return save_plot(fig, plot_dir, f"{tag}_....png")`
(or just a bare call, for the functions that don't currently return the
path), with every inline `plt.close(fig)` deleted. `main()` gains, right
after today's `if args.show: plt.show()`:

```python
    if args.show:
        plt.show()
    plt.close("all")
```

This is a peak-memory tradeoff worth naming, not hiding: figures now stay
open for the whole plotting phase instead of being freed as they're
produced. Given this project's toy-scale models and plot counts (order of
`2 * num_hidden_layers + ~7` figures per report run), this is expected to be
a non-issue -- but if a future model scale makes it one, the fix is to close
each figure right after `plt.show()` processes it (`plt.show()` doesn't need
already-closed figures) rather than reverting to per-function immediate
close, which would reintroduce the `--show` regression above.

## Steps

1. Add `save_plot` to `probe_lib.py`.
2. Replace the tail of `plot_probe` and `plot_probe_pca` (`probe_lib.py`):
   call `save_plot`, drop nothing (they had no `plt.close` to remove).
3. Replace the tail of all 7 plot functions in `adversarial_report.py`,
   importing `save_plot` from `probe_lib` and deleting each inline
   `plt.close(fig)`.
4. Add the `plt.show()` / `plt.close("all")` pair at the end of `main()`
   (currently just `if args.show: plt.show()`).
5. Grep both files for `fig.savefig` / `plt.close` to confirm no stray
   inline call sites remain outside `save_plot` and the new `main()` tail.

## Verification

- Grep confirms zero remaining inline `fig.savefig(...)` calls outside
  `save_plot`, and zero remaining `plt.close(fig)` calls outside `main()`'s
  new `plt.close("all")`.
- Smoke run: `python adversarial_report.py --tag <any existing run> --ckpt last`
  (no `--detailed`/`--steer`) on an existing checkpoint under `runs/`,
  confirming all expected `*.png` files still land in the run's plot dir
  with unchanged filenames.
- Repeat with `--detailed --steer <a valid hidden layer>` to exercise every
  plot function, including `plot_probe`/`plot_probe_pca`.
- Repeat once more with `--show` on a machine with a display (or `Agg`
  backend swapped for a real one) and confirm every plot -- not just
  `plot_probe`/`plot_probe_pca` -- actually pops a window, i.e. the `--show`
  regression this plan is designed to avoid didn't happen.
- `black --check adversarial_report.py probe_lib.py`.

## Risks / caveats

- Peak memory during the plotting phase goes up slightly (figures accumulate
  until the final `plt.close("all")` instead of being freed as produced) --
  see the sizing note in Design. Not expected to matter at this project's
  scale; if it ever does, close-after-show per-figure rather than reverting
  to per-function immediate close (that would reintroduce the `--show`
  regression this plan fixes).
- Output filenames and figure contents are otherwise unchanged -- this is
  mechanical extraction plus the one deliberate close-timing change above.
