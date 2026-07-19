# Plan: consolidate plot-output-directory handling

## Motivation
Three scripts (`train_probe.py`, `train_model_plot.py`, `adversarial_report.py`)
each independently do:
```python
out_dir = "plot"
os.makedirs(out_dir, exist_ok=True)
```
in their `main()`, then thread an `out_dir` parameter through their various
`plot_*` helpers. This is pure duplication — if the plot location ever needs
to change (e.g. move under `runs/<tag>/` instead of a flat top-level `plot/`),
it's currently a 3-way edit. Additionally, `adversarial_report.py`'s
`binary_probe_metrics` has a stray hardcoded `"plot"` string literal (line
~360) instead of reusing its own `out_dir` variable — an existing bug this
refactor incidentally fixes.

Separately, `binary_probe_metrics(..., tag=..., out_dir=...)` uses `tag` only
to build a plot filename/title and `out_dir is not None` only to decide
whether to plot at all — `tag` is otherwise always `args.tag` at every call
site. Dropping `tag` and keeping only the directory param removes that
redundant plumbing while preserving the "presence of the arg = plot this"
toggle.

## Note: plot location is undecided
Whether plots should live in a flat `plot/<tag>_*.png` layout (current) or
move under `runs/<tag>/plots/` (matching the `runs/<tag>/{checkpoints,logs}/`
convention in `paths.py`) is **not yet decided**. This plan intentionally
keeps the current flat `plot/<tag>/` location so it's a no-op relocation —
the whole point is that this becomes a **one-line change in one place**
(`paths.get_plot_dir`) whenever that decision is made. Do **not** touch
`.gitignore` as part of this work — there are existing plots on disk under
the current location and the ignore rule shouldn't change until the location
itself does.

## Steps

1. **`paths.py`**: add a new function
   ```python
   def plot_dir(tag: str) -> str:
       return os.path.join("plot", tag)
   ```
   (naming it `plot_dir` for consistency with `ckpt_dir`/`log_dir`, even
   though it's imported under a different name at call sites — see below).
   Update the module docstring line ("Per-run output directory layout: ...")
   to mention plots.

2. **Rename `out_dir` → `plot_dir` everywhere it's used as a parameter or
   local variable**, across all three scripts, so the name consistently
   reflects what the value is (a plot-output directory, and — where optional
   — the "should I plot" toggle):
   - `train_probe.py`: `plot_steering(..., out_dir)` → `plot_steering(..., plot_dir)`;
     `plot_probe(..., out_dir)` → `plot_probe(..., plot_dir)`; `main()`'s
     `out_dir = "plot"` local var → `plot_dir = get_plot_dir(args.tag)`.
   - `train_model_plot.py`: `plot_dynamics(tag, out_dir)`,
     `plot_learned_curves(model, tag, out_dir, ...)`,
     `plot_curves(tag, ckpt, out_dir)` — all rename their `out_dir` param to
     `plot_dir`; `main()`'s local var likewise.
   - `adversarial_report.py`: `plot_training_traces(..., out_dir)`,
     `plot_heldout_r2(..., out_dir)`, `plot_probe_gap(..., out_dir)` — rename
     param to `plot_dir`; `main()`'s local var likewise; fixes the stray
     hardcoded `"plot"` literal at the `binary_probe_metrics` call site in
     the section-2 loop (should reuse the `plot_dir` variable instead).

   Because the module-level function in `paths.py` is also naturally called
   `plot_dir`, importing it under that name would shadow the local variable
   of the same name in every `main()`. **Import it aliased**:
   ```python
   from paths import plot_dir as get_plot_dir
   ```
   in all three files, then `plot_dir = get_plot_dir(args.tag)` in each
   `main()`.

3. **`binary_probe_metrics`** (`adversarial_report.py`): drop the `tag`
   param entirely; keep only `plot_dir=None` (renamed from `out_dir`). Its
   presence (`is not None`) is the sole "write a plot" toggle. Build the
   `plot_probe` filename tag internally as `f"c{c_lo:g}-{c_hi:g}"` (no run
   tag prefix needed — that's ambient context the caller already has via
   `plot_dir`'s value, not something this per-pair helper should know about).
   Update its two call sites in `main()`:
   - section 2 loop: pass `plot_dir=plot_dir` (drop `tag=args.tag`)
   - section 3b loop: unchanged (no `plot_dir` passed → no plot, as today)

4. **Verify**: run each of the three scripts' CLI once against an existing
   checkpoint (e.g. `adv_lam0.25_scratch`) and confirm plots still land in
   the same place with the same filenames as before (this refactor should be
   behavior-preserving — only the internal plumbing changes, not where files
   land or how they're named for the four multi-plot helpers). Run
   `black --check` on all three files.

## Explicitly out of scope
- Moving plots to `runs/<tag>/plots/` or any other location — that's a
  separate decision to make later, and should be a single-line change to
  `paths.plot_dir` once decided.
- `.gitignore` changes.
