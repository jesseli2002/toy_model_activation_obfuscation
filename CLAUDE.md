# CLAUDE.md

Operational reference for this repo: what's here, how to train something, and
how to reproduce the writeup. See `README.md` for the writeup link.

## Source structure

**Train a model:**
```
python train_adversarial_logreg.py --config configs/example.json --tag my_first_run
```
`configs/example.json` is the one example config kept in the repo — see "Run
configs" below for what it means and how `--resume`/`--fork-from` differ from
a fresh run. `train_no_c.py` is a separate, standalone trainer for the c-blind
control (no adversarial penalty, hyperparameters as module constants).

**Supporting libraries**, used across most of the above: `model.py` (the
architecture), `data.py` (task sampling/eval), `config.py` (run config
schema), `probe_lib.py`/`probe_backend.py`/`probe_newton.py`/`torch_logreg.py`
(the adversarial probe and its solver backends), `checkpoint_lib.py`,
`paths.py` (`runs/<tag>/`, `plot/<tag>[/<ckpt>]/`), `stableadamw.py`,
`rate_meter.py`. `sweep_lib/` is the shared plotting/caching/discovery layer
under the `sweep_*.py` reporting scripts below.

Every script parses `--help` before its heavy imports (torch etc.) — safe to
probe quickly.

## Reproducing the writeup

The writeup (linked from `README.md`) splits its Results section into three
parts; this section is ordered to match, one command per part.

**Part 1 — analytic construction, and the analytic blog post's figures.**
```
python analytic.py        # verification report + plot/analytic/*.png
python analytic_demo.py   # -> plot/analytic/simplified_*.png
```
Both are closed-form checks, no trained model involved. `analytic.py`'s
docstring documents exactly what each check verifies (Saturation,
Obfuscation, v-channels, ideal-y decodability) — not duplicated here.
`analytic_demo.py` is the minimal, self-contained construction linked
directly from the analytic blog post; `sample_ideal_y`/`verify_ideal_y_decodability`
in `analytic.py` establish the task is solvable at all.

The **c-blind floor**: `analytic.no_c_task_loss` is the lowest task loss any
predictor of `x` alone can reach — a model scoring below it must be reading
`c`. `train_no_c.py` empirically confirms the floor is reachable; check its
result by reading the last line's `l_task` from `runs/baseline_no_c/logs/history.jsonl`.

**Part 2 — single-run empirical result.**
```
python make_one_run_plots.py
```
regenerates every figure for both writeup tags (`sweep7_lam0.1_tr0`,
`sweep3_lam0_tr0`) into `plot/<tag>/<ckpt>/`. No `--tag` flag — the tags are
a module constant, since this script targets exactly the writeup's runs.

**Part 3 — hyperparameter sweeps:**
```
python sweep7_analysis.py                 # lambda sweep -> plot/sweep7/
python plot_train_dist_curve.py            # ID-vs-OOD curve for sweep11_..._tr0
python sweep_width_analysis.py             # -> plot/sweep_width/
python sweep_layer_analysis.py             # -> plot/sweep_layer/
```

## The background work

Not writeup figures, but part of reproducing the process, not just the
result:
- `sweep_threshold_report.py` — how the loss/AUROC pass thresholds were
  picked, by rendering diagnostics for representative runs across a sweep.
- `sweep_group_report.py` — the worked example of ad-hoc cross-sweep
  comparison (edit its `GROUPS` constant to point at whatever you're
  comparing; three examples are kept as templates).
- `sweep_inspect_training.py` — superimposed loss curves + final AUROC across
  same-settings runs; edit `RUN_TAGS` to point at whatever you're inspecting.

All three, like the sweep `*_analysis.py` scripts, take their settings from
module constants rather than a CLI — argparse would be churn for shapes that
are still changing.

## Run configs

`configs/example.json` documents the format and gives you something to train
from immediately — it is **not** a record of what any run used. That's
`runs/<tag>/config.json` (fully resolved) and `runs/<tag>/input_config.json`
(the file as passed to `--config`), both kept for every published run.
`--resume`/`--fork-from` read these rather than `--config` — see
`train_adversarial_logreg.py`'s docstring for the three run modes.

## Run data

525 curated runs (`checkpoints/{best,last}.pt`, `config.json`,
`input_config.json`, `logs/history.jsonl` — the `iter_*.pt` series is
dropped) live on Hugging Face Hub, public repo
`cooleytukey/toy_model_of_activation_obfuscation`, under a `runs/` path
matching the local layout — no auth needed to download:
```
hf download cooleytukey/toy_model_of_activation_obfuscation --repo-type model --local-dir . --include "runs/*"
```
`plot/metrics_cache.json` is not published — it's keyed by checkpoint mtime,
which a download doesn't preserve, so it silently rebuilds (slowly, once) on
first use rather than being shipped stale.
