# Does LayerNorm improve training reliability?

**Answer: no. At lam=0.1 it makes things dramatically worse, and an LR warmup
mitigates but does not fix it.** LayerNorm is not the dial to turn for this
problem; the residual-stream evidence below points at a different one.

Run tags: `sweep4_ln_lam0_tr*` (n=6), `sweep4_ln_lam0.1_tr*` (n=4),
`sweep4_lnwarm_lam0.1_tr*` (n=6). Baselines are the existing
`sweep3_lam0_tr*` (n=14 with checkpoints) and `sweep3_lam0.1_tr*` (n=15).

## Method

`--layer-norm` did not exist; `ResidualMLPConfig.layer_norm` was implemented
(pre-norm on each block's input) but unreachable from the CLI. Exposing it was
the only production code change.

The LN arms differ from their baseline in **exactly one config field**
(`model.layer_norm: false -> true`), verified by diffing the written
`runs/<tag>/config.json`; seeds are matched pairwise. The warmup arm adds
exactly one more (`lr_warmup_iters: 0 -> 5000`), from a config file kept
outside the repo so the remote's file sync could not delete it.

Task loss is `data.eval_task_loss` recomputed from each run's last checkpoint
under its own training noise -- matching `sweep_analysis.py` rather than
reading `history.jsonl`, whose last record can postdate the final checkpoint.
See `sweep4_layernorm_report.py`.

## 1. lam=0: the task is learnable with LN, but LN costs ~33% headroom

| arm | n | < 7e-3 | median | min-max |
|---|---|---|---|---|
| no-LN | 14 | 14/14 | 2.412e-3 | 2.10-2.81e-3 |
| LN | 6 | 6/6 | 3.195e-3 | 3.12-3.29e-3 |

Mann-Whitney U = 84 = 6x14, the maximum: *complete separation*, every LN run
worse than every no-LN run (p=1e-4, two-sided).

Note LN is markedly **more repeatable** here (spread 0.18e-3 vs 0.71e-3). It
raises the floor while tightening the distribution -- worth knowing, but at
lam=0 nothing was failing anyway, so it buys nothing.

## 2. lam=0.1: LN is catastrophic

Median `l_task` at matched iteration:

| iter | no-LN | LN | LN + warmup5k |
|---|---|---|---|
| 10k | 6.38e-2 | 2.26e-1 | 1.86e-1 |
| 15k | 4.69e-2 | 5.24e-1 | 1.65e-1 |
| 30k | 1.70e-2 | 1.65e-1 | - |
| 50k | 1.05e-2 | 1.21e-1 | - |
| 67k | 8.05e-3 | 1.00e-1 | - |

The no-LN baseline converges to ~6-8e-3; LN plateaus around 1e-1, ~12x worse,
and the gap *widened* monotonically (3.0x -> 9.5x -> 15.5x over iters 11k-18k).
Every LN run was worse than every baseline run.

Trajectories show the shape of the failure: LN drives `l_probe` to ~0.03 by
iter 2k (hiding early and well), then `l_task` explodes -- one run from 0.135
at iter 2k to 1.02 at iter 5k. `explode_factor=0` in these configs, so the
revert-and-retry guard is disabled and nothing catches it.

A 5000-iter LR warmup removes the acute blowup and is 3.2x better at iter 15k,
but is still 3.5x worse than baseline there. It mitigates; it does not fix.

## 3. Why -- and this reframes the original hypothesis

The prompt for this work was that residual RMS "sometimes explodes". Across the
no-LN `sweep3_lam0.1` runs (n=15), *peak* RMS turns out to carry no signal at
all:

| statistic | Spearman rho vs task loss | p |
|---|---|---|
| peak RMS | +0.021 | 0.94 |
| **final-layer RMS** | **+0.729** | **0.002** |
| peak/final decay ratio | -0.575 | 0.025 |

Every run inflates the stream at the penalized layer -- **that inflation is the
hiding mechanism, not a pathology**. What separates a run that learns the task
is whether it brings the stream back *down* before the unembed: passing runs
decay 2.4-4.3x, failing runs 1.1-2.1x.

The LN runs confirm the mechanism directly:

| arm | peak RMS | final RMS | peak/final |
|---|---|---|---|
| LN, lam=0 | ~3.2 | ~2.9 | ~1.1 (flat profile) |
| LN, lam=0.1 | 22-98 | 12-86 | 1.01-3.64 |

At lam=0 LN holds the stream flat at ~3.2 across all 12 layers. At lam=0.1 it
inflates to 22-98 at the penalized layer and never decays (one run: 63.1 peak,
62.2 final -- a ratio of 1.01).

This is what pre-norm *should* be expected to do here. Every downstream block
normalizes its input, so inflating the residual stream becomes nearly free;
only the unembed still reads raw scale, and that lone pressure is too weak to
force decay. Without LN the inflated stream disrupts every downstream block,
which is precisely what forces the model to decay it again.

## Recommendations

1. **Don't enable LN as implemented.** It removes the pressure that makes the
   task work.
2. If an intervention is wanted for the real failure mode, it should target
   **the residual scale the output sees**, not the peak. A final LayerNorm
   before the unembed is the natural candidate -- it is the opposite of what
   was tested here, and untested. This would be a model change, not a config
   change.
3. `explode_factor=0` disables revert-and-retry in these configs, so
   `n_exploded` is uninformative and loss excursions go uncaught. Worth
   revisiting independently of LN.

## Caveats

- The lam=0.1 LN arm is n=4, not the planned 15. The divergence was unambiguous
  (all 4 worse than all 15 baselines, gap widening), so the remaining 11 were
  paused rather than deleted -- resumable by editing the pool's concurrency
  file. n=4 supports "LN is much worse"; it does not support a precise effect
  size.
- Reported p-values are two-sided; halve for the directional reading.
- The mechanism claim in section 3 is correlational (n=15 baseline, n=10 LN)
  plus a coherent account, not a causal intervention.
