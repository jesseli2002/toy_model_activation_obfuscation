# Does LayerNorm improve training reliability?

**Answer: no. At lam=0.1 it makes things dramatically worse, and an LR warmup
mitigates but does not fix it.** LayerNorm is not the dial to turn for this
problem; the residual-stream evidence below points at a different one.

## Runs used

Every arm below is config-identical within itself apart from `seed`, verified
by comparing each run's own `runs/<tag>/config.json` (not the launch command).
Across **all five** arms exactly three fields differ -- also computed from the
config files rather than asserted:

| arm | tags | n | seeds | `layer_norm` | `lam` | `lr_warmup_iters` |
|---|---|---|---|---|---|---|
| baseline, lam=0 | `sweep3_lam0_tr{0..13}` | 14 | 0-13 | false | 0 | 0 |
| baseline, lam=0.1 | `sweep3_lam0.1_tr{0..14}` | 15 | 0-14 | false | 0.1 | 0 |
| LN, lam=0 | `sweep4_ln_lam0_tr{0..5}` | 6 | 0-5 | **true** | 0 | 0 |
| LN, lam=0.1 | `sweep4_ln_lam0.1_tr{0..3}` | 4 | 0-3 | **true** | 0.1 | 0 |
| LN + warmup, lam=0.1 | `sweep4_lnwarm_lam0.1_tr{0..5}` | 6 | 0-5 | **true** | 0.1 | **5000** |

`sweep3_lam0_tr14` exists but has no checkpoint, hence n=14 rather than 15.

All other settings are shared by every arm:

```
model:  num_x=32  d_model=64  d_mlp=16  num_blocks=12
        out_init_scale=0.1  activation=gelu  leaky_relu_slope=0.0
adversarial:
        penalty_layers=[2]  lam_warmup_iters=0  probe_C=1.0
        probe_init_iters=1000  class_threshold=1.5
        probe_loss_kind=probe-report  probe_subsample=4
        probe_retrain_interval=2  probe_resample_interval=512
        probe_loss_trim_frac=0.05  resid_noise_std=0.03  grad_clip=1.0
        x_p_outer=null  x_threshold=1.0  batch_size=16384  lr=0.003
        adam_eps=1e-05  adam_beta1=0.7  adam_beta2=0.97
        optimizer_kind=stableadamw  stableadamw_d=1.0
        explode_factor=0.0  explode_clip_divisor=20  explode_window_iters=10
        lr_min_frac=0.1
CLI:    --max-iters 100000  --probe-backend newton
        --log-interval 125  --ckpt-interval 2000
```

The `sweep4_*` runs were trained on the vast.ai remote and synced back to
local `runs/`. The warmup arm's config lives at
`/home/agent/work/configs/lam0.1_warmup5k.json` **on the remote** (outside the
repo, so the source sync could not delete it); it is reproduced verbatim into
each of those runs' `runs/<tag>/config.json`, which is the durable copy.

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

Final, at 100k iters -- the same iteration count as the baseline:

| arm | n | < 7e-3 | median | min-max |
|---|---|---|---|---|
| no-LN | 15 | **6/15 (40%)** | 7.848e-3 | 5.64e-3 - 3.05e-2 |
| LN | 4 | **0/4 (0%)** | 9.072e-2 | 7.71e-2 - 1.25e-1 |

Mann-Whitney U = 60 = 4x15, again the maximum: complete separation, p=5e-4.
The median is 11.6x worse, and the *best* LN run (7.71e-2) is 2.5x worse than
the *worst* baseline run (3.05e-2).

Worth noting which test showed this. The threshold count (0/4 vs 6/15) gives
Fisher p=0.26 -- not significant at n=4. The rank test on the raw losses gives
p=5e-4. Binarizing an unambiguous effect threw away all of its power; this is
the concrete reason to lead with the distribution rather than the
below-threshold fraction.

Median `l_task` at matched iteration, showing how it got there:

| iter | no-LN | LN | LN + warmup5k |
|---|---|---|---|
| 10k | 6.38e-2 | 2.26e-1 | 1.86e-1 |
| 15k | 4.69e-2 | 5.24e-1 | 1.64e-1 |
| 20k | 2.15e-2 | 2.44e-1 | 1.66e-1 |
| 30k | 1.70e-2 | 1.65e-1 | - |
| 40k | 1.30e-2 | - | 1.66e-1 |
| 50k | 1.05e-2 | 1.21e-1 | - |
| 67k | 8.05e-3 | 1.00e-1 | - |
| 100k (final) | **7.85e-3** | **9.07e-2** | - |

The no-LN baseline converges to ~6-8e-3; LN plateaus around 1e-1, ~12x worse,
and the gap *widened* monotonically (3.0x -> 9.5x -> 15.5x over iters 11k-18k).
Every LN run was worse than every baseline run.

Trajectories show the shape of the failure: LN drives `l_probe` to ~0.03 by
iter 2k (hiding early and well), then `l_task` explodes -- one run from 0.135
at iter 2k to 1.02 at iter 5k. `explode_factor=0` in these configs, so the
revert-and-retry guard is disabled and nothing catches it.

A 5000-iter LR warmup removes the acute blowup -- 3.2x better than plain LN at
iter 15k -- but then **plateaus**: 1.64e-1, 1.66e-1, 1.66e-1 at iters 15k, 20k,
40k, while the baseline falls to 1.30e-2 (a 12.8x gap). Since the baseline
improves only ~1.65x over its own final 60% of training, no plausible late
gain brings the warmup arm near 7e-3. Warmup mitigates the instability; it does
not make LN viable.

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

Pooling both lam=0.1 arms at 100k (n=19) makes final-layer RMS the single
universal predictor, and demotes the other two:

| statistic | Spearman rho | p |
|---|---|---|
| peak RMS | +0.337 | 0.16 |
| **final-layer RMS** | **+0.805** | **<1e-4** |
| peak/final decay | -0.372 | 0.12 |

(The decay ratio is significant *within* the no-LN arm alone, but final RMS is
what survives pooling -- it is the quantity that predicts across both
architectures, which is the stronger claim.)

The LN runs confirm the mechanism directly:

| arm | peak RMS | final RMS |
|---|---|---|
| LN, lam=0 | ~3.2 | ~2.9 (flat profile, ratio ~1.1) |
| LN, lam=0.1 | 29-237 | **22-170** |
| no-LN lam=0.1, runs that passed | 19-63 | 7.8-20.1 |

At lam=0 LN holds the stream flat at ~3.2 across all 12 layers -- it does
exactly what LayerNorm is supposed to do. At lam=0.1 the same architecture
inflates to 29-237 at the penalized layer and never recovers: the worst run
peaks at 237 and still reads 170 at the unembed, an order of magnitude above
any baseline run that learned the task.

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
  paused rather than deleted -- resumable by setting the pool's concurrency
  file `/home/agent/work/conc.txt` back to 10 on the remote, where the queue
  still holds them. n=4 supports "LN is much worse"; it does not support a
  precise effect size.
- The LN+warmup arm's final numbers are pending; the figures quoted here are at
  matched iteration (10k-40k), where it had already plateaued around 1.2e-1
  against a baseline at 1.7e-2. Its conclusion ("mitigates, does not fix") rests
  on trajectory, not on 100k endpoints.
- Reported p-values are two-sided; halve for the directional reading.
- The mechanism claim in section 3 is correlational (n=15 baseline, n=10 LN)
  plus a coherent account, not a causal intervention.
