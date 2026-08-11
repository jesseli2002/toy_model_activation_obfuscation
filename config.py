"""Shared hyperparameters for the probe-obfuscation toy model.

These are defaults; entry-point scripts accept CLI overrides so a run can
diverge from them without editing this file.
"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import ClassVar

# Data distributions
X_LOW, X_HIGH = -3.0, 3.0  # x ~ U[X_LOW, X_HIGH]
C_LOW, C_HIGH = 1.0, 2.0  # c ~ U[X_LOW, X_HIGH]

# Training
MAX_ITERS = 100_000


ACTIVATION_CHOICES = ["leaky_relu", "gelu"]
PROBE_BACKEND_CHOICES = ["auto", "sklearn", "torch", "newton"]
OPTIMIZER_KIND_CHOICES = ["adamw", "stableadamw"]
LR_SCHEDULE_CHOICES = ["cosine", "exponential"]


class _CheckpointConfigMixin:
    """Shared to_dict/from_dict for config dataclasses stored verbatim (as
    dicts) in checkpoints, so they must survive old checkpoints gaining new
    fields over time.

    Subclasses must be @dataclass and define a ClassVar `_LEGACY_DEFAULTS`
    dict covering every optional field. Two distinct notions of "default"
    apply to each such field, and they are deliberately kept separate:
      - the dataclass field default is what a fresh Config(...) call gets if
        the field is omitted -- i.e. the default for code written going
        forward.
      - _LEGACY_DEFAULTS is what an *old* checkpoint -- one saved before the
        field existed, so its config dict is missing the key -- is backfilled
        to by from_dict(). This is what that field's value effectively WAS,
        historically, before it became configurable.
    These do NOT always coincide -- see the inline comments on each
    subclass's _LEGACY_DEFAULTS for its own divergent fields. Any future
    change that bumps a field's forward default must leave _LEGACY_DEFAULTS
    alone, so old checkpoints keep reconstructing with the hyperparameters
    they were actually run with.

    This mixin owns no fields of its own (no dataclass field-ordering
    concerns from mixing it into @dataclass subclasses) and reads
    `_LEGACY_DEFAULTS`/`dataclasses.fields(cls)` off whichever subclass calls
    it, via ordinary classmethod `cls` polymorphism.
    """

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = d.keys() - known
        if unknown:
            warnings.warn(
                f"{cls.__name__}.from_dict: dropping unrecognized key(s) "
                f"{sorted(unknown)} -- checkpoint saved by a newer version?"
            )
        present = {k: v for k, v in d.items() if k in known}
        # union over dicts, preferring `present`
        return cls(**(cls._LEGACY_DEFAULTS | present))


@dataclass(kw_only=True)
class ResidualMLPConfig(_CheckpointConfigMixin):
    """Architecture + init hyperparameters for ResidualMLP.

    `num_x`/`d_model`/`d_mlp`/`num_blocks` have no default -- every caller
    must pin the architecture explicitly, rather than this dataclass and its
    CLI callers each carrying their own copy of the same default.
    """

    num_x: int
    d_model: int
    d_mlp: int
    num_blocks: int
    out_init_scale: float = 0.1
    activation: str = "gelu"
    leaky_relu_slope: float = 0.0
    layer_norm: bool = False

    # Historical values for fields absent from an old checkpoint's config
    # dict. Every optional field above must have an entry here.
    _LEGACY_DEFAULTS: ClassVar[dict] = {
        "num_blocks": 4,
        "out_init_scale": 0.1,
        "activation": "leaky_relu",
        "leaky_relu_slope": 0.0,
        "layer_norm": False,
    }


@dataclass(kw_only=True)
class LogregAdversarialConfig(_CheckpointConfigMixin):
    """Training-hyperparameter metadata for train_adversarial_logreg.py,
    stored verbatim (as a dict) alongside the model's own ResidualMLPConfig.

    Every field here is config-file-only: no default, no CLI flag, a
    required key in the --config JSON file, so omitting it fails loudly (a
    TypeError from this constructor) instead of silently taking on a value.
    See train_adversarial_logreg.py's module docstring for the rationale
    behind this split (architecture vs. config-file vs. bookkeeping). `seed`
    is a CLI flag instead (bookkeeping, on `LogregRunConfig`), so a trial
    can't be re-run with a different seed by editing --config and forgetting
    to bump it.
    """

    # No default -- required key in --config JSON.
    lam: float
    penalty_layers: list | str  # "all", or an explicit list of layer indices
    # Three phases, run in order: lam=0 for lam0_warmup_iters, then lam
    # ramping linearly 0 -> lam over lam_warmup_iters, then lam held constant
    # for the remainder of training. Either or both may be 0 to skip that
    # phase.
    lam0_warmup_iters: int
    lam_warmup_iters: int
    probe_C: float
    probe_init_iters: int
    class_threshold: float
    probe_loss_kind: str
    probe_subsample: int
    probe_retrain_interval: int
    # 0 disables resampling -- the probe dataset stays fixed for the whole
    # run, as it always did before this field existed.
    probe_resample_interval: int
    # Per-tail fraction of each class's projected scores dropped before
    # averaging in the adversarial penalty (e.g. 0.05 drops below the 5th and
    # above the 95th percentile, keeping the middle 90%) -- robustifies the
    # penalty against a handful of extreme activations dominating the mean.
    # 0.0 disables trimming (plain mean).
    probe_loss_trim_frac: float
    resid_noise_std: float
    # False (legacy): probe fit and penalty both read a clean probe pass, as
    # before this field existed. True: both instead read one pass noised at
    # resid_noise_std -- fit and penalty share that single noisy pass (not
    # noisy-eval-of-a-clean-fit), which is what keeps a clean-fitted w_eff
    # from blowing up on noise landing in near-zero-clean-variance
    # directions (see plans/archive/resid_stream_noise_plan.md). The task
    # pass is unaffected either way -- always noisy at resid_noise_std.
    probe_noise: bool
    # Clipped per-block, not whole-model: a global L2 norm sums correlated
    # per-block gradient contributions (every block shares the same backprop
    # signal) under one sqrt, so its natural scale grows with num_blocks even
    # though nothing about per-block gradients themselves changed with depth.
    # Per-block norms stay roughly architecture-independent, so a single
    # fixed threshold generalizes without recalibration. 0 = no clipping.
    grad_clip: float
    x_p_outer: float | None
    x_threshold: float
    batch_size: int
    lr: float
    # AdamW's eps/beta1/beta2 (PyTorch defaults: 1e-8, 0.9, 0.999). A parameter
    # whose gradient has been near-zero for a while (small running exp_avg_sq)
    # can take a disproportionately large adaptive step the moment its
    # gradient becomes non-trivial again, since exp_avg_sq hasn't caught up
    # yet -- a source of loss spikes grad_clip above can't touch, since
    # clipping acts on the raw gradient, not AdamW's per-parameter adaptive
    # step. Raising eps and/or lowering beta2 both mitigate this; empirically,
    # lowering beta2 (e.g. 0.99) converged faster with fewer residual spikes
    # than raising eps (e.g. 1e-5) alone.
    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    # Linear warmup from ~0 to lr over the first lr_warmup_iters steps, then
    # decay from lr down to lr * lr_min_frac over the remaining steps, shaped
    # by lr_schedule (0 warmup / lr_min_frac=1.0 reproduces the old
    # constant-lr behavior exactly, under either shape). Computed as a pure
    # function of (it, max_iters), so it needs no scheduler object/state and
    # just falls out correctly across --resume/--fork-from.
    #
    # Both endpoints are pinned, which leaves an exponential no free shape
    # parameter: it is the unique geometric interpolation, lr_min_frac
    # ** progress. Its one extra constraint over cosine is lr_min_frac > 0 --
    # a geometric decay approaches its floor but never reaches it, so a zero
    # floor has no finite decay rate (validated at config-load time).
    # Relative to cosine it leaves peak lr immediately rather than holding
    # flat near the top, and correspondingly spends longer in the low-lr
    # tail, so a matched-feel exponential run usually wants a larger
    # lr_min_frac than the cosine run it is being compared against.
    lr_warmup_iters: int
    lr_min_frac: float
    lr_schedule: str  # "cosine" or "exponential" (LR_SCHEDULE_CHOICES)
    # "adamw" or "stableadamw" (per-tensor update-clipping in place of
    # grad_clip/adam_eps/adam_beta2 band-aids) -- both consume adam_eps
    # /adam_beta1/adam_beta2 above with the same meaning.
    optimizer_kind: str
    # StableAdamW's update-clipping threshold (Adafactor's `d`): a tensor's
    # step is scaled down only once its RMS exceeds this, so lowering it
    # clips more aggressively while leaving quiet steps untouched. 1.0 is
    # what every off-the-shelf implementation hard-codes. Ignored by
    # optimizer_kind="adamw".
    stableadamw_d: float
    # Revert-and-retry: after each step, re-check the loss on the same batch,
    # and if it jumped by more than explode_factor x the smallest loss seen
    # in the last explode_window_iters iterations (explode_window_iters=1
    # compares only against this iteration's own pre-step loss), undo the
    # step (model + optimizer state) and redo it with grad_clip divided by
    # explode_clip_divisor. Comparing against a window rather than just this
    # iteration's own pre-step loss catches gradual multi-step creep (e.g.
    # loss roughly doubling on several consecutive steps, each one under
    # explode_factor on its own). This only catches genuine large-gradient
    # -norm events -- adam_eps/adam_beta2 above are the fix for the small
    # -gradient-norm spikes this retry can't affect (confirmed empirically:
    # redoing with a tighter clip is a no-op when the original gradient norm
    # was already below the tighter threshold). explode_factor=0 disables.
    explode_factor: float
    explode_clip_divisor: float
    explode_window_iters: int

    _LEGACY_DEFAULTS: ClassVar[dict] = {
        "lam": 0.5,
        "lam0_warmup_iters": 0,
        "lam_warmup_iters": 0,
        "lr_warmup_iters": 0,
        "lr_min_frac": 1.0,
        "lr_schedule": "cosine",  # legacy runs predate the shape being selectable
        "penalty_layers": None,
        "probe_C": 1.0,
        "probe_init_iters": 1000,
        "class_threshold": 1.5,
        "probe_loss_kind": "meandiff-relu",
        "probe_subsample": 1,  # legacy runs fit on the full batch every step
        "probe_retrain_interval": 1,  # legacy runs refit every iteration
        "probe_resample_interval": 0,  # legacy runs never resampled the probe set
        "probe_loss_trim_frac": 0.0,  # legacy runs used the plain (untrimmed) mean
        "resid_noise_std": 0.0,  # legacy runs trained with no residual-stream noise
        "probe_noise": False,  # legacy runs fit/scored the probe on a clean pass
        "grad_clip": 0.0,  # legacy runs trained with no gradient clipping
        "x_p_outer": None,  # legacy runs sampled x plain-uniform, unbiased
        "x_threshold": 1.0,
        "batch_size": 4096 * 4,  # legacy runs took this from a CLI default
        "lr": 3e-3,  # legacy runs took this from a CLI default
        "adam_eps": 1e-8,  # legacy runs used AdamW's plain default
        "adam_beta1": 0.9,  # legacy runs used AdamW's plain default
        "adam_beta2": 0.999,  # legacy runs used AdamW's plain default
        "optimizer_kind": "adamw",  # legacy runs predate StableAdamW support
        "stableadamw_d": 1.0,  # legacy runs predate the threshold being tunable
        "explode_factor": 0.0,  # legacy runs had no explode-detection
        "explode_clip_divisor": 5.0,  # inert when explode_factor is 0
        "explode_window_iters": 1,  # legacy runs only compared to the immediately preceding step
    }


@dataclass
class ForkedFrom:
    tag: str
    iter: int


@dataclass
class LogregRunConfig:
    """Complete definition of one train_adversarial_logreg.py run, serialized
    to runs/<tag>/config.json as that run directory's read-only record.

    Sole definition of that file's schema -- document it here, not in the
    script that writes it.

    `model` is frozen at tag creation (--fork-from inherits it from the source
    checkpoint); `adversarial` is freshly resolved for every new tag;
    `forked_from` is present only on a forked tag.
    """

    model: ResidualMLPConfig
    adversarial: LogregAdversarialConfig
    seed: int
    forked_from: ForkedFrom | None = None

    def to_dict(self) -> dict:
        d = {
            "model": self.model.to_dict(),
            "adversarial": self.adversarial.to_dict(),
            "seed": self.seed,
        }
        if self.forked_from is not None:
            d["forked_from"] = dataclasses.asdict(self.forked_from)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LogregRunConfig":
        forked_from = d.get("forked_from")
        return cls(
            model=ResidualMLPConfig.from_dict(d["model"]),
            adversarial=LogregAdversarialConfig.from_dict(d["adversarial"]),
            seed=d["seed"],
            forked_from=ForkedFrom(**forked_from) if forked_from is not None else None,
        )
