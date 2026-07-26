"""Shared hyperparameters for the probe-obfuscation toy model.

These are defaults. `train.py` accepts CLI overrides (notably --num-x) so we can
run the num_x=1 base case and scale up incrementally without editing this file.
"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import ClassVar

SEED = 913768

# Data distributions
X_LOW, X_HIGH = -3.0, 3.0  # x ~ U[X_LOW, X_HIGH]
C_LOW, C_HIGH = 1.0, 2.0  # c ~ U[X_LOW, X_HIGH]

# Training
# BATCH_SIZE = 4096
BATCH_SIZE = 4096 * 4
LR = 3e-3
MAX_ITERS = 100_000
EARLY_STOP_LOSS = 1e-12  # float32 eps^2 ~ 1.4e-14; 1e-12 is a sane "exact" bar


PROBE_LOSS_CHOICES = ["squared", "absolute", "squared-var", "absolute-std", "lda"]
PROBE_LOGREG_LOSS_CHOICES = ["meandiff-relu", "meandiff"]
ACTIVATION_CHOICES = ["leaky_relu", "gelu"]
PROBE_BACKEND_CHOICES = ["auto", "sklearn", "torch"]


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


@dataclass
class ResidualMLPConfig(_CheckpointConfigMixin):
    """Architecture + init hyperparameters for ResidualMLP."""

    num_x: int = 32
    d_model: int = 256
    d_mlp: int | None = None
    num_blocks: int = 8
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

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = self.num_x


@dataclass
class AdversarialConfig(_CheckpointConfigMixin):
    """Training-hyperparameter metadata for train_adversarial.py, stored
    verbatim (as a dict) alongside the model's own ResidualMLPConfig."""

    lam: float = 0.5
    lam_warmup_iters: int = 0
    penalty_layers: list | None = None
    init: str = "scratch"
    warmstart_path: str | None = "runs/nx32/checkpoints/best.pt"
    seed: int = SEED
    probe_loss: str = "lda"
    probe_loss_eps: float = 1e-7
    lda_shrinkage: float = 1e-3
    probe_loss_detach_denom: bool = False
    resid_noise_std: float = 0.1

    _LEGACY_DEFAULTS: ClassVar[dict] = {
        "lam": 0.5,
        "lam_warmup_iters": 0,
        "penalty_layers": None,
        "init": "warmstart",
        "warmstart_path": None,
        "seed": 913768,
        "probe_loss": "squared",  # legacy runs used the hardcoded squared penalty
        "probe_loss_eps": 1e-7,
        "lda_shrinkage": 1e-3,
        "probe_loss_detach_denom": False,
        "resid_noise_std": 0.0,  # legacy runs trained with no residual-stream noise
    }


@dataclass(kw_only=True)
class LogregAdversarialConfig(_CheckpointConfigMixin):
    """Training-hyperparameter metadata for train_adversarial_logreg.py,
    stored verbatim (as a dict) alongside the model's own ResidualMLPConfig.

    Fields split two ways (see plans/rare_flags_config_plan.md): `lam` is
    touched often enough to stay a CLI flag, so it keeps an ordinary default
    here. Everything else has NO default -- it's a required key in the
    --config JSON file, so omitting it fails loudly (a TypeError from this
    constructor) instead of silently taking on a value.
    """

    # CLI-common: touched often, so exposed as a flag with a default here.
    lam: float = 0.5

    # Config-file-only: no default -- required key in --config JSON.
    penalty_layers: list | str  # "all", or an explicit list of layer indices
    lam_warmup_iters: int
    seed: int
    probe_C: float
    probe_init_iters: int
    class_threshold: float
    probe_loss_kind: str
    probe_subsample: int
    probe_retrain_interval: int
    resid_noise_std: float
    # Clipped per-block, not whole-model: a global L2 norm sums correlated
    # per-block gradient contributions (every block shares the same backprop
    # signal) under one sqrt, so its natural scale grows with num_blocks even
    # though nothing about per-block gradients themselves changed with depth.
    # A 16-arch sweep found per-block grad_norm's p99 stays ~0.2-0.4 regardless
    # of depth/width (vs. whole-model medians spanning ~9x with a clear depth
    # trend) -- so a single fixed threshold generalizes without
    # recalibration; 0.5 is a reasonable value (comfortably above that p99
    # ceiling, well below collapse-triggering spikes). 0 = no clipping.
    grad_clip: float
    x_p_outer: float | None
    x_threshold: float
    batch_size: int
    lr: float
    # AdamW's eps/beta2 (PyTorch defaults: 1e-8, 0.999). A parameter whose
    # gradient has been near-zero for a while (small running exp_avg_sq)
    # can take a disproportionately large adaptive step the moment its
    # gradient becomes non-trivial again, since exp_avg_sq hasn't caught up
    # yet -- a source of loss spikes grad_clip above can't touch, since
    # clipping acts on the raw gradient, not AdamW's per-parameter adaptive
    # step (see xbias_experiment.tmp.py's investigation). Raising eps and/or
    # lowering beta2 both mitigate this; empirically, lowering beta2 (e.g.
    # 0.99) converged faster with fewer residual spikes than raising eps
    # (e.g. 1e-5) alone.
    adam_eps: float
    adam_beta2: float
    # Same-iteration revert-and-retry: after each step, re-check the loss on
    # the same batch, and if it jumped by more than explode_factor x, undo
    # the step (model + optimizer state) and redo it with grad_clip divided
    # by explode_clip_divisor. This only catches genuine large-gradient-norm
    # events -- adam_eps/adam_beta2 above are the fix for the small-gradient
    # -norm spikes this retry can't affect (confirmed empirically: redoing
    # with a tighter clip is a no-op when the original gradient norm was
    # already below the tighter threshold). 0 = disabled.
    explode_factor: float
    explode_clip_divisor: float

    _LEGACY_DEFAULTS: ClassVar[dict] = {
        "lam": 0.5,
        "lam_warmup_iters": 0,
        "penalty_layers": None,
        "seed": 913768,
        "probe_C": 1.0,
        "probe_init_iters": 1000,
        "class_threshold": 1.5,
        "probe_loss_kind": "meandiff-relu",
        "probe_subsample": 1,  # legacy runs fit on the full batch every step
        "probe_retrain_interval": 1,  # legacy runs refit every iteration
        "resid_noise_std": 0.0,  # legacy runs trained with no residual-stream noise
        "grad_clip": 0.0,  # legacy runs trained with no gradient clipping
        "x_p_outer": None,  # legacy runs sampled x plain-uniform, unbiased
        "x_threshold": 1.0,
        "batch_size": 4096 * 4,  # legacy runs took this from a CLI default
        "lr": 3e-3,  # legacy runs took this from a CLI default
        "adam_eps": 1e-8,  # legacy runs used AdamW's plain default
        "adam_beta2": 0.999,  # legacy runs used AdamW's plain default
        "explode_factor": 0.0,  # legacy runs had no explode-detection
        "explode_clip_divisor": 5.0,  # inert when explode_factor is 0
    }
