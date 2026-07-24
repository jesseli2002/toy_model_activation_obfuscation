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
    These do NOT always coincide -- see each subclass's docstring for its own
    divergent fields. Any future change that bumps a field's forward default
    must leave _LEGACY_DEFAULTS alone, so old checkpoints keep reconstructing
    with the hyperparameters they were actually run with.

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
    """Architecture + init hyperparameters for ResidualMLP.

    See _CheckpointConfigMixin for the to_dict/from_dict/_LEGACY_DEFAULTS
    mechanism. Here, num_blocks forward-defaults to 8 but
    _LEGACY_DEFAULTS["num_blocks"] stays 4 (old checkpoints were trained with
    4 blocks before the field's default was bumped).
    """

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
    verbatim (as a dict) alongside the model's own ResidualMLPConfig. Kept
    separate from ResidualMLPConfig since the probe-loss variant is a
    training choice, not an architecture choice.

    See _CheckpointConfigMixin for the to_dict/from_dict/_LEGACY_DEFAULTS
    mechanism. Here, probe_loss forward-defaults to the new "lda" objective,
    but old checkpoints trained under the hardcoded squared penalty are
    backfilled to "squared" so they reconstruct with the objective they were
    actually trained under; similarly init forward-defaults to "scratch" but
    old checkpoints backfill to "warmstart".
    """

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


@dataclass
class LogregAdversarialConfig(_CheckpointConfigMixin):
    """Training-hyperparameter metadata for train_adversarial_logreg.py,
    stored verbatim (as a dict) alongside the model's own ResidualMLPConfig.
    Sibling to AdversarialConfig: that one drives the closed-form DoM/LDA
    penalty; this one drives the simultaneous stateful-probe design (see
    train_adversarial_logreg.py's module docstring). warmstart_path is None
    when the model was inited from scratch (`--no-warmstart`) rather than
    warm-started.

    See _CheckpointConfigMixin for the to_dict/from_dict/_LEGACY_DEFAULTS
    mechanism. Here, probe_subsample and probe_retrain_interval both
    forward-default to batching/interval values but backfill to 1 (legacy
    runs fit on the full batch every iteration, with no subsampling or
    interval skipping).
    """

    lam: float = 0.5
    lam_warmup_iters: int = 0
    penalty_layers: list | None = None
    warmstart_path: str | None = None
    seed: int = SEED
    probe_C: float = 1.0
    probe_init_iters: int = 1000
    class_threshold: float = 1.5
    probe_loss_kind: str = "meandiff-relu"
    probe_subsample: int = 8
    probe_retrain_interval: int = 16
    resid_noise_std: float = 0.1

    _LEGACY_DEFAULTS: ClassVar[dict] = {
        "lam": 0.5,
        "lam_warmup_iters": 0,
        "penalty_layers": None,
        "warmstart_path": None,
        "seed": 913768,
        "probe_C": 1.0,
        "probe_init_iters": 1000,
        "class_threshold": 1.5,
        "probe_loss_kind": "meandiff-relu",
        "probe_subsample": 1,  # legacy runs fit on the full batch every step
        "probe_retrain_interval": 1,  # legacy runs refit every iteration
        "resid_noise_std": 0.0,  # legacy runs trained with no residual-stream noise
    }
