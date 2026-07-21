"""Shared hyperparameters for the probe-obfuscation toy model.

These are defaults. `train.py` accepts CLI overrides (notably --num-x) so we can
run the num_x=1 base case and scale up incrementally without editing this file.
"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import ClassVar, Optional

SEED = 913768

# Data distributions
X_LOW, X_HIGH = -3.0, 3.0  # x ~ U[X_LOW, X_HIGH]
C_LOW, C_HIGH = 1.0, 2.0  # c ~ U[X_LOW, X_HIGH]

# Training
# BATCH_SIZE = 4096
BATCH_SIZE = 4096 * 4
LR = 3e-3
MAX_ITERS = 100_000_000
EARLY_STOP_LOSS = 1e-12  # float32 eps^2 ~ 1.4e-14; 1e-12 is a sane "exact" bar


PROBE_LOSS_CHOICES = ["squared", "absolute", "squared-var", "absolute-std", "lda"]


@dataclass
class ResidualMLPConfig:
    """Architecture + init hyperparameters for ResidualMLP.

    Stored verbatim (as a dict) under checkpoint["config"]. Two distinct
    notions of "default" apply to each optional field below, and they are
    deliberately kept separate:
      - the field default (e.g. `num_blocks: int = 8`) is what a fresh
        ResidualMLPConfig(...) call gets if the field is omitted -- i.e. the
        default for code written going forward.
      - _LEGACY_DEFAULTS is what an *old* checkpoint -- one saved before the
        field existed, so its config dict is missing the key -- is backfilled
        to by from_dict(). This is what that field's value effectively WAS,
        historically, before it became configurable.
    These do NOT all coincide: num_blocks forward-defaults to 8 but
    _LEGACY_DEFAULTS["num_blocks"] stays 4 (old checkpoints were trained with
    4 blocks before the field's default was bumped). Any future change that
    bumps a field's forward default must leave _LEGACY_DEFAULTS alone, so old
    checkpoints keep reconstructing with the architecture they were actually
    trained with.
    """

    num_x: int = 32
    d_model: int = 256
    d_mlp: Optional[int] = None
    num_blocks: int = 8
    out_init_scale: float = 0.1
    leaky_relu_slope: float = 0.0
    layer_norm: bool = False

    # Historical values for fields absent from an old checkpoint's config
    # dict. Every optional field above must have an entry here.
    _LEGACY_DEFAULTS: ClassVar[dict] = {
        "num_blocks": 4,
        "out_init_scale": 0.1,
        "leaky_relu_slope": 0.0,
        "layer_norm": False,
    }

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = self.num_x

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualMLPConfig":
        """Build a config from a checkpoint's config dict: fields the dict
        predates are backfilled from _LEGACY_DEFAULTS (not the field default
        above)."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = d.keys() - known
        if unknown:
            warnings.warn(
                f"ResidualMLPConfig.from_dict: dropping unrecognized key(s) "
                f"{sorted(unknown)} -- checkpoint saved by a newer version?"
            )
        present = {k: v for k, v in d.items() if k in known}
        # union over dicts, preferring `present`
        return cls(**(cls._LEGACY_DEFAULTS | present))


@dataclass
class AdversarialConfig:
    """Training-hyperparameter metadata for train_adversarial.py, stored
    verbatim (as a dict) alongside the model's own ResidualMLPConfig. Kept
    separate from ResidualMLPConfig since the probe-loss variant is a
    training choice, not an architecture choice.

    Same _LEGACY_DEFAULTS / from_dict backfill idiom as ResidualMLPConfig:
    the field default below is what a fresh AdversarialConfig(...) gets when
    the field is omitted; _LEGACY_DEFAULTS is what an old checkpoint (saved
    before the field existed) is backfilled to. Several fields deliberately
    diverge between the two: probe_loss forward-defaults to the new "lda"
    objective, but old checkpoints trained under the hardcoded squared
    penalty are backfilled to "squared" so they reconstruct with the
    objective they were actually trained under. Similarly init/warmstart_path
    now forward-default to a scratch run pointed at the canonical nx32
    checkpoint, while old checkpoints (all warmstart runs with no
    warmstart_path recorded) still backfill to init="warmstart",
    warmstart_path=None; and seed keeps its legacy literal even though the
    forward default now references the module-level SEED constant.
    """

    lam: float = 0.5
    lam_warmup_iters: int = 0
    penalty_layers: Optional[list] = None
    init: str = "scratch"
    warmstart_path: Optional[str] = "runs/nx32/checkpoints/best.pt"
    seed: int = SEED
    probe_loss: str = "lda"
    probe_loss_eps: float = 1e-7
    lda_shrinkage: float = 1e-3
    probe_loss_detach_denom: bool = False

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
    }

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdversarialConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = d.keys() - known
        if unknown:
            warnings.warn(
                f"AdversarialConfig.from_dict: dropping unrecognized key(s) "
                f"{sorted(unknown)} -- checkpoint saved by a newer version?"
            )
        present = {k: v for k, v in d.items() if k in known}
        return cls(**(cls._LEGACY_DEFAULTS | present))
