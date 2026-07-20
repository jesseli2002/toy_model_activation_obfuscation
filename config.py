"""Shared hyperparameters for the probe-obfuscation toy model.

These are defaults. `train.py` accepts CLI overrides (notably --num-x) so we can
run the num_x=1 base case and scale up incrementally without editing this file.
"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import ClassVar, Optional

SEED = 913768

# Architecture
D_MODEL = 256
NUM_X = 32
NUM_BLOCKS = 4  # trainable residual MLP blocks (train.py --num-blocks overrides)
# d_mlp is not a module constant: it depends on num_x. Use d_mlp_for(num_x) below
# (train.py resolves --d-mlp through it). The canonical nx32 run uses d_mlp = num_x.

# Data distributions
X_LOW, X_HIGH = -3.0, 3.0  # x ~ U[X_LOW, X_HIGH]
C_LOW, C_HIGH = 1.0, 2.0  # c ~ U[X_LOW, X_HIGH]

# Training
# BATCH_SIZE = 4096
BATCH_SIZE = 4096 * 4
LR = 3e-3
LEAKY_RELU_SLOPE = 0.0  # 0.0 = plain ReLU
MAX_ITERS = 100_000_000
EARLY_STOP_LOSS = 1e-12  # float32 eps^2 ~ 1.4e-14; 1e-12 is a sane "exact" bar


def d_mlp_for(num_x: int) -> int:
    """Default d_mlp given num_x (exact construction needs num_x per block)."""
    return num_x


PROBE_LOSS_CHOICES = ["squared", "absolute", "squared-var", "absolute-std", "lda"]


@dataclass
class AdversarialConfig:
    """Training-hyperparameter metadata for train_adversarial.py, stored
    verbatim (as a dict) alongside the model's own ResidualMLPConfig. Kept
    separate from ResidualMLPConfig since the probe-loss variant is a
    training choice, not an architecture choice.

    Same _LEGACY_DEFAULTS / from_dict backfill idiom as ResidualMLPConfig:
    the field default below is what a fresh AdversarialConfig(...) gets when
    the field is omitted; _LEGACY_DEFAULTS is what an old checkpoint (saved
    before the field existed) is backfilled to. probe_loss is the one field
    where these deliberately diverge: forward default is the new "lda"
    objective, but old checkpoints trained under the hardcoded squared
    penalty are backfilled to "squared" so they reconstruct with the
    objective they were actually trained under.
    """

    lam: float = 0.5
    lam_warmup_iters: int = 0
    penalty_layers: Optional[list] = None
    init: str = "warmstart"
    warmstart_path: Optional[str] = None
    seed: int = 913768
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
