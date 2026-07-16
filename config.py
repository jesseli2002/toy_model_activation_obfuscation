"""Shared hyperparameters for the probe-obfuscation toy model.

These are defaults. `train.py` accepts CLI overrides (notably --num-x) so we can
run the num_x=1 base case and scale up incrementally without editing this file.
"""

SEED = 913768

# Architecture
D_MODEL = 512
NUM_X = 32
D_MLP = NUM_X + 1  # one spare neuron per block beyond the exact-construction need

# Data distributions
X_LOW, X_HIGH = -3.0, 3.0  # x ~ U[-3, 3]
C_LOW, C_HIGH = 1.0, 2.0  # c ~ U[1, 2]

# Training
BATCH_SIZE = 4096
LR = 3e-3
MAX_ITERS = 100_000
EARLY_STOP_LOSS = 1e-12  # float32 eps^2 ~ 1.4e-14; 1e-12 is a sane "exact" bar


def d_mlp_for(num_x: int) -> int:
    """Default d_mlp given num_x (exact construction needs num_x per block)."""
    return num_x + 1
