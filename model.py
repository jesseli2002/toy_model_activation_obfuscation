"""Residual MLP for the saturation task, with NUM_BLOCKS trainable MLP blocks.

Architecture (row-vector convention, batch on dim 0):
    x_full = [x, c]                      # (B, num_x+1)
    r_0 = x_full @ W_E                   # (B, d_model),  W_E = [I; 0] fixed
    r_{i+1} = r_i + ReLU(r_i @ W_in_i + b_in_i) @ W_out_i + b_out_i   # per block
    y  = r_NUM_BLOCKS @ W_U               # (B, num_x+1),  W_U = W_E^T fixed

W_E / W_U are fixed (non-trainable buffers) with unit-norm orthogonal rows: the
first num_x+1 residual directions are the input coordinates, the rest are unused
at init. Loss is taken over the first num_x outputs only; the c-slot is free.

The nonlinearity is LeakyReLU(negative_slope=leaky_relu_slope); leaky_relu_slope=0.0
(the default) reproduces plain ReLU exactly.
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

NUM_BLOCKS = 2


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, leaky_relu_slope: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.leaky_relu_slope = leaky_relu_slope
        self.W_in = nn.Parameter(torch.empty(d_model, d_mlp))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.empty(d_mlp, d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def reset_parameters(self, out_init_scale: float = 0.1):
        # W_in: standard Kaiming, accounting for the leaky-ReLU negative slope
        # (a=0.0 reduces to the plain-ReLU case).
        nn.init.kaiming_uniform_(
            self.W_in, a=self.leaky_relu_slope, nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.b_in)
        # W_out: small but nonzero. Nonzero so W_in gets gradient at step 0
        # (zeros would stall it); small so the block starts near identity
        # (r_{i+1} ~ r_i), which is already a good init since sat(x,c) = x
        # off-saturation.
        nn.init.normal_(self.W_out, std=out_init_scale / (self.d_mlp**0.5))
        nn.init.zeros_(self.b_out)

    def forward(
        self, r: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        h: Float[Tensor, "batch d_mlp"] = torch.nn.functional.leaky_relu(
            r @ self.W_in + self.b_in, negative_slope=self.leaky_relu_slope
        )
        o: Float[Tensor, "batch d_model"] = h @ self.W_out + self.b_out
        return o


class ResidualMLP(nn.Module):
    def __init__(
        self,
        num_x: int,
        d_model: int,
        d_mlp: int,
        out_init_scale: float = 0.1,
        leaky_relu_slope: float = 0.0,
        num_blocks: int = NUM_BLOCKS,
    ):
        super().__init__()
        self.num_x = num_x
        self.d_in = num_x + 1  # x plus the scalar c
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.leaky_relu_slope = leaky_relu_slope
        self.num_blocks = num_blocks
        assert d_model >= self.d_in, "d_model must fit the input coordinates"

        # Fixed embedding W_E = [I; 0], shape (d_in, d_model); unembed = W_E^T.
        W_E: Float[Tensor, "d_in d_model"] = torch.zeros(self.d_in, d_model)
        W_E[:, : self.d_in] = torch.eye(self.d_in)
        self.register_buffer("W_E", W_E)
        self.register_buffer("W_U", W_E.t().contiguous())  # (d_model, d_in)

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(d_model, d_mlp, leaky_relu_slope)
                for _ in range(num_blocks)
            ]
        )
        self.reset_parameters(out_init_scale)

    def reset_parameters(self, out_init_scale: float = 0.1):
        for block in self.blocks:
            block.reset_parameters(out_init_scale)

    def forward(
        self, x_full: Float[Tensor, "batch d_in"], return_cache: bool = False
    ) -> (
        Float[Tensor, "batch d_in"]
        | tuple[Float[Tensor, "batch d_in"], list[Float[Tensor, "batch d_model"]]]
    ):
        r: Float[Tensor, "batch d_model"] = x_full @ self.W_E
        caches = [r]
        for block in self.blocks:
            r = r + block(r)
            caches.append(r)
        y: Float[Tensor, "batch d_in"] = r @ self.W_U
        if return_cache:
            return y, caches
        return y

    def task_output(
        self, x_full: Float[Tensor, "batch d_in"]
    ) -> Float[Tensor, "batch num_x"]:
        """The first num_x outputs (the part the loss constrains)."""
        return self.forward(x_full)[:, : self.num_x]
