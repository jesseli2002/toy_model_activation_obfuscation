"""2-layer residual MLP for the saturation task.

Architecture (row-vector convention, batch on dim 0):
    x_full = [x, c]                      # (B, num_x+1)
    r0 = x_full @ W_E                    # (B, d_model),  W_E = [I; 0] fixed
    o0 = ReLU(r0 @ W_in0 + b_in0) @ W_out0 + b_out0
    r1 = r0 + o0
    o1 = ReLU(r1 @ W_in1 + b_in1) @ W_out1 + b_out1
    r2 = r1 + o1
    y  = r2 @ W_U                        # (B, num_x+1),  W_U = W_E^T fixed

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


class ResidualMLP(nn.Module):
    def __init__(
        self,
        num_x: int,
        d_model: int,
        d_mlp: int,
        out_init_scale: float = 0.1,
        leaky_relu_slope: float = 0.0,
    ):
        super().__init__()
        self.num_x = num_x
        self.d_in = num_x + 1  # x plus the scalar c
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.leaky_relu_slope = leaky_relu_slope
        assert d_model >= self.d_in, "d_model must fit the input coordinates"

        # Fixed embedding W_E = [I; 0], shape (d_in, d_model); unembed = W_E^T.
        W_E: Float[Tensor, "d_in d_model"] = torch.zeros(self.d_in, d_model)
        W_E[:, : self.d_in] = torch.eye(self.d_in)
        self.register_buffer("W_E", W_E)
        self.register_buffer("W_U", W_E.t().contiguous())  # (d_model, d_in)

        # Two trainable MLP blocks.
        self.W_in = nn.ParameterList(
            [nn.Parameter(torch.empty(d_model, d_mlp)) for _ in range(2)]
        )
        self.b_in = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_mlp)) for _ in range(2)]
        )
        self.W_out = nn.ParameterList(
            [nn.Parameter(torch.empty(d_mlp, d_model)) for _ in range(2)]
        )
        self.b_out = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model)) for _ in range(2)]
        )
        self.reset_parameters(out_init_scale)

    def reset_parameters(self, out_init_scale: float = 0.1):
        for i in range(2):
            # W_in: standard Kaiming, accounting for the leaky-ReLU negative slope
            # (a=0.0 reduces to the plain-ReLU case).
            nn.init.kaiming_uniform_(
                self.W_in[i], a=self.leaky_relu_slope, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.b_in[i])
            # W_out: small but nonzero. Nonzero so W_in gets gradient at step 0
            # (zeros would stall it); small so blocks start near identity (r2 ~ r0,
            # y ~ x), which is already a good init since sat(x,c) = x off-saturation.
            nn.init.normal_(self.W_out[i], std=out_init_scale / (self.d_mlp**0.5))
            nn.init.zeros_(self.b_out[i])

    def forward(
        self, x_full: Float[Tensor, "batch d_in"], return_cache: bool = False
    ) -> (
        Float[Tensor, "batch d_in"]
        | tuple[Float[Tensor, "batch d_in"], list[Float[Tensor, "batch d_model"]]]
    ):
        r0: Float[Tensor, "batch d_model"] = x_full @ self.W_E
        h0: Float[Tensor, "batch d_mlp"] = torch.nn.functional.leaky_relu(
            r0 @ self.W_in[0] + self.b_in[0], negative_slope=self.leaky_relu_slope
        )
        o0: Float[Tensor, "batch d_model"] = h0 @ self.W_out[0] + self.b_out[0]
        r1: Float[Tensor, "batch d_model"] = r0 + o0
        h1: Float[Tensor, "batch d_mlp"] = torch.nn.functional.leaky_relu(
            r1 @ self.W_in[1] + self.b_in[1], negative_slope=self.leaky_relu_slope
        )
        o1: Float[Tensor, "batch d_model"] = h1 @ self.W_out[1] + self.b_out[1]
        r2: Float[Tensor, "batch d_model"] = r1 + o1
        y: Float[Tensor, "batch d_in"] = r2 @ self.W_U
        if return_cache:
            return y, [r0, r1, r2]
        return y

    def task_output(
        self, x_full: Float[Tensor, "batch d_in"]
    ) -> Float[Tensor, "batch num_x"]:
        """The first num_x outputs (the part the loss constrains)."""
        return self.forward(x_full)[:, : self.num_x]
