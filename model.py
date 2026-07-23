"""Residual MLP for the saturation task, with num_blocks trainable MLP blocks.

Architecture (row-vector convention, batch on dim 0):
    x_full = [x, c]                      # (B, num_x+1)
    r_0 = x_full @ W_E                   # (B, d_model),  W_E = [I; 0] fixed
    r_{i+1} = r_i + ReLU(r_i @ W_in_i + b_in_i) @ W_out_i + b_out_i   # per block
    y  = r_num_blocks @ W_U               # (B, num_x+1),  W_U = W_E^T fixed

W_E / W_U are fixed (non-trainable buffers) with unit-norm orthogonal rows: the
first num_x+1 residual directions are the input coordinates, the rest are unused
at init. Loss is taken over the first num_x outputs only; the c-slot is free.

The nonlinearity is selectable via `activation`:
    - "gelu" (default): GELU (leaky_relu_slope is unused).
    - "leaky_relu": LeakyReLU(negative_slope=leaky_relu_slope);
      leaky_relu_slope=0.0 reproduces plain ReLU exactly.
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from config import ACTIVATION_CHOICES, ResidualMLPConfig


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        assert activation in ACTIVATION_CHOICES, f"unknown activation {activation!r}"
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.activation = activation
        self.leaky_relu_slope = leaky_relu_slope
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None
        self.W_in = nn.Parameter(torch.empty(d_model, d_mlp))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.empty(d_mlp, d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def reset_parameters(self, out_init_scale: float = 0.1):
        # W_in: standard Kaiming. leaky_relu accounts for the negative slope
        # (a=0.0 reduces to the plain-ReLU case); gelu has no dedicated gain
        # in torch, so it's approximated with the plain-ReLU gain (gelu is
        # ReLU-shaped, so this is close enough -- not an exact match).
        if self.activation == "leaky_relu":
            nn.init.kaiming_uniform_(
                self.W_in, a=self.leaky_relu_slope, nonlinearity="leaky_relu"
            )
        else:
            nn.init.kaiming_uniform_(self.W_in, nonlinearity="relu")
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
        r_in: Float[Tensor, "batch d_model"] = (
            self.layer_norm(r) if self.layer_norm is not None else r
        )
        pre_act: Float[Tensor, "batch d_mlp"] = r_in @ self.W_in + self.b_in
        h: Float[Tensor, "batch d_mlp"]
        if self.activation == "leaky_relu":
            h = torch.nn.functional.leaky_relu(
                pre_act, negative_slope=self.leaky_relu_slope
            )
        else:
            h = torch.nn.functional.gelu(pre_act)
        o: Float[Tensor, "batch d_model"] = h @ self.W_out + self.b_out
        return o


class ResidualMLP(nn.Module):
    def __init__(self, config: ResidualMLPConfig):
        super().__init__()
        self.config = config
        self.num_x = config.num_x
        self.d_in = config.num_x + 1  # x plus the scalar c
        self.d_model = config.d_model
        self.d_mlp = config.d_mlp
        self.activation = config.activation
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_blocks = config.num_blocks
        assert config.d_model >= self.d_in, "d_model must fit the input coordinates"

        # Fixed embedding W_E = [I; 0], shape (d_in, d_model); unembed = W_E^T.
        W_E: Float[Tensor, "d_in d_model"] = torch.zeros(self.d_in, config.d_model)
        W_E[:, : self.d_in] = torch.eye(self.d_in)
        self.register_buffer("W_E", W_E)
        self.register_buffer("W_U", W_E.t().contiguous())  # (d_model, d_in)

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    config.d_model,
                    config.d_mlp,
                    config.activation,
                    config.leaky_relu_slope,
                    config.layer_norm,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.reset_parameters(config.out_init_scale)

    def reset_parameters(self, out_init_scale: float = 0.1):
        for block in self.blocks:
            block.reset_parameters(out_init_scale)

    def save(self, path: str, **extra):
        """Save weights + config (and any extra top-level checkpoint fields,
        e.g. optimizer state, training metadata) to `path`."""
        torch.save(
            {"model": self.state_dict(), "config": self.config.to_dict(), **extra},
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> tuple["ResidualMLP", dict]:
        """Load a checkpoint saved by `save()`. Returns (model, full checkpoint
        dict) so callers can still reach any extra fields (opt state, iter,
        training metadata) that rode along in the checkpoint."""
        ck = torch.load(path, map_location=map_location)
        model = cls(ResidualMLPConfig.from_dict(ck["config"]))
        model.load_state_dict(ck["model"])
        return model, ck

    def forward(
        self,
        x_full: Float[Tensor, "batch d_in"],
        return_cache: bool = False,
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> (
        Float[Tensor, "batch d_in"]
        | tuple[Float[Tensor, "batch d_in"], list[Float[Tensor, "batch d_model"]]]
    ):
        """`noise_std > 0` adds absolute Gaussian noise to the residual stream
        after every block except the last, i.e. onto caches 1..num_blocks-1 --
        the penalized hidden layers (see plans/resid_stream_noise_plan.md).
        Embedding (cache 0) and the final residual (cache num_blocks -> y) are
        never injected into directly. `noise_std=0.0` (default) is bit-identical
        to the pre-noise forward."""
        r: Float[Tensor, "batch d_model"] = x_full @ self.W_E
        caches = [r]
        for i, block in enumerate(self.blocks):
            r = r + block(r)
            if noise_std > 0.0 and i + 1 < self.num_blocks:
                r = r + noise_std * torch.randn(
                    r.shape, device=r.device, dtype=r.dtype, generator=generator
                )
            caches.append(r)
        y: Float[Tensor, "batch d_in"] = r @ self.W_U
        if return_cache:
            return y, caches
        return y

    def task_output(
        self,
        x_full: Float[Tensor, "batch d_in"],
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "batch num_x"]:
        """The first num_x outputs (the part the loss constrains)."""
        return self.forward(x_full, noise_std=noise_std, generator=generator)[
            :, : self.num_x
        ]
