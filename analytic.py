"""Hand-built exact weights for the saturation task.

Identity used:  sat(x, c) = x - ReLU(x - c) + ReLU(-x - c)   (for c > 0).

  Block 0 (per x-coordinate i, using one hidden neuron each):
      h_i = ReLU(x_i - c)             # W_in0[:,i] = e_i - e_c
      o0 writes -h_i to direction i   # W_out0[i,i] = -1
      => r1[i] = x_i - ReLU(x_i - c) = min(x_i, c);   r1[c] = c  (untouched)

  Block 1 (per x-coordinate i):
      g_i = ReLU(-r1[i] - c)          # W_in1[:,i] = -e_i - e_c
      o1 writes +g_i to direction i   # W_out1[i,i] = +1
      => r2[i] = min(x_i,c) + ReLU(-min(x_i,c)-c) = min(x_i,c) + ReLU(-x_i-c)
              = x_i - ReLU(x_i-c) + ReLU(-x_i-c) = sat(x_i, c)

Needs num_x hidden neurons per block; d_mlp = num_x+1 leaves one spare (left zero).
Use this as a representability anchor and, if optimization struggles, as a warm start.

This construction is ReLU-only: build_exact_model() below hardwires plain ReLU
weights, so warm-starting a leaky-ReLU model (model.py's leaky_relu_slope != 0.0)
with it is only exact at slope 0.0. An exact leaky-ReLU analytic solution does
exist, just not implemented here (it would need d_mlp >= 2*num_x per block instead
of num_x+1). Sketch, using L_a(z) = LeakyReLU(z, negative_slope=a):

    Key identity (exact for any a with a**2 != 1):
        ReLU(z) = [L_a(z) + a * L_a(-z)] / (1 - a**2)
    Proof by cases:
        z >= 0:  L_a(z) = z,    L_a(-z) = -a*z  => numerator = z*(1-a**2) => z = ReLU(z)
        z <  0:  L_a(z) = a*z,  L_a(-z) = -z     => numerator = 0          => 0 = ReLU(z)

    So every single ReLU neuron above can be reproduced exactly by a *pair* of
    leaky-ReLU neurons, one fed z and one fed -z, with output weights scaled by
    1/(1-a**2) and a/(1-a**2) respectively (coeff = the original neuron's output
    weight):
      Block 0, coordinate i (was: input e_i-e_c, output -1):
        neuron A: input  e_i-e_c, output -1/(1-a**2)
        neuron B: input  e_c-e_i, output -a/(1-a**2)
      Block 1, coordinate i (was: input -e_i-e_c, output +1):
        neuron C: input -e_i-e_c, output +1/(1-a**2)
        neuron D: input  e_i+e_c, output +a/(1-a**2)
"""

import torch
from jaxtyping import Float
from torch import Tensor

from model import ResidualMLP


def build_exact_model(num_x: int, d_model: int, d_mlp: int) -> ResidualMLP:
    assert d_mlp >= num_x, "need at least num_x hidden neurons per block"
    m = ResidualMLP(num_x, d_model, d_mlp)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        c_dir = num_x  # residual/index of the c coordinate
        # Block 0: neuron i computes ReLU(x_i - c), written back with weight -1.
        for i in range(num_x):
            m.blocks[0].W_in[i, i] = 1.0
            m.blocks[0].W_in[c_dir, i] = -1.0
            m.blocks[0].W_out[i, i] = -1.0
        # Block 1: neuron i computes ReLU(-r1[i] - c), written back with weight +1.
        for i in range(num_x):
            m.blocks[1].W_in[i, i] = -1.0
            m.blocks[1].W_in[c_dir, i] = -1.0
            m.blocks[1].W_out[i, i] = 1.0
    return m


def verify(
    num_x: int = 32,
    d_model: int = 512,
    d_mlp: int | None = None,
    n: int = 200_000,
    seed: int = 0,
) -> float:
    from config import d_mlp_for
    from data import sample_batch

    if d_mlp is None:
        d_mlp = d_mlp_for(num_x)
    m = build_exact_model(num_x, d_model, d_mlp)
    g = torch.Generator().manual_seed(seed)
    x_full, y = sample_batch(n, num_x, generator=g)
    with torch.no_grad():
        pred: Float[Tensor, "n num_x"] = m.task_output(x_full)
    max_err = (pred - y).abs().max().item()
    print(
        f"[analytic] num_x={num_x} d_model={d_model} d_mlp={d_mlp} "
        f"n={n}: max abs elementwise error = {max_err:.3e}"
    )
    return max_err


if __name__ == "__main__":
    verify()
