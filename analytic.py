"""
Hand-built exact weights for saturation and obfuscation tasks.

### Saturation
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

### Obfuscation
First MLP block erases c by construction.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from model import ResidualMLP, ResidualMLPConfig
from train_probe import capture_layers

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device).manual_seed(64865313)


def build_exact_model(
    num_x: int, d_model: int, d_mlp: int, num_blocks: int = 4
) -> ResidualMLP:
    assert d_mlp >= num_x, "need at least num_x hidden neurons per block"
    # The exact construction wires blocks 0 and 1; extra blocks stay ~identity.
    assert num_blocks >= 2, "exact construction requires num_blocks >= 2"
    m = ResidualMLP(ResidualMLPConfig(num_x, d_model, d_mlp, num_blocks=num_blocks))
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


def build_exact_obfuscator(
    num_x: int, d_model: int, d_mlp: int, num_blocks: int = 4
) -> ResidualMLP:
    """
    Analytic full obfuscation/erasure of c (that does not try to solve the task).
    """
    assert d_mlp >= 1, "need at least one MLP neuron"
    assert num_blocks >= 1, "exact construction requires num_blocks >= 1"
    m = ResidualMLP(ResidualMLPConfig(num_x, d_model, d_mlp, num_blocks=num_blocks))

    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        c_dir = num_x  # residual/index of the c coordinate

        # Neuron computes -ReLU(c + 10) + 10. For c > -10 (always), this becomes -c.
        m.blocks[0].W_in[c_dir, 0] = 1
        m.blocks[0].b_in[0] = 10
        m.blocks[0].W_out[0, c_dir] = -1
        m.blocks[0].b_out[c_dir] = 10

    return m


def _verify_model(
    num_x: int = 32,
    d_model: int = 512,
    d_mlp: int | None = None,
    n: int = 20_000,
) -> float:
    from data import sample_batch

    if d_mlp is None:
        # Exact construction requires d_mlp == num_x (build_exact_model asserts
        # d_mlp >= num_x, and num_x is the tight minimum).
        d_mlp = num_x
    m = build_exact_model(num_x, d_model, d_mlp)
    x_full, y = sample_batch(n, num_x, generator=generator, device=device)
    with torch.no_grad():
        pred: Float[Tensor, "n num_x"] = m.task_output(x_full)
    max_err = (pred - y).abs().max().item()
    print(
        f"[analytic] num_x={num_x} d_model={d_model} d_mlp={d_mlp} "
        f"n={n}: max abs elementwise error = {max_err:.3e}"
    )
    return max_err


def _verify_obfuscator(
    num_x: int = 32,
    d_model: int = 512,
    d_mlp: int | None = None,
    n: int = 20_000,
) -> float:
    from data import sample_batch

    if d_mlp is None:
        # Exact construction requires d_mlp == num_x (build_exact_model asserts
        # d_mlp >= num_x, and num_x is the tight minimum).
        d_mlp = num_x

    num_blocks = 4
    m = build_exact_obfuscator(num_x, d_model, d_mlp, num_blocks=num_blocks)
    x_full, y = sample_batch(n, num_x, generator=generator, device=device)
    with torch.no_grad():
        pred: Float[Tensor, "n num_x"] = m.task_output(x_full)
    max_err = (pred - y).abs().max().item()
    print(
        f"[analytic] num_x={num_x} d_model={d_model} d_mlp={d_mlp} "
        f"n={n}: max abs elementwise error = {max_err:.3e}"
    )

    acts = capture_layers(m, x_full, layers=torch.arange(1, num_blocks + 1))
    acts = acts.reshape(n, num_blocks, d_model)

    # Activations in first num_x dimensions is x (no info about c)
    assert torch.all(x_full[:, None, :num_x] == acts[:, :, :num_x])

    # Activations in num_x index is zero (c has been cancelled)
    assert torch.max(torch.abs(acts[:, :, num_x])) < 1e-6

    # Activations everywhere else are zero
    assert torch.all(acts[:, :, num_x + 1 :] == 0)


if __name__ == "__main__":
    _verify_model()
    _verify_obfuscator()
