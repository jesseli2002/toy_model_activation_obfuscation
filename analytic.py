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
"""

import torch

from model import ResidualMLP


def build_exact_model(num_x, d_model, d_mlp):
    assert d_mlp >= num_x, "need at least num_x hidden neurons per block"
    m = ResidualMLP(num_x, d_model, d_mlp)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        c_dir = num_x  # residual/index of the c coordinate
        # Block 0: neuron i computes ReLU(x_i - c), written back with weight -1.
        for i in range(num_x):
            m.W_in[0][i, i] = 1.0
            m.W_in[0][c_dir, i] = -1.0
            m.W_out[0][i, i] = -1.0
        # Block 1: neuron i computes ReLU(-r1[i] - c), written back with weight +1.
        for i in range(num_x):
            m.W_in[1][i, i] = -1.0
            m.W_in[1][c_dir, i] = -1.0
            m.W_out[1][i, i] = 1.0
    return m


def verify(num_x=32, d_model=512, d_mlp=None, n=200_000, seed=0):
    from config import d_mlp_for
    from data import sample_batch

    if d_mlp is None:
        d_mlp = d_mlp_for(num_x)
    m = build_exact_model(num_x, d_model, d_mlp)
    g = torch.Generator().manual_seed(seed)
    x_full, y = sample_batch(n, num_x, generator=g)
    with torch.no_grad():
        pred = m.task_output(x_full)
    max_err = (pred - y).abs().max().item()
    print(
        f"[analytic] num_x={num_x} d_model={d_model} d_mlp={d_mlp} "
        f"n={n}: max abs elementwise error = {max_err:.3e}"
    )
    return max_err


if __name__ == "__main__":
    verify()
