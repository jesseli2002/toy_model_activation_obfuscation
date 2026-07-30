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

import sympy as sp
import torch
from jaxtyping import Float
from torch import Tensor

from model import ResidualMLP, ResidualMLPConfig
from probe_lib import capture_layers

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device).manual_seed(64865313)


def build_exact_model(
    num_x: int, d_model: int, d_mlp: int, num_blocks: int = 4
) -> ResidualMLP:
    assert d_mlp >= num_x, "need at least num_x hidden neurons per block"
    # The exact construction wires blocks 0 and 1; extra blocks stay ~identity.
    assert num_blocks >= 2, "exact construction requires num_blocks >= 2"
    # activation pinned to leaky_relu (slope 0 = plain ReLU): the hand-set
    # weights below are an exact ReLU identity, independent of whatever
    # ResidualMLPConfig's own default activation happens to be.
    m = ResidualMLP(
        ResidualMLPConfig(
            num_x=num_x,
            d_model=d_model,
            d_mlp=d_mlp,
            num_blocks=num_blocks,
            activation="leaky_relu",
        )
    )
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
    # activation pinned to leaky_relu (slope 0 = plain ReLU): the hand-set
    # weights below are an exact ReLU identity, independent of whatever
    # ResidualMLPConfig's own default activation happens to be.
    m = ResidualMLP(
        ResidualMLPConfig(
            num_x=num_x,
            d_model=d_model,
            d_mlp=d_mlp,
            num_blocks=num_blocks,
            activation="leaky_relu",
        )
    )

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


def reference_task_losses(
    x_low: float, x_high: float, c_low: float, c_high: float
) -> dict[str, float]:
    """Analytic L_task (MSE against sat(x,-c,c)) for reference solutions the
    model tends to pass through early in training, under x ~ U[x_low,x_high],
    c ~ U[c_low,c_high] (requires x_low == -x_high, 0 < c_low <= c_high <= x_high):

    - do_nothing: pred = x
    - linreg: pred = k*x, with k chosen to minimize the loss (best linear fit)
    - clamp: pred = clip(x, -c_avg, c_avg), c_avg = mean(c)

    Used as reference lines on the training-loss plot in adversarial_report.py.
    """
    assert x_low == -x_high and 0 < c_low <= c_high <= x_high
    x, c = sp.symbols("x c", real=True)
    xl, xh = sp.nsimplify(x_low), sp.nsimplify(x_high)
    cl, ch = sp.nsimplify(c_low), sp.nsimplify(c_high)
    norm = (xh - xl) * (ch - cl)

    def int_x(expr, lo, hi):
        return sp.integrate(expr, (x, lo, hi))

    def avg_c(expr_of_c, lo, hi):
        return sp.integrate(sp.simplify(expr_of_c), (c, lo, hi)) / norm

    # do_nothing: pred=x. Nonzero only outside [-c,c].
    do_nothing_of_c = int_x((x + c) ** 2, xl, -c) + int_x((x - c) ** 2, c, xh)
    do_nothing = sp.nsimplify(sp.simplify(avg_c(do_nothing_of_c, cl, ch)))

    # linreg: pred=k*x, k*=E[xy]/E[x^2] (least-squares slope, no intercept).
    Exy_of_c = int_x(-c * x, xl, -c) + int_x(x * x, -c, c) + int_x(c * x, c, xh)
    Ex2_of_c = int_x(x * x, xl, xh)
    Ey2_of_c = int_x(c**2, xl, -c) + int_x(x * x, -c, c) + int_x(c**2, c, xh)
    Exy, Ex2, Ey2 = (
        sp.nsimplify(sp.simplify(avg_c(e, cl, ch)))
        for e in (Exy_of_c, Ex2_of_c, Ey2_of_c)
    )
    k_star = sp.simplify(Exy / Ex2)
    linreg = sp.nsimplify(sp.simplify(Ey2 - k_star**2 * Ex2))

    # clamp: pred=clip(x,-a,a), a=mean(c). c-integral splits at a since the two
    # region layouts (whether -c/c fall inside or outside [-a,a]) swap there.
    a = (cl + ch) / 2

    def clamp_terms_c_above_a(c_):
        return (
            int_x((-a + c_) ** 2, xl, -c_)
            + int_x((-a - x) ** 2, -c_, -a)
            + int_x(0, -a, a)
            + int_x((a - x) ** 2, a, c_)
            + int_x((a - c_) ** 2, c_, xh)
        )

    def clamp_terms_c_below_a(c_):
        return (
            int_x((-a + c_) ** 2, xl, -a)
            + int_x((x + c_) ** 2, -a, -c_)
            + int_x(0, -c_, c_)
            + int_x((x - c_) ** 2, c_, a)
            + int_x((a - c_) ** 2, a, xh)
        )

    integral_below = sp.integrate(sp.simplify(clamp_terms_c_below_a(c)), (c, cl, a))
    integral_above = sp.integrate(sp.simplify(clamp_terms_c_above_a(c)), (c, a, ch))
    clamp = sp.nsimplify(sp.simplify((integral_below + integral_above) / norm))

    return {
        "do_nothing": float(do_nothing),
        "linreg": float(linreg),
        "clamp": float(clamp),
        "linreg_k": float(k_star),
    }


def no_c_task_loss(
    x_low: float,
    x_high: float,
    c_low: float,
    c_high: float,
    x_p_outer: float | None = None,
    x_threshold: float = 1.0,
) -> float:
    """Analytic minimum per-coordinate L_task over ALL predictors that see x
    but not c -- i.e. a floor no c-blind function can beat, so a model scoring
    below it demonstrably reads c.

    Since c is independent of x, the best such predictor is f(x) = E_c[y] and
    its loss is E_x[Var_c(sat(x,-c,c))], which this integrates exactly (no
    residual-stream noise in the model; the first block can scale the signal
    up arbitrarily, so noise imposes no extra floor here). Compare `clamp` in
    reference_task_losses, which is the same idea restricted to the
    best-single-threshold clip and hence strictly worse.

    x_p_outer/x_threshold mirror data.sample_batch's optional |x|-biased
    sampling (None = plain uniform); they reweight x's density and so change
    the floor.
    """
    assert x_low == -x_high and 0 < c_low <= c_high <= x_high
    x, c = sp.symbols("x c", real=True)
    xh = sp.nsimplify(x_high)
    cl, ch = sp.nsimplify(c_low), sp.nsimplify(c_high)

    def avg_c(expr, lo, hi):
        """E over the sub-range [lo,hi] of c ~ U[cl,ch] (unnormalized by the
        sub-range's own width -- the pieces below sum to a full expectation)."""
        return sp.integrate(expr, (c, lo, hi)) / (ch - cl)

    # sat(x,-c,c) is odd in x, so Var_c is even: integrate over x >= 0 and
    # double. There sat = min(x,c), splitting on where x sits w.r.t. [cl,ch].
    regions = []  # (x_lo, x_hi, Var_c(min(x,c)))
    for lo, hi, mean, sq in [
        (sp.Integer(0), cl, x, x**2),
        (
            cl,
            ch,
            avg_c(c, cl, x) + avg_c(x, x, ch),
            avg_c(c**2, cl, x) + avg_c(x**2, x, ch),
        ),
        (ch, xh, avg_c(c, cl, ch), avg_c(c**2, cl, ch)),
    ]:
        regions.append((lo, hi, sp.simplify(sq - mean**2)))

    # x's density on x > 0 (symmetric), piecewise-constant; `breaks` are where
    # it switches, so the x-integrals below can be split there.
    if x_p_outer is None:
        breaks = []

        def density(_lo, _hi):
            return 1 / (2 * xh)

    else:
        assert 0.0 <= x_p_outer <= 1.0 and 0 < x_threshold < x_high
        p_out = sp.nsimplify(x_p_outer)
        t = sp.nsimplify(x_threshold)
        breaks = [t]

        def density(lo, hi):
            inner = (lo + hi) / 2 < t
            return (1 - p_out) / (2 * t) if inner else p_out / (2 * (xh - t))

    total = sp.Integer(0)
    for lo, hi, var in regions:
        edges = sorted({lo, hi} | {b for b in breaks if lo < b < hi})
        for a, b in zip(edges, edges[1:]):
            total += 2 * density(a, b) * sp.integrate(var, (x, a, b))
    return float(sp.simplify(total))


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
