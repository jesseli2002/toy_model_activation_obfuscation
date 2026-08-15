"""
Closed-form reference constructions and checks for the toy task -- exact
weights, exact loss floors, and exact linear-decodability tests -- none of
which involve a trained model. Everything here answers "what does the math
say should happen" independent of whether training actually finds it, and
serves as ground truth that adversarial_report.py and analytic_demo.py
compare trained-model results against.

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
exist -- each ReLU neuron can be reproduced by a pair of leaky-ReLU neurons fed
z and -z -- but isn't implemented here (it would need d_mlp >= 2*num_x per
block instead of num_x+1).

### Obfuscation
First MLP block erases c by construction.

### v-channels
sample_v_channels()/verify_v_channels() check a different, mean-constant
encoding of c (as opposed to the erase-to-zero one above): (x1, v1, v2)'s
higher moments do shift with c, but neither a difference-of-means nor a
logistic-regression probe can turn that into above-chance decoding -- both
land at chance. See analytic_demo.py's block 0 for where the v1/v2
construction is actually used inside a network, and its "Decode" block for
what it actually takes to read c out linearly (extra ReLU features, not just
(x1, v1, v2)).

### Ideal-y decodability
sample_ideal_y()/verify_ideal_y_decodability() check whether c is linearly
decodable from the *ideal* task output y = clamp(x,-c,c) itself, with no
network involved. y|c is zero-mean and symmetric for every c (x is symmetric
about 0), so no linear direction should separate two c values above chance --
DoM and logistic regression are expected near chance, while a trivial
nonlinear statistic (max_j|y_j|, which converges to c almost surely) is
expected near-perfect. Confirms the task is solvable at all (c can be read
out, just not linearly) and is a baseline for the same question asked of
trained models' internal activations.

Run this file directly to verify every construction above and save the
supporting plots to plot/analytic/.
"""

import os

import sympy as sp
import torch
from jaxtyping import Float
from torch import Tensor

from model import ResidualMLP, ResidualMLPConfig
from probe_lib import capture_layers

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device).manual_seed(64865313)

PLOT_DIR = "plot/analytic"


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


def _no_c_moments(
    x_high: float, c_low: float, c_high: float
) -> tuple[sp.Symbol, list[tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]]]:
    """Piecewise mean/variance over c of sat(x,-c,c), for x >= 0.

    Returns (x_symbol, [(x_lo, x_hi, mean, var), ...]) with c ~ U[c_low,c_high].
    The mean is the best c-blind predictor of the target and the variance is the
    loss it incurs; both halves of that pair are derived here so the analytic
    floor and its plot cannot drift apart. Only x >= 0 is covered: sat is odd in
    x, so the mean is odd and the variance even.
    """
    x, c = sp.symbols("x c", real=True)
    xh = sp.nsimplify(x_high)
    cl, ch = sp.nsimplify(c_low), sp.nsimplify(c_high)

    def avg_c(expr, lo, hi):
        """E over the sub-range [lo,hi] of c ~ U[cl,ch] (unnormalized by the
        sub-range's own width -- the pieces below sum to a full expectation)."""
        return sp.integrate(expr, (c, lo, hi)) / (ch - cl)

    # On x >= 0, sat(x,-c,c) = min(x,c); the pieces split on where x sits
    # w.r.t. [cl,ch] (min is x below cl, c above ch, and mixed in between).
    regions = []
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
        regions.append((lo, hi, sp.simplify(mean), sp.simplify(sq - mean**2)))
    return x, regions


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
    xh = sp.nsimplify(x_high)
    x, regions = _no_c_moments(x_high, c_low, c_high)

    # Var_c is even in x, so integrate over x >= 0 and double.
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
    for lo, hi, _mean, var in regions:
        edges = sorted({lo, hi} | {b for b in breaks if lo < b < hi})
        for a, b in zip(edges, edges[1:]):
            total += 2 * density(a, b) * sp.integrate(var, (x, a, b))
    return float(sp.simplify(total))


def plot_no_c_predictor(
    x_low: float,
    x_high: float,
    c_low: float,
    c_high: float,
    out_path: str = os.path.join(PLOT_DIR, "no_c_predictor.png"),
    n_c: int = 4,
    n_points: int = 1001,
) -> str:
    """Plot the best c-blind predictor f(x) = E_c[sat(x,-c,c)] behind
    no_c_task_loss, against a sample of the sat targets it has to average over.

    Shares _no_c_moments with the loss, so the curve shown is exactly the
    predictor whose loss that function reports. Returns the written path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert x_low == -x_high and 0 < c_low <= c_high <= x_high
    x, regions = _no_c_moments(x_high, c_low, c_high)
    # Piecewise over x >= 0, extended to x < 0 by oddness below.
    mean_pw = sp.Piecewise(
        *[(mean, x <= hi) for _, hi, mean, _ in regions[:-1]], (regions[-1][2], True)
    )
    mean_fn = sp.lambdify(x, mean_pw, "numpy")

    xs = np.linspace(x_low, x_high, n_points)
    pred = np.sign(xs) * mean_fn(np.abs(xs))
    loss = no_c_task_loss(x_low, x_high, c_low, c_high)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, c_val in enumerate(np.linspace(c_low, c_high, n_c)):
        ax.plot(
            xs,
            np.clip(xs, -c_val, c_val),
            linestyle="--",
            color="black",
            linewidth=1.0,
            alpha=0.7,
            label="sat(x,-c,c) for sampled c" if i == 0 else None,
        )
    ax.plot(
        xs, pred, color="tab:blue", linewidth=2.2, label=r"$E_c[\mathrm{sat}(x,-c,c)]$"
    )
    ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.set_xlabel("x")
    ax.set_ylabel("prediction")
    ax.set_title(
        f"Best c-blind predictor, c ~ U[{c_low:g}, {c_high:g}]\n"
        f"per-coordinate L_task floor = {loss:.4f}"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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
    m = build_exact_model(num_x, d_model, d_mlp).to(device)
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
) -> tuple[bool, str]:
    """Build the obfuscator and check the three residual invariants: x is
    passed through verbatim, c is erased (~0), and every other scratch
    coordinate stays zero. Its task-output error is not a pass criterion --
    build_exact_obfuscator deliberately does not solve the task, only erases
    c (see the module docstring)."""
    from data import sample_batch

    if d_mlp is None:
        # Exact construction requires d_mlp == num_x (build_exact_model asserts
        # d_mlp >= num_x, and num_x is the tight minimum).
        d_mlp = num_x

    num_blocks = 4
    m = build_exact_obfuscator(num_x, d_model, d_mlp, num_blocks=num_blocks).to(device)
    x_full, y = sample_batch(n, num_x, generator=generator, device=device)
    acts = capture_layers(m, x_full, layers=list(range(1, num_blocks + 1)))
    acts = acts.reshape(n, num_blocks, d_model)

    x_preserved = bool(torch.all(x_full[:, None, :num_x] == acts[:, :, :num_x]))
    c_erased = torch.max(torch.abs(acts[:, :, num_x])).item()
    rest_zero = bool(torch.all(acts[:, :, num_x + 1 :] == 0))
    ok = x_preserved and c_erased < 1e-6 and rest_zero
    detail = (
        f"x preserved={x_preserved}, max|c dim|={c_erased:.2e}, rest zero={rest_zero}"
    )
    print(f"[analytic] num_x={num_x} d_model={d_model} d_mlp={d_mlp} n={n}: {detail}")
    return ok, detail


def sample_v_channels(
    n: int, c: float, rng: "np.random.Generator"
) -> "Float[np.ndarray, 'n 3']":
    """(x1, v1, v2): x1 ~ U[-3, 3], v1/v2 built from four ReLUs exactly as
    analytic_demo.py's block 0 wires them (kept as a local literal copy here
    rather than an import, since analytic_demo.py sets torch's default dtype
    to float64 as an import-time side effect that this module must not
    inherit). E_x1[v] = 0 for every c by construction, so a raw difference-of-means
    probe sits at chance; mean-constancy says nothing about higher moments --
    std(v2) and cov(v1,v2) do shift with c (verified numerically) -- but a
    linear probe still can't turn that into decision-relevant separation:
    logistic regression on (x1, v1, v2) also lands at chance (checked across
    seeds and a wide range of C, not just one run). That's the actual point
    of the construction: linearly decoding c needs the extra decode ReLUs
    analytic_demo.py adds, not just (x1, v1, v2). Domain-specific to x1 in
    [-3,3], c in [1,2]; not meant to be reparametrized.
    """
    import numpy as np

    relu = lambda z: np.maximum(z, 0.0)
    x1 = rng.uniform(-3, 3, n)
    v1 = -2 * relu(-x1 - c) + 2 * relu(x1 - 3 + c) - c + 1.5
    v2 = -4 * relu(-x1 - c / 2) + 4 * relu(x1 + c / 2 - 3) - c + 3.0
    return np.stack([x1, v1, v2], axis=1)


def _dom_probe(X, y) -> tuple["np.ndarray", float]:
    """Difference-of-means direction/threshold, midpoint-of-means bias."""
    mu0, mu1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    w = mu1 - mu0
    b = -0.5 * (mu0 + mu1) @ w
    return w, b


def verify_v_channels(
    c_lo: float = 1.0,
    c_hi: float = 2.0,
    n: int = 20_000,
    seed: int = 0,
    plot_dir: str = PLOT_DIR,
) -> dict[str, float]:
    """Check that the (x1, v1, v2) mean-constant encoding is *not* linearly
    decodable: both a difference-of-means probe and a logistic regression
    probe are expected near chance, even though std(v2) and cov(v1,v2) shift
    with c -- a linear model can't turn that shift into separation on its
    own. Probes get x1 as well as v1, v2: x1 is itself a residual
    coordinate, and disambiguating which x1-band a sample falls in is part of
    decoding c from the encoding, so omitting it would understate what a real
    probe sees. No trained network involved -- just the closed-form encoding,
    sampled and fed straight to both classifiers. Saves the supporting
    figures to plot_dir and returns both AUROCs.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(seed)

    def make_split():
        X = np.concatenate(
            [sample_v_channels(n, c_lo, rng), sample_v_channels(n, c_hi, rng)]
        )
        y = np.concatenate([np.zeros(n), np.ones(n)])
        return X, y

    X_tr, y_tr = make_split()
    X_te, y_te = make_split()

    w_dom, b_dom = _dom_probe(X_tr, y_tr)
    dom_scores = X_te @ w_dom + b_dom
    dom_auroc = float(roc_auc_score(y_te, dom_scores))

    clf = LogisticRegression(C=1000).fit(X_tr, y_tr)
    w_lr, b_lr = clf.coef_[0], float(clf.intercept_[0])
    lr_scores = X_te @ w_lr + b_lr
    logreg_auroc = float(roc_auc_score(y_te, lr_scores))

    print(
        f"[analytic] v-channels (x1,v1,v2), c={c_lo:g} vs c={c_hi:g}: "
        f"DoM AUROC={dom_auroc:.4f}  logreg AUROC={logreg_auroc:.4f}"
    )

    os.makedirs(plot_dir, exist_ok=True)
    lo_label, hi_label = f"c={c_lo:g}", f"c={c_hi:g}"

    # --- scatter of (v1, v2) with both decision boundaries, projected at mean x1 ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X_te[y_te == 0, 1:].T, s=4, alpha=0.01, label=lo_label)
    ax.scatter(*X_te[y_te == 1, 1:].T, s=4, alpha=0.01, label=hi_label)
    v1g = np.linspace(X_te[:, 1].min(), X_te[:, 1].max(), 200)
    x1_mean = X_te[:, 0].mean()
    for w, b, style, name in [
        (w_dom, b_dom, "k--", f"DoM (AUROC={dom_auroc:.3f})"),
        (w_lr, b_lr, "r-", f"logreg (AUROC={logreg_auroc:.3f})"),
    ]:
        if abs(w[2]) > 1e-9:
            v2g = -(w[0] * x1_mean + w[1] * v1g + b) / w[2]
            ax.plot(v1g, v2g, style, label=name, alpha=0.7)
    ax.set_xlabel("v1")
    ax.set_ylabel("v2")
    ax.set_title(f"v1/v2 encoding, {lo_label} vs {hi_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "v_channels_2d.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[analytic] wrote {path}")

    # --- histograms of each probe's 1D projection ---
    fig, (ax_dom, ax_lr) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, w, b, name, auroc in [
        (ax_dom, w_dom, b_dom, "difference of means", dom_auroc),
        (ax_lr, w_lr, b_lr, "logistic regression", logreg_auroc),
    ]:
        proj_lo = X_te[y_te == 0] @ w + b
        proj_hi = X_te[y_te == 1] @ w + b
        lo, hi = min(proj_lo.min(), proj_hi.min()), max(proj_lo.max(), proj_hi.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(proj_lo, bins=bins, alpha=0.5, label=lo_label)
        ax.hist(proj_hi, bins=bins, alpha=0.5, label=hi_label)
        ax.axvline(0.0, color="k", ls="--", lw=1)
        ax.set_title(f"{name}\nAUROC={auroc:.3f}")
        ax.set_xlabel("projection (boundary at 0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax_dom.set_ylabel("count")
    fig.tight_layout()
    path = os.path.join(plot_dir, "v_channels_hist.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[analytic] wrote {path}")

    # --- ROC curves for both probes ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for scores, name, auroc, color in [
        (dom_scores, "DoM", dom_auroc, "k"),
        (lr_scores, "logreg", logreg_auroc, "r"),
    ]:
        fpr, tpr, _ = roc_curve(y_te, scores)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("FPR (false positive rate)")
    ax.set_ylabel("TPR (true positive rate)")
    ax.set_title(f"ROC curves, {lo_label} vs {hi_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(plot_dir, "v_channels_roc.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[analytic] wrote {path}")

    return {"dom_auroc": dom_auroc, "logreg_auroc": logreg_auroc}


def sample_ideal_y(
    n: int,
    num_x: int,
    c: float,
    x_low: float,
    x_high: float,
    rng: "np.random.Generator",
) -> "Float[np.ndarray, 'n num_x']":
    """y = clamp(x,-c,c), x ~ U[x_low,x_high]^num_x -- the data-generating
    process's ideal output, with no network involved."""
    import numpy as np

    x = rng.uniform(x_low, x_high, size=(n, num_x))
    return np.clip(x, -c, c)


def verify_ideal_y_decodability(
    num_x: int = 32,
    c_lo: float = 1.0,
    c_hi: float = 2.0,
    x_low: float = -3.0,
    x_high: float = 3.0,
    n_train: int = 20_000,
    n_test: int = 50_000,
    seed: int = 0,
) -> dict[str, float]:
    """Is c linearly decodable from the *ideal* task output y alone? y|c is
    zero-mean and symmetric for every c (x is symmetric about 0), so no
    linear direction (raw DoM or logistic regression) should separate c_lo
    from c_hi above chance -- the class-conditional means coincide. Contrast
    with the nonlinear statistic max_j|y_j|, which converges to c almost
    surely (c IS the clip boundary), to confirm the gap is "linear vs
    nonlinear", not "no signal at all". Mirrors the probe harness's binary
    dataset construction, minus any trained model.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)

    y_lo_tr = sample_ideal_y(n_train, num_x, c_lo, x_low, x_high, rng)
    y_hi_tr = sample_ideal_y(n_train, num_x, c_hi, x_low, x_high, rng)
    y_lo_te = sample_ideal_y(n_test, num_x, c_lo, x_low, x_high, rng)
    y_hi_te = sample_ideal_y(n_test, num_x, c_hi, x_low, x_high, rng)

    X_train = np.concatenate([y_lo_tr, y_hi_tr], axis=0)
    y_train = np.concatenate([np.zeros(n_train), np.ones(n_train)])
    X_test = np.concatenate([y_lo_te, y_hi_te], axis=0)
    y_test = np.concatenate([np.zeros(n_test), np.ones(n_test)])

    # --- raw difference-of-means ---
    mu_lo, mu_hi = y_lo_tr.mean(axis=0), y_hi_tr.mean(axis=0)
    w_dom = mu_hi - mu_lo
    midpoint = ((mu_hi + mu_lo) / 2) @ w_dom
    dom_acc = float(((X_test @ w_dom > midpoint).astype(float) == y_test).mean())

    # --- logistic regression (linear, standardized) ---
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logreg.fit(X_train, y_train)
    logreg_acc = float(logreg.score(X_test, y_test))

    # --- nonlinear contrast: max|y_j| as a trivial c-estimator ---
    max_abs_lo = np.abs(y_lo_te).max(axis=1)
    max_abs_hi = np.abs(y_hi_te).max(axis=1)
    threshold = (c_lo + c_hi) / 2
    nonlinear_pred = (np.concatenate([max_abs_lo, max_abs_hi]) > threshold).astype(
        float
    )
    nonlinear_acc = float((nonlinear_pred == y_test).mean())

    print(
        f"[analytic] ideal-y decodability: num_x={num_x} x=[{x_low},{x_high}] "
        f"c_lo={c_lo} c_hi={c_hi} n_train={n_train}/class n_test={n_test}/class: "
        f"DoM acc={dom_acc:.4f}  logreg acc={logreg_acc:.4f}  "
        f"max|y_j| acc={nonlinear_acc:.4f}"
    )
    return {
        "dom_acc": dom_acc,
        "logreg_acc": logreg_acc,
        "nonlinear_acc": nonlinear_acc,
    }


def _run_checks(
    bounds: tuple[float, float, float, float],
) -> list[tuple[str, bool, str]]:
    """Run every verification check in this module and return
    (name, passed, detail) rows for the console report."""
    checks = []

    max_err = _verify_model()
    checks.append(
        (
            "exact model reproduces sat(x,-c,c)",
            max_err < 1e-4,
            f"max abs err = {max_err:.3e}",
        )
    )

    obf_ok, obf_detail = _verify_obfuscator()
    checks.append(("obfuscator erases c from every probed layer", obf_ok, obf_detail))

    losses = reference_task_losses(*bounds)
    floor = no_c_task_loss(*bounds)
    ordered = losses["do_nothing"] > losses["linreg"] > losses["clamp"] > floor
    checks.append(
        (
            "loss ordering do_nothing > linreg > clamp > c-blind floor",
            ordered,
            f"{losses['do_nothing']:.4f} > {losses['linreg']:.4f} > "
            f"{losses['clamp']:.4f} > {floor:.4f}",
        )
    )

    v = verify_v_channels()
    # Both probes are expected near chance -- that's the whole point of the
    # encoding: (x1, v1, v2) alone isn't linearly decodable, even though
    # std(v2) and cov(v1,v2) do shift with c (confirmed at n=2e6 in review).
    # The band is loose (not 0.02) because at n=20_000 sampling noise alone
    # moves either AUROC by ~0.05-0.1 seed to seed -- verified empirically,
    # not guessed.
    for name, auroc in [
        ("v-channels: DoM probe near chance", v["dom_auroc"]),
        ("v-channels: logreg probe near chance", v["logreg_auroc"]),
    ]:
        checks.append((name, abs(auroc - 0.5) < 0.15, f"AUROC={auroc:.4f}"))

    iy = verify_ideal_y_decodability()
    checks.append(
        (
            "ideal-y: DoM probe near chance",
            abs(iy["dom_acc"] - 0.5) < 0.1,
            f"acc={iy['dom_acc']:.4f}",
        )
    )
    checks.append(
        (
            "ideal-y: logreg probe near chance",
            abs(iy["logreg_acc"] - 0.5) < 0.1,
            f"acc={iy['logreg_acc']:.4f}",
        )
    )
    checks.append(
        (
            "ideal-y: max|y_j| nonlinear estimator near-perfect",
            iy["nonlinear_acc"] > 0.95,
            f"acc={iy['nonlinear_acc']:.4f}",
        )
    )

    return checks


if __name__ == "__main__":
    import sys

    import config

    bounds = (config.X_LOW, config.X_HIGH, config.C_LOW, config.C_HIGH)
    print("reference task losses:")
    print(reference_task_losses(*bounds))
    print()
    print(f"[analytic] c-blind minimum L_task = {no_c_task_loss(*bounds):.6f}")
    print(f"[analytic] wrote {plot_no_c_predictor(*bounds)}")
    print()

    checks = _run_checks(bounds)
    print("\n== verification report ==")
    name_width = max(len(name) for name, _, _ in checks)
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{name_width}}  {detail}")
    if not all_ok:
        sys.exit(1)
