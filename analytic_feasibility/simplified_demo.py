"""Self-contained, minimal demo of an analytic construction where a residual
MLP computes `y = sat(x, -c, +c)` exactly while every *probed* layer stays
mean-constant in `c` -- so no affine (difference-of-means or logistic
regression) probe can recover `c`, even though the model needs it to compute
`y`. See the accompanying blog post for the derivation; this script just
builds the network, checks it is exact, and probes it.

1. **Encode** (block 0). Two channels `v1(x1, c)`, `v2(x1, c)` built from four
   ReLUs, with `E_x1[v] = 0` for every `c` -- so they carry `c` without moving
   any residual mean. Their four kinks cut `x1` into 5 bands whose
   left-to-right order is fixed over `c in [1, 2]`. `c` is then erased from
   the residual.
2. **Decode** (unprobed block). Inside a band, `c` is an affine function of
   `(x1, v1, v2)` -- but we cannot tell which band we are in without `c`. The
   fix: three ReLUs `R_i = relu(P_i(x1, v1, v2))` whose `P_i` vanish
   identically along an existing kink curve, so they add no new kinks.
   Together with `{1, x1, v1, v2}` they span the full space of piecewise
   linear functions on the 5 bands, so `c` is an exact *affine* read-out of
   `(x1, v1, v2, R_1, R_2, R_3)`.
3. **Use + clear** (probed block). The next block reads `c` as a pre-activation
   (folding the affine read-out into its input matrix), finishes some `x`
   coordinates, and zeroes the `R_i` dims -- so the probed residual is back to
   holding only `[sat-done x, pending x, 0, v1, v2, 0]`.

Outputs: exactness checks, per-layer difference-of-means / logistic-regression
probe AUROCs, and plots.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-x", type=int, default=16, help="x coordinates (even)")
    p.add_argument("--c-lo", type=float, default=1.0, help="probe negative class")
    p.add_argument("--c-hi", type=float, default=2.0, help="probe positive class")
    p.add_argument("--n", type=int, default=20_000, help="samples per class per split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir", type=str, default=".", help="Directory where plots are saved to"
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import os
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import Tensor

# The construction is exact, so it is worth checking at float64 precision --
# errors land at ~1e-15 rather than float32's ~1e-6.
torch.set_default_dtype(torch.float64)

X_LO, X_HI = -3.0, 3.0  # x_i ~ U[X_LO, X_HI]
C_LO, C_HI = 1.0, 2.0  # c ~ U[C_LO, C_HI]


# --------------------------------------------------------------------------
# 0. The architecture: a plain ReLU residual MLP. Weights are set analytically
#    below rather than trained, so they are buffers, not Parameters.
# --------------------------------------------------------------------------
class ResidualMLPBlock(nn.Module):
    """r <- r + relu(r @ W_in + b_in) @ W_out + b_out"""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.register_buffer("W_in", torch.zeros(d_model, d_mlp))
        self.register_buffer("b_in", torch.zeros(d_mlp))
        self.register_buffer("W_out", torch.zeros(d_mlp, d_model))
        self.register_buffer("b_out", torch.zeros(d_model))

    def forward(
        self, r: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        return torch.relu(r @ self.W_in + self.b_in) @ self.W_out + self.b_out

    def neurons_used(self) -> int:
        return int((self.W_in.abs().sum(0) + self.b_in.abs() > 0).sum())


class ResidualMLP(nn.Module):
    """Input `[x, c]` is embedded by a fixed rectangular identity `W_E = [I; 0]`,
    then `num_blocks` residual blocks run; `y` is the first `num_x` residual
    dims. Residual dims beyond `num_x + 1` start at zero and are free scratch.
    """

    def __init__(self, num_x: int, d_model: int, d_mlp: int, num_blocks: int):
        super().__init__()
        self.num_x = num_x
        self.d_in = num_x + 1  # x plus the scalar c
        self.d_model = d_model
        W_E: Float[Tensor, "d_in d_model"] = torch.zeros(self.d_in, d_model)
        W_E[:, : self.d_in] = torch.eye(self.d_in)
        self.register_buffer("W_E", W_E)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(d_model, d_mlp) for _ in range(num_blocks)
        )

    def residuals(
        self, x: Float[Tensor, "batch num_x"], c: Float[Tensor, " batch"]
    ) -> list[Float[Tensor, "batch d_model"]]:
        """[r_0, r_1, ..., r_L]; r_k is the residual after block k-1."""
        r = torch.cat([x, c[:, None]], dim=1) @ self.W_E
        caches = [r]
        for block in self.blocks:
            r = r + block(r)
            caches.append(r)
        return caches

    def task_output(
        self, x: Float[Tensor, "batch num_x"], c: Float[Tensor, " batch"]
    ) -> Float[Tensor, "batch num_x"]:
        return self.residuals(x, c)[-1][:, : self.num_x]


# --------------------------------------------------------------------------
# 1. Decode: three ReLUs that add no new kinks
# --------------------------------------------------------------------------
# (v1, v2, the encoding channels from block 0 below, are built directly as
# weights -- there is no standalone Python function for them, since nothing
# here needs to evaluate them outside the network itself.)
# Each P_i = a0 + a1*x1 + a2*v1 + a3*v2 is built to vanish identically along
# one kink curve, so relu(P_i) is one-sided there and introduces no kink of
# its own. Only two of the four kink curves admit such a P; see the blog post
# for the derivation. The three chosen P_i:
#   P_1 = 2*x1 - 2*v1 + v2   (vanishes at x1 = -c/2)
#   P_2 = -2*x1 + v1 - 1.5   (vanishes at x1 = -c/2)
#   P_3 = v1 - 1.5           (vanishes at x1 = 3-c/2)
P_COEFFS = [  # each (a0, a1, a2, a3)
    (0.0, 2.0, -2.0, 1.0),
    (-1.5, -2.0, 1.0, 0.0),
    (-1.5, 0.0, 1.0, 0.0),
]

# c = w . [1, x1, v1, v2, relu(P_1), relu(P_2), relu(P_3)], exactly.
DECODE_W = torch.tensor([3.0, 4.0, -4.0, 1.0, -2.0, 4.0, 2.0])


# --------------------------------------------------------------------------
# 2. Building the literal network
# --------------------------------------------------------------------------
# Residual layout: [x_0 .. x_{n-1}] [c] [v1] [v2] [R_0 R_1 R_2]
ERASE_BIG = 10.0  # always-on offset: relu(z + ERASE_BIG) = z + ERASE_BIG


@dataclass
class Schedule:
    """Which coordinate is finished by which block, and which layers are probed.

    block 0     -> r1  unprobed: write v1, v2 (c still linear here)
    block 1     -> r2  probed:   finish `first_batch` coords with linear c,
                                 erase c
    block 2k    -> r_{2k+1} unprobed: write the 3 decode ReLUs R_1..R_3
    block 2k+1  -> r_{2k+2} probed:   finish `per_period` coords via the
                                      affine c read-out, clear the R_i dims
    """

    num_x: int
    d_mlp: int
    first_batch: int
    per_period: int
    num_blocks: int

    def is_probed(self, layer: int) -> bool:
        """Probed layers are the even ones (r_0 holds c by construction)."""
        return layer > 0 and layer % 2 == 0

    def n_finished(self, layer: int) -> int:
        """How many coordinates are saturated at (probed) `layer`."""
        return min(
            self.first_batch + max(0, (layer - 2) // 2) * self.per_period, self.num_x
        )


def build_network(num_x: int) -> tuple[ResidualMLP, Schedule]:
    assert num_x % 2 == 0, "num_x must be even (d_mlp = num_x / 2)"
    d_mlp = num_x // 2
    C, V1, V2, R0 = num_x, num_x + 1, num_x + 2, num_x + 3
    d_model = R0 + 3
    # Neuron budget: 2 relus per finished coord, plus 1 erasure/clear each.
    first_batch = min((d_mlp - 1) // 2, num_x - 1)
    per_period = (d_mlp - 3) // 2
    assert first_batch >= 1 and per_period >= 1, "d_mlp too small"
    # x1 (coord 0) gates the decode, so it must be saturated last.
    coord_order = list(range(1, num_x)) + [0]
    batches = [
        coord_order[i : i + per_period] for i in range(first_batch, num_x, per_period)
    ]
    sched = Schedule(
        num_x=num_x,
        d_mlp=d_mlp,
        first_batch=first_batch,
        per_period=per_period,
        num_blocks=2 + 2 * len(batches),
    )
    net = ResidualMLP(num_x, d_model, d_mlp, sched.num_blocks)
    blocks = iter(net.blocks)

    # --- block 0: write v1, v2 (4 kinked neurons + 1 always-on) ---
    blk = next(blocks)
    blk.W_in[0, 0], blk.W_in[C, 0] = -1, -1  # relu(-x1 - c)
    blk.W_in[0, 1], blk.W_in[C, 1], blk.b_in[1] = 1, 1, -3  # relu(x1 + c - 3)
    blk.W_in[0, 2], blk.W_in[C, 2] = -1, -0.5  # relu(-x1 - c/2)
    blk.W_in[0, 3], blk.W_in[C, 3], blk.b_in[3] = 1, 0.5, -3  # relu(x1 + c/2 - 3)
    blk.W_in[C, 4], blk.b_in[4] = 1, ERASE_BIG  # always-on, carries -c
    blk.W_out[0, V1], blk.W_out[1, V1], blk.W_out[4, V1] = -2, 2, -1
    blk.b_out[V1] = ERASE_BIG + 1.5
    blk.W_out[2, V2], blk.W_out[3, V2], blk.W_out[4, V2] = -4, 4, -1
    blk.b_out[V2] = ERASE_BIG + 3.0

    # --- block 1: finish the first batch with linear c, then erase c ---
    blk = next(blocks)
    for m, i in enumerate(coord_order[:first_batch]):
        blk.W_in[i, 2 * m], blk.W_in[C, 2 * m] = 1, -1  # relu(x_i - c)
        blk.W_in[i, 2 * m + 1], blk.W_in[C, 2 * m + 1] = -1, -1  # relu(-x_i - c)
        blk.W_out[2 * m, i], blk.W_out[2 * m + 1, i] = -1, 1  # o[i] = sat(x_i) - x_i
    e = 2 * first_batch
    blk.W_in[C, e], blk.b_in[e] = 1, ERASE_BIG
    blk.W_out[e, C], blk.b_out[C] = -1, ERASE_BIG

    # --- periodic blocks ---
    # c_hat is an affine read of the residual, so relu(x_i - c_hat) is a legal
    # pre-activation for the block right after R_1..R_3 are written.
    chat_vec = torch.zeros(d_model)
    chat_vec[0], chat_vec[V1], chat_vec[V2] = DECODE_W[1], DECODE_W[2], DECODE_W[3]
    chat_vec[R0 : R0 + 3] = DECODE_W[4:]
    chat_const = DECODE_W[0]
    for batch in batches:
        # decode block: R_i = relu(P_i(x1, v1, v2))  (3 neurons)
        blk = next(blocks)
        for j, (a0, a1, a2, a3) in enumerate(P_COEFFS):
            blk.W_in[0, j], blk.W_in[V1, j], blk.W_in[V2, j] = a1, a2, a3
            blk.b_in[j] = a0
            blk.W_out[j, R0 + j] = 1

        # use + clear block
        blk = next(blocks)
        for m, i in enumerate(batch):
            blk.W_in[:, 2 * m] = -chat_vec  # relu(x_i - c_hat)
            blk.W_in[i, 2 * m] += 1
            blk.b_in[2 * m] = -chat_const
            blk.W_in[:, 2 * m + 1] = -chat_vec  # relu(-x_i - c_hat)
            blk.W_in[i, 2 * m + 1] += -1
            blk.b_in[2 * m + 1] = -chat_const
            blk.W_out[2 * m, i], blk.W_out[2 * m + 1, i] = -1, 1
        for j in range(3):  # clear R_j with an always-on neuron
            nz = 2 * len(batch) + j
            blk.W_in[R0 + j, nz], blk.b_in[nz] = 1, 1
            blk.W_out[nz, R0 + j], blk.b_out[R0 + j] = -1, 1

    return net, sched


# --------------------------------------------------------------------------
# 3. Exactness check
# --------------------------------------------------------------------------
def check_network(net: ResidualMLP, sched: Schedule, gen, n_samples: int = 50_000):
    """Verify the network computes sat(x, -c, c) exactly. (Whether `c` is
    detectable at each layer is what the probes below are for.)"""
    max_used = max(b.neurons_used() for b in net.blocks)
    print(
        f"network: num_x={sched.num_x}, d_mlp={sched.d_mlp} (= num_x/2), "
        f"d_model={net.d_model}, blocks={sched.num_blocks}, "
        f"max neurons used in any block = {max_used} (budget {sched.d_mlp})"
    )
    x = uniform((n_samples, sched.num_x), X_LO, X_HI, gen)
    for c_val in (C_LO, 1.37, 1.5, 2.0):
        y = net.task_output(x, torch.full((n_samples,), c_val))
        task_err = float((y - x.clamp(-c_val, c_val)).abs().max())
        print(f"  c={c_val:<5g} task max err = {task_err:.2e}")
        assert task_err < 1e-9


def uniform(shape, lo, hi, gen) -> Tensor:
    return torch.rand(shape, generator=gen) * (hi - lo) + lo


# --------------------------------------------------------------------------
# 4. Linear probes (difference of means & logistic regression)
# --------------------------------------------------------------------------


def probe_layer(train, test) -> dict:
    """Fit both probes on one layer's residuals; score on the held-out split."""
    X_tr, y_tr = train  # X shape (batch, d_model); y shape (d_model)
    X_te, y_te = test
    out = {}

    # Compute difference-of-means
    mu0, mu1 = X_tr[y_tr == 0].mean(0), X_tr[y_tr == 1].mean(0)
    w = mu1 - mu0
    b = float(-0.5 * (mu0 + mu1) @ w)
    s = X_te @ w + b
    out["difference of means"] = dict(
        scores=s,
        auroc=float(roc_auc_score(y_te, s)) if np.ptp(s) > 0 else 0.5,
        acc=float(((s > 0) == y_te).mean()),
    )

    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    s = X_te @ clf.coef_[0] + float(clf.intercept_[0])
    out["logistic regression"] = dict(
        scores=s,
        auroc=float(roc_auc_score(y_te, s)),
        acc=float(clf.score(X_te, y_te)),
    )
    return out


def run_probes(net: ResidualMLP, sched: Schedule, args, gen) -> dict[int, dict]:
    """Probe every residual layer (both probed and unprobed, for contrast)."""

    def split():
        x = uniform((2 * args.n, sched.num_x), X_LO, X_HI, gen)
        c = torch.cat(
            [torch.full((args.n,), args.c_lo), torch.full((args.n,), args.c_hi)]
        )
        y = np.concatenate([np.zeros(args.n), np.ones(args.n)])
        return [r.numpy() for r in net.residuals(x, c)], y

    rs_tr, y_tr = split()
    rs_te, y_te = split()

    results = {}
    for layer in range(1, sched.num_blocks + 1):
        res = probe_layer((rs_tr[layer], y_tr), (rs_te[layer], y_te))
        # Largest per-dimension shift in conditional mean -- the quantity the
        # construction drives to zero at probed layers. Reported both raw and
        # as a z-score, since at a hidden layer the raw number is pure sampling
        # noise (~1/sqrt(n)) and only the z-score says whether it is real.
        lo, hi = rs_te[layer][y_te == 0], rs_te[layer][y_te == 1]
        delta = hi.mean(0) - lo.mean(0)
        se = np.sqrt(lo.var(0) / len(lo) + hi.var(0) / len(hi))
        # A zero-variance dim that still shifts (e.g. the c dim itself at layer
        # 1) is infinitely detectable, not undefined.
        z = np.where(se > 0, np.abs(delta) / np.where(se > 0, se, 1.0), np.inf)
        z[(se == 0) & (np.abs(delta) < 1e-12)] = 0.0
        res["mean_shift"] = float(np.abs(delta).max())
        res["mean_shift_z"] = float(z.max())
        res["probed"] = sched.is_probed(layer)
        results[layer] = res
    return results


# --------------------------------------------------------------------------
# 5. Plots
# --------------------------------------------------------------------------
PROBE_NAMES = ["difference of means", "logistic regression"]


def plot_probe_hists(sched, results, args, out_dir) -> pathlib.Path:
    """One row per probed layer; one histogram per probe type, with AUROC."""
    layers = [k for k, v in results.items() if v["probed"]]
    fig, axes = plt.subplots(
        len(layers),
        len(PROBE_NAMES),
        figsize=(5.2 * len(PROBE_NAMES), 2.5 * len(layers)),
        squeeze=False,
    )
    lo_label, hi_label = f"c={args.c_lo:g}", f"c={args.c_hi:g}"
    for row, layer in enumerate(layers):
        for col, name in enumerate(PROBE_NAMES):
            ax = axes[row][col]
            s = results[layer][name]["scores"]
            span = max(s.max() - s.min(), 1e-12)
            bins = np.linspace(s.min() - 0.02 * span, s.max() + 0.02 * span, 60)
            ax.hist(s[: args.n], bins=bins, alpha=0.55, label=lo_label)
            ax.hist(s[args.n :], bins=bins, alpha=0.55, label=hi_label)
            ax.axvline(0.0, color="k", ls="--", lw=1)
            ax.set_title(
                f"layer {layer} ({sched.n_finished(layer)}/{sched.num_x} coords done)"
                f" -- {name}\n"
                f"AUROC={results[layer][name]['auroc']:.4f}, "
                f"acc={results[layer][name]['acc']:.4f}",
                fontsize=9,
            )
            ax.set_xlabel("probe projection (threshold at 0)")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("count")
            if row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        f"Probes on every probed (even) residual layer, {lo_label} vs {hi_label}",
        y=1.0,
    )
    fig.tight_layout()
    path = out_dir / "simplified_probe_hist.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[plot] wrote {path}")
    return path


def plot_auroc_by_layer(results, out_dir) -> pathlib.Path:
    """AUROC vs layer, probed layers against the unprobed ones."""
    layers = sorted(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in zip(PROBE_NAMES, ("k", "r")):
        ax.plot(
            layers,
            [results[k][name]["auroc"] for k in layers],
            "o-",
            color=color,
            label=name,
            alpha=0.8,
        )
    for k in layers:
        if not results[k]["probed"]:
            ax.axvspan(k - 0.5, k + 0.5, color="0.85", zorder=0)
    ax.axhline(0.5, color="b", ls="--", lw=1, label="chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("residual layer (grey = unprobed layer)")
    ax.set_ylabel("AUROC")
    ax.set_title(
        "c-detectability per layer: hidden at probed layers, "
        "plainly visible at unprobed ones"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "simplified_probe_auroc.png"
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")
    return path


def plot_learned_curves(net, out_dir, c_values=(1.0, 1.333, 1.667, 2.0)):
    """y(x) per coordinate at fixed c, showing the network exactly reproduces
    sat(x, -c, c)."""
    xs = torch.linspace(X_LO, X_HI, 400)
    fig, axes = plt.subplots(
        1, len(c_values), figsize=(4 * len(c_values), 4), sharey=True
    )
    for ax, c in zip(np.atleast_1d(axes), c_values):
        for j in range(net.num_x):
            # Coordinates are independent, so sweep one and hold the rest at 0.
            x = torch.zeros(len(xs), net.num_x)
            x[:, j] = xs
            y = net.task_output(x, torch.full((len(xs),), c))[:, j]
            ax.plot(xs.numpy(), y.numpy(), alpha=0.5, zorder=5)
        ax.plot(
            xs.numpy(),
            xs.clamp(-c, c).numpy(),
            "k--",
            lw=1,
            label="target sat",
            zorder=2,
        )
        ax.set_title(f"c = {c:.3f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("y")
    fig.suptitle(
        f"analytic construction: y(x) per coordinate, fixed c; "
        f"{net.num_x} lines/panel (all coincide with the target)"
    )
    fig.tight_layout()
    path = out_dir / "simplified_curves.png"
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")
    return path


# --------------------------------------------------------------------------
@torch.no_grad()
def main(args):
    gen = torch.Generator().manual_seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== network exactness ==")
    net, sched = build_network(args.num_x)
    check_network(net, sched, gen)

    print(f"\n== probes (c={args.c_lo:g} vs c={args.c_hi:g}, n_test={2 * args.n}) ==")
    results = run_probes(net, sched, args, gen)
    print(
        f"{'layer':>6} {'probed':>7} {'max |dmean|':>12} {'as z':>9} "
        f"{'DoM AUROC':>10} {'logreg AUROC':>13}"
    )
    for layer in sorted(results):
        r = results[layer]
        print(
            f"{layer:>6} {str(r['probed']):>7} {r['mean_shift']:>12.2e} "
            f"{r['mean_shift_z']:>9.1f} "
            f"{r['difference of means']['auroc']:>10.4f} "
            f"{r['logistic regression']['auroc']:>13.4f}"
        )
    print(
        "  (probed layers: |dmean| is sampling noise -- z is O(1); the "
        "encoding is mean-constant in c exactly)"
    )

    print()
    plot_learned_curves(net, out_dir)
    plot_probe_hists(sched, results, args, out_dir)
    plot_auroc_by_layer(results, out_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main(args)
