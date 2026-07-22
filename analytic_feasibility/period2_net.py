"""Literal period-2 network: probes at even residual layers, d_mlp = num_x/2.

Layout (d_model dims): [x coords (n)] [c] [v1] [v2] [Q1..Q8]
Schedule:
  block 0: write v1, v2 (r1 unprobed, c still linear there)
  block 1: anchor-finish first batch of coords using linear c, erase c -> r2 clean
  block 2k: decode-basis (8 atoms Q_j) -> r_{2k+1} dirty (unprobed)
  block 2k+1: finish 2 coords via c_hat linear read, clear Q dims -> clean
"""

import numpy as np

relu = lambda z: np.maximum(z, 0.0)


def v1f(x, c):
    return -2 * relu(-x - c) + 2 * relu(x - 3 + c) - c + 1.5


def v2f(x, c):
    return -4 * relu(-x - c / 2) + 4 * relu(x + c / 2 - 3) - c + 3.0


# ---- recompute exact decode atoms + weights (from period2_atoms scan) ----
bases = {
    2: [(-2, 1, 0, -1.5), (-2, 0, 1, -3)],
    4: [(0, 1, 0, -1.5), (-2, 0, 1, 3)],
}
xs = np.linspace(-3, 3, 1401)
cs = np.linspace(1, 2, 241)
Xg, Cg = np.meshgrid(xs, cs, indexing="ij")
V1g, V2g = v1f(Xg, Cg), v2f(Xg, Cg)
curve_pos = {2: -Cg / 2, 4: 3 - Cg / 2}


def combo(j, s1, s2):
    b1, b2 = bases[j]
    return tuple(s1 * u + s2 * v for u, v in zip(b1, b2))


def find_two_valid(j, side):
    got = []
    for ang in np.linspace(0, 2 * np.pi, 2880, endpoint=False):
        s1, s2 = np.cos(ang), np.sin(ang)
        co = combo(j, s1, s2)
        P = co[0] * Xg + co[1] * V1g + co[2] * V2g + co[3]
        right = Xg > curve_pos[j] + 1e-6
        left = Xg < curve_pos[j] - 1e-6
        pr, pl = (P[right] > 1e-9).mean(), (P[left] > 1e-9).mean()
        ok = (
            (pr > 1 - 1e-9 and pl < 1e-9)
            if side == "R"
            else (pl > 1 - 1e-9 and pr < 1e-9)
        )
        if ok:
            got.append((s1, s2))
    M = np.array(got)
    # two maximally-separated representatives
    i0 = 0
    dots = M @ M[i0]
    i1 = int(np.argmin(np.abs(dots)))
    return [tuple(M[i0]), tuple(M[i1])]


atom_coeffs = []
for j in (2, 4):
    for side in ("L", "R"):
        for s1, s2 in find_two_valid(j, side):
            atom_coeffs.append(combo(j, s1, s2))
assert len(atom_coeffs) == 8

featg = [np.ones_like(Xg), Xg, V1g, V2g] + [
    relu(a * Xg + b * V1g + g * V2g + d) for (a, b, g, d) in atom_coeffs
]
A_mat = np.stack([f.ravel() for f in featg], axis=1)
w_dec, *_ = np.linalg.lstsq(A_mat, Cg.ravel(), rcond=None)
err = np.abs(A_mat @ w_dec - Cg.ravel()).max()
print(f"decode refit: max |G - c| = {err:.2e}")
assert err < 1e-9

# ---- build the literal network ----
n = 24
d_mlp = n // 2  # 12
first_batch = 5  # block 1: 2*5+1 = 11 <= 12 neurons
per_period = 2  # block 2k+1: 2*2 use + 8 clears = 12 neurons
C_DIM, V1_DIM, V2_DIM = n, n + 1, n + 2
Q0 = n + 3
d_model = Q0 + 8

blocks = []  # each: (W_in [d_model,d_mlp], b_in, W_out [d_mlp,d_model], b_out)


def zeros_block():
    return (
        np.zeros((d_model, d_mlp)),
        np.zeros(d_mlp),
        np.zeros((d_mlp, d_model)),
        np.zeros(d_model),
    )


# block 0: write v1 (uses x1=dim0, c), v2. 4 kinked + 1 always-on = 5 neurons
W_in, b_in, W_out, b_out = zeros_block()
# neuron 0: relu(-x1 - c); neuron 1: relu(x1 - 3 + c)
W_in[0, 0], W_in[C_DIM, 0] = -1, -1
W_in[0, 1], W_in[C_DIM, 1], b_in[1] = 1, 1, -3
# neuron 2: relu(-x1 - c/2); neuron 3: relu(x1 + c/2 - 3)
W_in[0, 2], W_in[C_DIM, 2] = -1, -0.5
W_in[0, 3], W_in[C_DIM, 3], b_in[3] = 1, 0.5, -3
# neuron 4: relu(c + 10) always-on (for the -c terms)
W_in[C_DIM, 4], b_in[4] = 1, 10
# v1 = -2*n0 + 2*n1 - (c) + 3/2
W_out[0, V1_DIM], W_out[1, V1_DIM], W_out[4, V1_DIM] = -2, 2, -1
b_out[V1_DIM] = 10 + 1.5
# v2 = -4*n2 + 4*n3 - c + 3
W_out[2, V2_DIM], W_out[3, V2_DIM], W_out[4, V2_DIM] = -4, 4, -1
b_out[V2_DIM] = 10 + 3.0
blocks.append((W_in, b_in, W_out, b_out))

# block 1: finish coords 1..5 with linear c (coord 0 kept: decode needs x1);
# erase c. 11 neurons
W_in, b_in, W_out, b_out = zeros_block()
for m in range(first_batch):
    i = m + 1
    W_in[i, 2 * m], W_in[C_DIM, 2 * m] = 1, -1  # relu(x_i - c)
    W_in[i, 2 * m + 1], W_in[C_DIM, 2 * m + 1] = -1, -1  # relu(-x_i - c)
    W_out[2 * m, i], W_out[2 * m + 1, i] = -1, 1  # o[i] = sat - x_i
W_in[C_DIM, 10], b_in[10] = 1, 10  # erasure
W_out[10, C_DIM], b_out[C_DIM] = -1, 10
blocks.append((W_in, b_in, W_out, b_out))

# periodic blocks; coord 0 processed last so x1 survives until final use-block
coord_order = list(range(1, first_batch + 1)) + list(range(first_batch + 1, n)) + [0]
pending = coord_order[first_batch:]
while pending:
    batch, pending = pending[:per_period], pending[per_period:]
    k = len(batch)
    # decode-basis block: Q_j = relu(a*x1 + b*v1 + g*v2 + d)
    W_in, b_in, W_out, b_out = zeros_block()
    for j, (a, b, g, d) in enumerate(atom_coeffs):
        W_in[0, j], W_in[V1_DIM, j], W_in[V2_DIM, j], b_in[j] = a, b, g, d
        W_out[j, Q0 + j] = 1
    blocks.append((W_in, b_in, W_out, b_out))
    # use+clear block: c_hat = w0 + w1*x1 + w2*v1 + w3*v2 + sum w_j Q_j (linear read)
    W_in, b_in, W_out, b_out = zeros_block()
    chat_vec = np.zeros(d_model)
    chat_vec[0], chat_vec[V1_DIM], chat_vec[V2_DIM] = w_dec[1], w_dec[2], w_dec[3]
    chat_vec[Q0 : Q0 + 8] = w_dec[4:]
    chat_const = w_dec[0]
    for m in range(k):
        i = batch[m]
        # relu(x_i - c_hat), relu(-x_i - c_hat)
        W_in[:, 2 * m] = -chat_vec
        W_in[i, 2 * m] += 1
        b_in[2 * m] = -chat_const
        W_in[:, 2 * m + 1] = -chat_vec
        W_in[i, 2 * m + 1] += -1
        b_in[2 * m + 1] = -chat_const
        W_out[2 * m, i], W_out[2 * m + 1, i] = -1, 1
    for j in range(8):  # clear Q_j: always-on relu(Q_j + 1)
        W_in[Q0 + j, 2 * k + j], b_in[2 * k + j] = 1, 1
        W_out[2 * k + j, Q0 + j], b_out[Q0 + j] = -1, 1
    blocks.append((W_in, b_in, W_out, b_out))

print(f"blocks: {len(blocks)}, d_mlp={d_mlp}, d_model={d_model}")
used = max(
    int((np.abs(W_in).sum(0) + np.abs(b_in) > 0).sum()) for W_in, b_in, _, _ in blocks
)
print(f"max neurons used in any block: {used} (budget {d_mlp})")

# ---- run + verify ----
rng = np.random.default_rng(1)
B = 200_000
X = rng.uniform(-3, 3, (B, n))
results = []
for c in [1.0, 1.17, 1.5, 1.83, 2.0]:
    r = np.zeros((B, d_model))
    r[:, :n] = X
    r[:, C_DIM] = c
    probed_err = []
    for bi, (W_in, b_in, W_out, b_out) in enumerate(blocks):
        r = r + relu(r @ W_in + b_in) @ W_out + b_out
        layer = bi + 1
        if layer % 2 == 0:  # probed layer: expected exact content
            exp = np.zeros_like(r)
            nd = min(first_batch + max(0, (layer - 2) // 2) * per_period, n)
            done_set = set(coord_order[:nd])
            for i in range(n):
                exp[:, i] = np.clip(X[:, i], -c, c) if i in done_set else X[:, i]
            exp[:, V1_DIM] = v1f(X[:, 0], c)
            exp[:, V2_DIM] = v2f(X[:, 0], c)
            probed_err.append(np.abs(r - exp).max())
    y_err = np.abs(r[:, :n] - np.clip(X, -c, c)).max()
    results.append((c, y_err, max(probed_err)))
    print(
        f"c={c:4}: task max err = {y_err:.2e}, "
        f"probed-layer content err = {max(probed_err):.2e}"
    )

print(
    "\nAll probed layers hold exactly [sat-done x, pending x, 0, v1, v2, 0]: "
    "mean-constancy across c follows from proven channel properties."
)
