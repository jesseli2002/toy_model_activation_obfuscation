"""Numeric checks for feasibility-analysis claims."""

import numpy as np

rng = np.random.default_rng(0)
relu = lambda z: np.maximum(z, 0.0)

N = 2_000_000
x = rng.uniform(-3, 3, N)
x2 = rng.uniform(-3, 3, N)
cs = np.linspace(1, 2, 21)

print("== Claim 1: v1, v2 mean-constant across c ==")
for c in [1.0, 1.25, 1.5, 1.75, 2.0]:
    v1 = -2 * relu(-x - c) + 2 * relu(x - 3 + c) - c + 1.5
    v2 = -4 * relu(-x - c / 2) + 4 * relu(x + c / 2 - 3) - c + 3.0
    print(f"c={c:4}: E[v1]={v1.mean():+.4f}  E[v2]={v2.mean():+.4f}")

print("\n== Claim 2: exact c decode from (x, v1, v2), all regions ==")
xg = np.linspace(-3, 3, 1201)
worst = 0.0
for c in np.linspace(1, 2, 101):
    v1 = -2 * relu(-xg - c) + 2 * relu(xg - 3 + c) - c + 1.5
    v2 = -4 * relu(-xg - c / 2) + 4 * relu(xg + c / 2 - 3) - c + 3.0
    dec = np.select(
        [xg <= -2, xg <= -1, xg <= 1, xg <= 2],
        [v1 - 2 * xg - 1.5, v2 - 4 * xg - 3, 1.5 - v1, 3 - v2],
        default=v1 - 2 * xg + 4.5,
    )
    worst = max(worst, np.abs(dec - c).max())
print(f"max |decoded c - c| over grid: {worst:.2e}")

print("\n== Claim 3: t = x - sat(x,c) regenerates both relu kinks ==")
worst_a = worst_b = 0.0
for c in np.linspace(1, 2, 51):
    sat = np.clip(xg, -c, c)
    t = xg - sat
    worst_a = max(worst_a, np.abs(relu(t) - relu(xg - c)).max())
    worst_b = max(worst_b, np.abs(relu(-t) - relu(-xg - c)).max())
print(
    f"max |relu(t)-relu(x-c)| = {worst_a:.2e}, max |relu(-t)-relu(-x-c)| = {worst_b:.2e}"
)
print(
    "E[t] across c:",
    [f"{(x - np.clip(x, -c, c)).mean():+.4f}" for c in [1.0, 1.5, 2.0]],
)

print(
    "\n== Claim 4: shared-compensator phi_i = relu(x_i-c) - relu(x_1-c) mean-const =="
)
for c in [1.0, 1.5, 2.0]:
    phi = relu(x - c) - relu(x2 - c)
    print(f"c={c}: E[phi]={phi.mean():+.5f}")

print("\n== Claim 5: 2-block exact partial solution, all probed means const ==")
# n coords; J = (n-1)//2 encoded. Block 0: erase c, write t_j to spares.
# Block 1: o[j] = -t_j via always-on neuron. Simulate residual means.
n_x, J = 9, 4
M = 400_000
for c in [1.0, 1.5, 2.0]:
    X = rng.uniform(-3, 3, (M, n_x))
    # r1: coords x (untouched), c-dim erased to 0, spares t_j for j<J
    T = X[:, :J] - np.clip(X[:, :J], -c, c)
    r1_means = np.concatenate([X.mean(0), [0.0], T.mean(0)])
    # r2: encoded coords j -> sat(x_j,c); spares still t_j
    r2_coords = np.clip(X[:, :J], -c, c)
    r2_means = np.concatenate([r2_coords.mean(0), X[:, J:].mean(0), [0.0], T.mean(0)])
    print(
        f"c={c}: max|E r1|={np.abs(r1_means).max():.4f}  max|E r2|={np.abs(r2_means).max():.4f}"
    )

print("\n== Claim 6 (no-go check): single-side relu(x_i - c) alone leaks ==")
for c in [1.0, 1.5, 2.0]:
    print(
        f"c={c}: E[relu(x-c)]={relu(x - c).mean():+.4f}  (analytic (3-c)^2/12={((3-c)**2)/12:.4f})"
    )
