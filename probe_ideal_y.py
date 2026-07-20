"""Sanity check: is c linearly decodable from the *ideal* task output alone?

No trained model involved -- this tests the data-generating process directly:
x ~ U[x_low, x_high]^num_x, y = clamp(x, -c, c) elementwise, c pinned at c_lo/c_hi
(mirrors train_probe.py's binary_dataset). Since x is symmetric about 0, y|c is
a zero-mean symmetric distribution for every c, so no single linear direction
(raw DoM or logistic regression) should separate c_lo from c_hi above chance --
the class-conditional means coincide. Contrast this with a trivial nonlinear
statistic, max_j|y_j|, which converges to c almost surely (c IS the clip
boundary), to confirm the gap is "linear vs nonlinear", not "no signal at all".

Usage:
    python probe_ideal_y.py --num-x 32 --c-lo 1.0 --c-hi 2.0
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-x", type=int, default=32)
    p.add_argument("--c-lo", type=float, default=1.0)
    p.add_argument("--c-hi", type=float, default=2.0)
    p.add_argument("--x-low", type=float, default=-3.0)
    p.add_argument("--x-high", type=float, default=3.0)
    p.add_argument("--n-train", type=int, default=20_000, help="per class")
    p.add_argument("--n-test", type=int, default=50_000, help="per class")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sample_y(n, num_x, c, x_low, x_high, rng):
    x = rng.uniform(x_low, x_high, size=(n, num_x))
    return np.clip(x, -c, c)


def main(args):
    rng = np.random.default_rng(args.seed)

    y_lo_tr = sample_y(
        args.n_train, args.num_x, args.c_lo, args.x_low, args.x_high, rng
    )
    y_hi_tr = sample_y(
        args.n_train, args.num_x, args.c_hi, args.x_low, args.x_high, rng
    )
    y_lo_te = sample_y(args.n_test, args.num_x, args.c_lo, args.x_low, args.x_high, rng)
    y_hi_te = sample_y(args.n_test, args.num_x, args.c_hi, args.x_low, args.x_high, rng)

    X_train = np.concatenate([y_lo_tr, y_hi_tr], axis=0)
    y_train = np.concatenate([np.zeros(args.n_train), np.ones(args.n_train)])
    X_test = np.concatenate([y_lo_te, y_hi_te], axis=0)
    y_test = np.concatenate([np.zeros(args.n_test), np.ones(args.n_test)])

    # --- raw difference-of-means ---
    mu_lo = y_lo_tr.mean(axis=0)
    mu_hi = y_hi_tr.mean(axis=0)
    w_dom = mu_hi - mu_lo
    midpoint = ((mu_hi + mu_lo) / 2) @ w_dom
    pred_dom = (X_test @ w_dom > midpoint).astype(float)
    dom_acc = float((pred_dom == y_test).mean())

    # --- logistic regression (linear, standardized) ---
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logreg.fit(X_train, y_train)
    logreg_acc = float(logreg.score(X_test, y_test))

    # --- nonlinear contrast: max|y_j| as a trivial c-estimator ---
    max_abs_lo = np.abs(y_lo_te).max(axis=1)
    max_abs_hi = np.abs(y_hi_te).max(axis=1)
    threshold = (args.c_lo + args.c_hi) / 2
    nonlinear_pred = (np.concatenate([max_abs_lo, max_abs_hi]) > threshold).astype(
        float
    )
    nonlinear_acc = float((nonlinear_pred == y_test).mean())

    print(
        f"ideal-y probe: num_x={args.num_x} x=[{args.x_low},{args.x_high}] "
        f"c_lo={args.c_lo} c_hi={args.c_hi} n_train={args.n_train}/class "
        f"n_test={args.n_test}/class"
    )
    print(
        f"  raw DoM              acc = {dom_acc:.4f}  (expect ~0.5: y|c is zero-mean, symmetric)"
    )
    print(f"  logreg (linear)      acc = {logreg_acc:.4f}  (expect ~0.5, same reason)")
    print(
        f"  max|y_j| (nonlinear) acc = {nonlinear_acc:.4f}  (expect ~1.0: c is the clip boundary)"
    )


if __name__ == "__main__":
    main(args)
