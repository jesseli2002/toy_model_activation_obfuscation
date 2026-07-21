"""Period-2 question: is c representable as ONE hidden relu layer over
(x1, v1, v2) on the encoding manifold?

Structure: G = A(x1,v1,v2) + sum_j [one-sided affine additions along the 4
v-kink curves], where each addition comes from the 2-dim family of affines
vanishing on that curve. Solve the cell-wise linear system G === c, then try
to realize additions as relu atoms (sign check), then verify numerically.
"""

import numpy as np
import sympy as sp

x1, c = sp.symbols("x1 c")

# v branches per cell (cells B0..B4 between curves C1..C4)
v1_br = [
    2 * x1 + c + sp.Rational(3, 2),
    sp.Rational(3, 2) - c,
    sp.Rational(3, 2) - c,
    2 * x1 + c - sp.Rational(9, 2),
    2 * x1 + c - sp.Rational(9, 2),
]
v2_br = [
    4 * x1 + c + 3,
    4 * x1 + c + 3,
    3 - c,
    3 - c,
    4 * x1 + c - 9,
]


# Curve-vanishing affine basis: P = al*x1 + be*v1 + ga*v2 + de with P==0 on curve.
# Constraints derived by substituting curve + on-curve v values.
def curve_basis(curve_idx):
    al, be, ga, de = sp.symbols(
        f"al{curve_idx} be{curve_idx} ga{curve_idx} de{curve_idx}"
    )
    # on-curve substitutions
    curves = {
        1: (-c, sp.Rational(3, 2) - c, 3 - 3 * c),
        2: (-c / 2, sp.Rational(3, 2) - c, 3 - c),
        3: (3 - c, sp.Rational(3, 2) - c, 3 - c),
        4: (3 - c / 2, sp.Rational(3, 2), 3 - c),
    }
    xc, v1c, v2c = curves[curve_idx]
    P_on = sp.expand(al * xc + be * v1c + ga * v2c + de)
    eqs = [sp.Eq(P_on.coeff(c, 1), 0), sp.Eq(P_on.coeff(c, 0), 0)]
    sol = sp.solve(eqs, [al, de], dict=True)[0]
    # free params: be, ga -> two basis vectors (al, be, ga, de)
    basis = []
    for freevals in [(1, 0), (0, 1)]:
        sub = {be: freevals[0], ga: freevals[1]}
        vec = (
            sp.simplify(sol[al].subs(sub)),
            freevals[0],
            freevals[1],
            sp.simplify(sol[de].subs(sub)),
        )
        basis.append(vec)
    return basis  # coefficients (alpha, beta, gamma, delta)


bases = {j: curve_basis(j) for j in (1, 2, 3, 4)}
print("curve-vanishing bases (alpha,beta,gamma,delta):")
for j, b in bases.items():
    print(f"  C{j}: {b[0]}  |  {b[1]}")

# Unknowns: affine A = a*x1 + b*v1 + g*v2 + k ; per curve j, per side s in
# {L, R}: params (s1, s2) combining the 2 basis affines.
a, b, g, k = sp.symbols("a b g k")
params = {}
for j in (1, 2, 3, 4):
    for side in ("L", "R"):
        params[(j, side)] = sp.symbols(f"p{j}{side}1 p{j}{side}2")


def affine_on_cell(coeffs, cell):
    al, be, ga, de = coeffs
    return sp.expand(al * x1 + be * v1_br[cell] + ga * v2_br[cell] + de)


def addition_on_cell(j, side, cell):
    s1, s2 = params[(j, side)]
    b1, b2 = bases[j]
    return s1 * affine_on_cell(b1, cell) + s2 * affine_on_cell(b2, cell)


eqs = []
for cell in range(5):
    G = a * x1 + b * v1_br[cell] + g * v2_br[cell] + k
    for j in (1, 2, 3, 4):
        # cell m is right of curves 1..m, left of curves m+1..4
        if j <= cell:
            G += addition_on_cell(j, "R", cell)
        else:
            G += addition_on_cell(j, "L", cell)
    diff = sp.expand(G - c)
    poly = sp.Poly(diff, x1, c)
    for coeff in poly.coeffs():
        eqs.append(sp.Eq(coeff, 0))

unknowns = [a, b, g, k] + [p for pr in params.values() for p in pr]
sol = sp.solve(eqs, unknowns, dict=True)
print(f"\nlinear system: {len(eqs)} equations, {len(unknowns)} unknowns")
if not sol:
    print("INFEASIBLE: no one-layer exact decode with curve-aligned atoms")
else:
    s = sol[0]
    free = [u for u in unknowns if u not in s]
    print("FEASIBLE. free params:", free)
    print({str(kk): sp.simplify(vv) for kk, vv in s.items()})
