"""
dynamics.py – Replicator dynamics and Q-value initialisation helpers.

Public API
----------
  replicator_field_2action(A, B)     -> X, Y, U, V  (2-D quiver grid)
  replicator_field_simplex(A)        -> Xc, Yc, Uc, Vc  (simplex quiver)
  init_q_boltzmann(pi, tau)          -> Q  (1-D array)
  print_replicator_equations(A, B, name)  (Task 2.3 analytical component)
"""
from __future__ import annotations
import numpy as np
from config import N_GRID

# Vertices of the equilateral triangle for the 2-simplex
_V = np.array([[0.0, 0.0],
               [1.0, 0.0],
               [0.5, np.sqrt(3) / 2]])


# ─────────────────────────────────────────────────────────────────────────────
# Q initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_q_boltzmann(pi, tau: float) -> np.ndarray:
    """
    Return Q values whose Boltzmann policy matches *pi*.
    Uses Q_i = τ · (log π_i − mean(log π)), zero-centred.
    """
    pi = np.asarray(pi, dtype=float)
    pi = np.clip(pi / pi.sum(), 1e-8, None)
    log_pi = np.log(pi)
    return tau * (log_pi - log_pi.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 2-action replicator field
# ─────────────────────────────────────────────────────────────────────────────

def replicator_field_2action(A, B, n: int = N_GRID):
    """
    Replicator dynamics for a 2×2 game.

    State:  (x, y)  where x = P(player-1 plays action-0)
                          y = P(player-2 plays action-0)

    Equations:
        ẋ = x(1−x) · [(Ay)_0 − (Ay)_1]
        ẏ = y(1−y) · [(B^T x)_0 − (B^T x)_1]

    Returns X, Y, U, V  (meshgrid + velocity components).
    """
    xs = np.linspace(0.03, 0.97, n)
    X, Y = np.meshgrid(xs, xs)
    U, V = np.zeros_like(X), np.zeros_like(Y)

    for i in range(n):
        for j in range(n):
            x, y = X[i, j], Y[i, j]
            xv = np.array([x, 1 - x])
            yv = np.array([y, 1 - y])

            f = A @ yv
            U[i, j] = x * (f[0] - xv @ f)

            g = B.T @ xv
            V[i, j] = y * (g[0] - yv @ g)

    return X, Y, U, V


# ─────────────────────────────────────────────────────────────────────────────
# 3-action simplex replicator field (self-play)
# ─────────────────────────────────────────────────────────────────────────────

def bary_to_cart(p: np.ndarray) -> np.ndarray:
    """Barycentric → Cartesian. p shape (..., 3)."""
    return p @ _V


def _cart_vel(dp: np.ndarray) -> np.ndarray:
    """
    Convert barycentric velocity dp=(dp0,dp1,dp2) to Cartesian.
    Jacobian:  J[:,k] = V[k] − V[2]  for k=0,1
    """
    J = np.stack([_V[0] - _V[2], _V[1] - _V[2]], axis=1)  # (2,2)
    return J @ dp[:2]


def replicator_field_simplex(A, n: int = N_GRID):
    """
    Replicator dynamics on the 2-simplex (3-action self-play game).

    ẋ_i = x_i · [(Ax)_i − x·Ax]

    Returns Cartesian quiver data: Xc, Yc, Uc, Vc.
    """
    pts_bary = np.array([
        [i / n, j / n, (n - i - j) / n]
        for i in range(1, n)
        for j in range(1, n - i)
    ])
    pts_cart = bary_to_cart(pts_bary)
    Uc = np.zeros(len(pts_bary))
    Vc = np.zeros(len(pts_bary))

    for k, p in enumerate(pts_bary):
        f   = A @ p
        dp  = p * (f - p @ f)
        uv  = _cart_vel(dp)
        Uc[k], Vc[k] = uv

    return pts_cart[:, 0], pts_cart[:, 1], Uc, Vc


# ─────────────────────────────────────────────────────────────────────────────
# Task 2.3 – print analytical replicator equations for 2×2 games
# ─────────────────────────────────────────────────────────────────────────────

def print_replicator_equations(A, B, name: str) -> None:
    """
    Derive and print the replicator equations for a 2×2 game.

    x = P(row plays action 0),  y = P(col plays action 0)

        ẋ = x(1−x) · [(A_00−A_10)·y + (A_01−A_11)·(1−y)]
           = x(1−x) · [c_y · y + c_0]

        ẏ = y(1−y) · [(B_00−B_01)·x + (B_10−B_11)·(1−x)]
           = y(1−y) · [d_x · x + d_0]
    """
    if A.shape != (2, 2):
        print(f"\n[{name}]  Analytical equations available for 2×2 games only.")
        return

    # Coefficients after expanding
    cy = A[0, 0] - A[1, 0]       # slope in y for ẋ
    c0 = A[0, 1] - A[1, 1]       # constant term
    cy_eff = cy - c0              # net coefficient of y after expansion

    dx = B[0, 0] - B[0, 1]       # slope in x for ẏ
    d0 = B[1, 0] - B[1, 1]
    dx_eff = dx - d0

    print(f"\n{'─'*65}")
    print(f"  Replicator Dynamics  –  {name}")
    print(f"{'─'*65}")
    print(f"  A =\n{A}\n  B =\n{B}\n")
    print(f"  ẋ = x(1−x)·[{cy_eff:+.4f}·y {c0:+.4f}]")
    if abs(cy_eff) > 1e-9:
        y_fp = -c0 / cy_eff
        tag  = f"  ← interior fixed point y* = {y_fp:.4f}" if 0 < y_fp < 1 else ""
        print(f"       ẋ = 0  at  y = {y_fp:.4f}{tag}")
    else:
        print(f"       ẋ has constant sign (no interior fixed point in y)")

    print(f"\n  ẏ = y(1−y)·[{dx_eff:+.4f}·x {d0:+.4f}]")
    if abs(dx_eff) > 1e-9:
        x_fp = -d0 / dx_eff
        tag  = f"  ← interior fixed point x* = {x_fp:.4f}" if 0 < x_fp < 1 else ""
        print(f"       ẏ = 0  at  x = {x_fp:.4f}{tag}")
    else:
        print(f"       ẏ has constant sign (no interior fixed point in x)")


if __name__ == "__main__":
    from games import ALL_GAMES
    for g in ALL_GAMES:
        print_replicator_equations(g.A, g.B, g.name)
