"""
config.py – Central hyper-parameters and output paths.
"""
from pathlib import Path

# ── learning hyper-parameters ─────────────────────────────────────────────────
N_STEPS   = 60_000   # training steps per trajectory
ALPHA     = 0.01     # Q-learning rate
TAU       = 0.2      # Boltzmann temperature  (lower → sharper, closer to NE)
EMA_DECAY = 0.9995  # EMA of instantaneous policy: time-constant ≈ 2000 steps
KAPPA     = 0.5      # leniency decay rate
EPSILON   = 0.1      # ε-greedy exploration probability
SEED      = 42

# ── plotting ──────────────────────────────────────────────────────────────────
N_GRID    = 20       # quiver resolution
SUBSAMPLE = 100      # keep every N-th trajectory point
DPI       = 150

# Default starting policies for 2-action games (x0, y0) = P(action-0)
STARTS_2ACTION = [
    (0.05, 0.05), (0.05, 0.50), (0.05, 0.95),
    (0.50, 0.05), (0.95, 0.05),
    (0.50, 0.95), (0.95, 0.50), (0.95, 0.95),
]

# Subsidy Game: cover BOTH basins of the separatrix x*=y*=10/11≈0.909
# so we see trajectories heading to (S1,S1) AND to (S2,S2)
STARTS_SUBSIDY = [
    # ── above separatrix → converge to (S1,S1) = (1,1) ──────────────
    (0.97, 0.97),
    (0.94, 0.94),
    (0.92, 0.92),
    (0.95, 0.87),   # off-diagonal, both above / one near sep
    # ── below separatrix → converge to (S2,S2) = (0,0) ──────────────
    (0.88, 0.88),   # just below
    (0.70, 0.70),
    (0.50, 0.50),
    (0.30, 0.30),
    (0.87, 0.95),   # off-diagonal
]

# Starting policies for 3-action games (simplex vertices + interior)
STARTS_3ACTION = [
    (0.90, 0.05, 0.05),
    (0.05, 0.90, 0.05),
    (0.05, 0.05, 0.90),
    (0.33, 0.33, 0.34),
    (0.60, 0.30, 0.10),
    (0.10, 0.60, 0.30),
    (0.30, 0.10, 0.60),
]

# ── output directories ────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(__file__).parent / "output"
OUT_2_2     = OUTPUT_ROOT / "task2_2"
OUT_2_3     = OUTPUT_ROOT / "task2_3"

for _p in (OUT_2_2, OUT_2_3):
    _p.mkdir(parents=True, exist_ok=True)
