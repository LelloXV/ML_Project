"""
games.py – Payoff matrices and game-theoretic analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Game:
    name:          str
    A:             np.ndarray
    B:             np.ndarray
    actions:       List[str]
    slug:          str
    custom_starts: Optional[List] = field(default=None, repr=False)
    custom_alpha:  Optional[float] = field(default=None, repr=False)  # slow-down for fast-converging games

    @property
    def n_actions(self) -> int:
        return self.A.shape[0]

    def nash_pure(self) -> List[Tuple[int, int]]:
        n = self.n_actions
        return [
            (i, j)
            for i in range(n) for j in range(n)
            if self.A[i, j] >= self.A[:, j].max() - 1e-9
            and self.B[i, j] >= self.B[i, :].max() - 1e-9
        ]

    def pareto_optimal(self) -> List[Tuple[int, int]]:
        n = self.n_actions
        result = []
        for i in range(n):
            for j in range(n):
                r1, r2 = self.A[i, j], self.B[i, j]
                if not any(
                    self.A[i2, j2] >= r1 and self.B[i2, j2] >= r2
                    and (self.A[i2, j2] > r1 or self.B[i2, j2] > r2)
                    for i2 in range(n) for j2 in range(n)
                ):
                    result.append((i, j))
        return result

    def mixed_nash_2x2(self):
        if self.n_actions != 2:
            return None
        A, B = self.A, self.B
        dB = (B[0, 0] - B[1, 0]) - (B[0, 1] - B[1, 1])
        dA = (A[0, 0] - A[0, 1]) - (A[1, 0] - A[1, 1])
        if abs(dB) < 1e-10 or abs(dA) < 1e-10:
            return None
        p = (B[1, 1] - B[1, 0]) / dB
        q = (A[1, 1] - A[0, 1]) / dA
        if 0 < p < 1 and 0 < q < 1:
            return np.array([p, 1 - p]), np.array([q, 1 - q])
        return None

    def summary(self) -> None:
        print(f"\n{'─'*60}\n  {self.name}\n{'─'*60}")
        print(f"  A:\n{self.A}\n  B:\n{self.B}")
        ne = self.nash_pure()
        if ne:
            for i, j in ne:
                print(f"  NE:     ({self.actions[i]}, {self.actions[j]})"
                      f"  payoffs=({self.A[i,j]:.3f}, {self.B[i,j]:.3f})")
        else:
            print("  No pure-strategy NE")
        mne = self.mixed_nash_2x2()
        if mne is not None:
            print(f"  Mixed NE: p*={mne[0][0]:.3f}, q*={mne[1][0]:.3f}")
        for i, j in self.pareto_optimal():
            print(f"  Pareto: ({self.actions[i]}, {self.actions[j]})"
                  f"  payoffs=({self.A[i,j]:.3f}, {self.B[i,j]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Game instances
# ─────────────────────────────────────────────────────────────────────────────

from config import STARTS_SUBSIDY

PRISONERS_DILEMMA = Game(
    name    = "Prisoner's Dilemma",
    A       = np.array([[-1., -4.], [ 0., -3.]]),
    B       = np.array([[-1.,  0.], [-4., -3.]]),
    actions = ["C", "D"],
    slug    = "prisoners_dilemma",
)

STAG_HUNT = Game(
    name    = "Stag Hunt",
    A       = np.array([[1., 0.], [2/3, 2/3]]),
    B       = np.array([[1., 2/3], [0., 2/3]]),
    actions = ["S", "H"],
    slug    = "stag_hunt",
)

SUBSIDY_GAME = Game(
    name          = "Subsidy Game",
    A             = np.array([[12., 0.], [11., 10.]]),
    B             = np.array([[12., 11.], [0., 10.]]),
    actions       = ["S1", "S2"],
    slug          = "subsidy_game",
    custom_starts = STARTS_SUBSIDY,
    custom_alpha  = 0.002,   # payoffs 10-12 cause very fast convergence; slow down to show trajectories
)

BIASED_RPS = Game(
    name    = "Biased Rock-Paper-Scissors",
    A       = np.array([[ 0.00, -0.05,  0.25],
                        [ 0.05,  0.00, -0.50],
                        [-0.25,  0.50,  0.00]]),
    B       = np.array([[ 0.00,  0.05, -0.25],
                        [-0.05,  0.00,  0.50],
                        [ 0.25, -0.50,  0.00]]),
    actions = ["R", "P", "S"],
    slug    = "biased_rps",
)

ALL_GAMES = [PRISONERS_DILEMMA, STAG_HUNT, SUBSIDY_GAME, BIASED_RPS]


if __name__ == "__main__":
    for g in ALL_GAMES:
        g.summary()
