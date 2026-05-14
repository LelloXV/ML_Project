"""
plotting.py – Visualisation with legends and NE annotations.

Every panel includes:
  • Replicator-dynamics vector field (gray arrows)
  • Learning trajectories with start ● and end ■ markers
  • Pure Nash equilibria: large gold ★ + labelled callout
  • Mixed Nash equilibrium: orange ◆ + labelled callout
  • A legend explaining every element
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from config  import N_STEPS, ALPHA, TAU, DPI, STARTS_2ACTION, STARTS_3ACTION
from dynamics import (
    replicator_field_2action, replicator_field_simplex,
    init_q_boltzmann, bary_to_cart, _V,
)
from games import Game


# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────

_QUIVER   = dict(color="gray", alpha=0.28, scale=25,
                 headwidth=3, headlength=3, width=0.003)
_NE_STAR  = dict(marker="*",  color="#FFD700", ms=20, zorder=12,
                 markeredgecolor="#333", markeredgewidth=0.8)
_MIXED_NE = dict(marker="D",  color="#FF8C00", ms=10, zorder=12,
                 markeredgecolor="#333", markeredgewidth=0.8)
_START    = dict(marker="o",  ms=5, zorder=6, markeredgewidth=0.4)
_END      = dict(marker="s",  ms=5, zorder=6, markeredgewidth=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Legend handles
# ─────────────────────────────────────────────────────────────────────────────

def _lh_line(col, label):
    return Line2D([0],[0], color=col, lw=1.5, label=label)

def _lh_start(col):
    return Line2D([0],[0], ls="none", marker="o", color=col,
                  ms=6, markeredgewidth=0.4, label="Start ●")

def _lh_end(col):
    return Line2D([0],[0], ls="none", marker="s", color=col,
                  ms=6, markeredgewidth=0.4, label="End ■")

def _lh_ne():
    return Line2D([0],[0], ls="none", marker="*", color="#FFD700",
                  ms=14, markeredgecolor="#333", markeredgewidth=0.6,
                  label="Nash Eq. ★")

def _lh_mne():
    return Line2D([0],[0], ls="none", marker="D", color="#FF8C00",
                  ms=8, markeredgecolor="#333", markeredgewidth=0.6,
                  label="Mixed Nash Eq. ◆")

def _lh_field():
    return Line2D([0],[0], color="gray", lw=1.5, alpha=0.5,
                  label="Replicator field →")


# ─────────────────────────────────────────────────────────────────────────────
# NE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ne_coords_2action(game: Game):
    coords = []
    for i, j in game.nash_pure():
        x, y = 1.0 - i, 1.0 - j
        coords.append((x, y, f"NE ({game.actions[i]},{game.actions[j]})", "pure"))
    mne = game.mixed_nash_2x2()
    if mne is not None:
        p, q = mne
        coords.append((p[0], q[0],
                        f"Mixed NE ({p[0]:.2f},{q[0]:.2f})", "mixed"))
    return coords


def _ne_coord_rps():
    """Mixed NE for Biased RPS: (R=10/16, P=5/16, S=1/16)."""
    return bary_to_cart(np.array([10/16, 5/16, 1/16]))


def _annotate_ne(ax, x, y, label, kind):
    ax.plot(x, y, ls="none", **(_NE_STAR if kind == "pure" else _MIXED_NE))
    dx = -0.14 if x > 0.5 else +0.14
    dy = -0.12 if y > 0.5 else +0.12
    ax.annotate(
        label, xy=(x, y), xytext=(x + dx, y + dy),
        fontsize=7.5, fontweight="bold", color="#333",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="#999", alpha=0.85, lw=0.7),
        arrowprops=dict(arrowstyle="-|>", color="#555",
                        lw=0.9, mutation_scale=10),
        zorder=15,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Axes formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("$x_1$  (P₁ plays action 0)", fontsize=9)
    ax.set_ylabel("$y_1$  (P₂ plays action 0)", fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Q-batch builder + starts selector
# ─────────────────────────────────────────────────────────────────────────────

def _starts(game: Game):
    if game.custom_starts is not None:
        return game.custom_starts
    return STARTS_2ACTION if game.n_actions == 2 else STARTS_3ACTION


def _alpha(game: Game) -> float:
    return game.custom_alpha if game.custom_alpha is not None else ALPHA


def _q_batch(starts, na):
    q1s, q2s = [], []
    for s in starts:
        if na == 2:
            x0, y0 = s
            q1s.append(init_q_boltzmann([x0, 1-x0], TAU))
            q2s.append(init_q_boltzmann([y0, 1-y0], TAU))
        else:
            p = np.array(s, dtype=float); p /= p.sum()
            q1s.append(init_q_boltzmann(p, TAU))
            q2s.append(init_q_boltzmann(p, TAU))
    return np.array(q1s), np.array(q2s)


# ─────────────────────────────────────────────────────────────────────────────
# Simplex helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_simplex(ax, labels):
    tri = np.vstack([_V, _V[0]])
    ax.plot(tri[:,0], tri[:,1], "k-", lw=1.5, zorder=5)
    offs = [(-0.07,-0.05),(1.04,-0.05),(0.50, np.sqrt(3)/2+0.06)]
    for (ox,oy), lab, h, v in zip(offs, labels,
                                   ["right","left","center"],
                                   ["top","top","bottom"]):
        ax.text(ox, oy, lab, fontsize=10, ha=h, va=v, fontweight="bold")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.14, np.sqrt(3)/2 + 0.20)
    ax.set_aspect("equal"); ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# Core drawing routines
# ─────────────────────────────────────────────────────────────────────────────

def _draw_2action(ax, game, batch_fn, kwargs, n_steps, color, alpha_line, seed):
    st = _starts(game)
    al = _alpha(game)

    X, Y, U, V = replicator_field_2action(game.A, game.B)
    norm = np.hypot(U, V) + 1e-10
    ax.quiver(X, Y, U/norm, V/norm, **_QUIVER)

    Q1b, Q2b = _q_batch(st, 2)
    traj = batch_fn(game.A, game.B, Q1b, Q2b,
                    n_steps=n_steps, alpha=al, seed=seed, **kwargs)
    step = max(1, n_steps // 300)
    for i in range(len(st)):
        xs = traj[::step, i, 0, 0]
        ys = traj[::step, i, 1, 0]
        ax.plot(xs, ys, "-", color=color, lw=1.0, alpha=alpha_line)
        ax.plot(xs[0],  ys[0],  ls="none", color=color,
                markeredgecolor=color, **_START)
        ax.plot(xs[-1], ys[-1], ls="none", color=color,
                markeredgecolor=color, **_END)

    _fmt(ax)

    has_mixed = False
    for x, y, lbl, kind in _ne_coords_2action(game):
        _annotate_ne(ax, x, y, lbl, kind)
        if kind == "mixed":
            has_mixed = True

    handles = [_lh_field(), _lh_line(color, "Trajectory"),
               _lh_start(color), _lh_end(color), _lh_ne()]
    if has_mixed:
        handles.append(_lh_mne())
    ax.legend(handles=handles, fontsize=7.5, loc="best",
              framealpha=0.88, edgecolor="#aaa")


def _draw_simplex_panel(ax, game, batch_fn, kwargs, n_steps, color, alpha_line, seed):
    st = _starts(game)
    al = _alpha(game)

    _draw_simplex(ax, game.actions)
    Xc, Yc, Uc, Vc = replicator_field_simplex(game.A)
    norm = np.hypot(Uc, Vc) + 1e-10
    ax.quiver(Xc, Yc, Uc/norm, Vc/norm, **_QUIVER)

    Q1b, Q2b = _q_batch(st, 3)
    traj = batch_fn(game.A, game.B, Q1b, Q2b,
                    n_steps=n_steps, alpha=al, seed=seed, **kwargs)
    step = max(1, n_steps // 300)
    for i in range(len(st)):
        p1 = traj[::step, i, 0, :]
        xy = bary_to_cart(p1)
        ax.plot(xy[:,0], xy[:,1], "-", color=color, lw=1.0, alpha=alpha_line)
        ax.plot(xy[0,0],  xy[0,1],  ls="none", color=color,
                markeredgecolor=color, **_START)
        ax.plot(xy[-1,0], xy[-1,1], ls="none", color=color,
                markeredgecolor=color, **_END)

    ne_xy = _ne_coord_rps()
    ax.plot(ne_xy[0], ne_xy[1], ls="none", **_MIXED_NE)
    ax.annotate(
        "Mixed NE\n(R=0.63, P=0.31, S=0.06)",
        xy=(ne_xy[0], ne_xy[1]),
        xytext=(ne_xy[0]+0.22, ne_xy[1]-0.10),
        fontsize=7, fontweight="bold", color="#333",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="#999", alpha=0.88, lw=0.7),
        arrowprops=dict(arrowstyle="-|>", color="#555",
                        lw=0.9, mutation_scale=10),
        zorder=15,
    )
    handles = [_lh_field(), _lh_line(color, "Trajectory (P₁)"),
               _lh_start(color), _lh_end(color), _lh_mne()]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right",
              framealpha=0.88, edgecolor="#aaa")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_panel(
    ax,
    game:       Game,
    batch_fn,
    kwargs:     dict  | None = None,
    n_steps:    int          = N_STEPS,
    color:      str          = "#1a1a2e",
    alpha_line: float        = 0.80,
    seed:       int          = 42,
) -> None:
    kwargs = kwargs or {}
    if game.n_actions == 2:
        _draw_2action(ax, game, batch_fn, kwargs, n_steps, color, alpha_line, seed)
    else:
        _draw_simplex_panel(ax, game, batch_fn, kwargs, n_steps, color, alpha_line, seed)


def plot_two_algos_panel(
    ax,
    game:    Game,
    fn1,     kw1: dict,
    fn2,     kw2: dict,
    label1:  str,
    label2:  str,
    col1:    str   = "#1a6faf",
    col2:    str   = "#c0392b",
    n_steps: int   = N_STEPS,
    seed:    int   = 42,
) -> list:
    st = _starts(game)
    na = game.n_actions
    al = _alpha(game)

    if na == 2:
        X, Y, U, V = replicator_field_2action(game.A, game.B)
        norm = np.hypot(U, V) + 1e-10
        ax.quiver(X, Y, U/norm, V/norm, **_QUIVER)
    else:
        _draw_simplex(ax, game.actions)
        Xc, Yc, Uc, Vc = replicator_field_simplex(game.A)
        norm = np.hypot(Uc, Vc) + 1e-10
        ax.quiver(Xc, Yc, Uc/norm, Vc/norm, **_QUIVER)

    algo_handles = []
    for fn, kw, col, lab in [(fn1,kw1,col1,label1),(fn2,kw2,col2,label2)]:
        Q1b, Q2b = _q_batch(st, na)
        traj = fn(game.A, game.B, Q1b, Q2b,
                  n_steps=n_steps, alpha=al, seed=seed, **kw)
        step = max(1, n_steps // 300)
        first = True
        for i in range(len(st)):
            if na == 2:
                xs = traj[::step, i, 0, 0]
                ys = traj[::step, i, 1, 0]
            else:
                pts = bary_to_cart(traj[::step, i, 0, :])
                xs, ys = pts[:,0], pts[:,1]
            h, = ax.plot(xs, ys, "-", color=col, lw=1.0, alpha=0.80,
                         label=lab if first else None)
            ax.plot(xs[0],  ys[0],  ls="none", color=col,
                    markeredgecolor=col, **_START)
            ax.plot(xs[-1], ys[-1], ls="none", color=col,
                    markeredgecolor=col, **_END)
            if first:
                algo_handles.append(h); first = False

    has_mixed = False
    if na == 2:
        _fmt(ax)
        for x, y, lbl, kind in _ne_coords_2action(game):
            _annotate_ne(ax, x, y, lbl, kind)
            if kind == "mixed": has_mixed = True
    else:
        ne_xy = _ne_coord_rps()
        ax.plot(ne_xy[0], ne_xy[1], ls="none", **_MIXED_NE)
        ax.annotate(
            "Mixed NE\n(R=0.63, P=0.31, S=0.06)",
            xy=(ne_xy[0], ne_xy[1]), xytext=(ne_xy[0]+0.22, ne_xy[1]-0.10),
            fontsize=7, fontweight="bold", color="#333",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#999", alpha=0.88, lw=0.7),
            arrowprops=dict(arrowstyle="-|>", color="#555",
                            lw=0.9, mutation_scale=10),
            zorder=15,
        )
        has_mixed = True

    handles = [_lh_field(), _lh_line(col1, label1), _lh_line(col2, label2),
               _lh_start("gray"), _lh_end("gray")]
    if na == 2 and game.nash_pure():
        handles.append(_lh_ne())
    if has_mixed:
        handles.append(_lh_mne())
    ax.legend(handles=handles, fontsize=7.5,
              loc="lower right" if na != 2 else "best",
              framealpha=0.88, edgecolor="#aaa")
    return algo_handles


def save_fig(fig, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p.name}")
