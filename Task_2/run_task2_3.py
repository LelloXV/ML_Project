"""
run_task2_3.py – Task 2.3: replicator dynamics + Boltzmann vs Lenient overlay.

Outputs in output/task2_3/:
  <game>_boltzmann.png    – Boltzmann trajectories on replicator field
  <game>_lenient.png      – Lenient Boltzmann trajectories on replicator field
  <game>_comparison.png   – both overlaid (blue vs red) with full legend
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config     import OUT_2_3, N_STEPS
from games      import ALL_GAMES
from algorithms import boltzmann_batch, lenient_boltzmann_batch
from dynamics   import print_replicator_equations
from plotting   import plot_panel, plot_two_algos_panel, save_fig

COL_BOLT = "#1a6faf"
COL_LENI = "#c0392b"
LBL_BOLT = "Boltzmann Q-learning"
LBL_LENI = "Lenient Boltzmann Q-learning"


def main():
    print("=" * 60)
    print("  Task 2.3  –  Replicator Dynamics & Verification")
    print("=" * 60)

    # ── 1. Print analytical replicator equations ──────────────────────────────
    print("\n── Analytical replicator equations ──────────────────────────")
    for g in ALL_GAMES:
        print_replicator_equations(g.A, g.B, g.name)

    # ── 2. Individual algorithm plots ─────────────────────────────────────────
    # plot_panel() builds the full legend internally — no extra ax.legend() needed
    print("\n── Individual plots ─────────────────────────────────────────")
    for game in ALL_GAMES:
        for batch_fn, slug, col, label in [
            (boltzmann_batch,         "boltzmann", COL_BOLT, LBL_BOLT),
            (lenient_boltzmann_batch, "lenient",   COL_LENI, LBL_LENI),
        ]:
            fig, ax = plt.subplots(figsize=(5, 4.8), constrained_layout=True)
            fig.suptitle(f"{label}\n{game.name}", fontsize=11, fontweight="bold")

            # plot_panel handles: quiver, trajectories, NE markers, full legend
            plot_panel(ax, game, batch_fn, {}, n_steps=N_STEPS,
                       color=col, alpha_line=0.85)

            save_fig(fig, OUT_2_3 / f"{game.slug}_{slug}.png")

    # ── 3. Comparison overlay plots ───────────────────────────────────────────
    # plot_two_algos_panel() handles: quiver, both trajectories, NE, full legend
    print("\n── Comparison plots ─────────────────────────────────────────")
    for game in ALL_GAMES:
        fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
        fig.suptitle(f"Boltzmann vs Lenient Boltzmann\n{game.name}",
                     fontsize=11, fontweight="bold")

        plot_two_algos_panel(
            ax, game,
            fn1=boltzmann_batch,         kw1={}, label1=LBL_BOLT, col1=COL_BOLT,
            fn2=lenient_boltzmann_batch, kw2={}, label2=LBL_LENI, col2=COL_LENI,
            n_steps=N_STEPS,
        )

        save_fig(fig, OUT_2_3 / f"{game.slug}_comparison.png")

    print(f"\nAll plots → {OUT_2_3.resolve()}")


if __name__ == "__main__":
    main()
