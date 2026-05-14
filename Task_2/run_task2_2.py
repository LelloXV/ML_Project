"""
run_task2_2.py – Task 2.2: one plot per (algorithm × game).

Output: output/task2_2/<algo>_<game>.png  (12 files)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config     import OUT_2_2, N_STEPS
from games      import ALL_GAMES
from algorithms import ALGORITHMS
from plotting   import plot_panel, save_fig

# ε-greedy in PD oscillates and needs significantly more steps to show
# meaningful time-averaged convergence behaviour.
_STEP_OVERRIDES = {
    ("egreedy", "prisoners_dilemma"): 120_000,
}

def main():
    print("=" * 60)
    print("  Task 2.2  –  Learning Dynamics in Matrix Games")
    print("=" * 60)

    for alg in ALGORITHMS:
        for game in ALL_GAMES:
            n = _STEP_OVERRIDES.get((alg["slug"], game.slug), N_STEPS)

            fig, ax = plt.subplots(figsize=(5, 4.8), constrained_layout=True)
            fig.suptitle(
                f"{alg['name']}  –  {game.name}",
                fontsize=11, fontweight="bold",
            )
            plot_panel(ax, game, alg["batch_fn"], alg["kwargs"], n_steps=n)
            save_fig(fig, OUT_2_2 / f"{alg['slug']}_{game.slug}.png")

    print(f"\nAll plots → {OUT_2_2.resolve()}")

if __name__ == "__main__":
    main()
