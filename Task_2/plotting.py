import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(all_trajectories, game, agent_name, nash_equilibria):
    n_actions = game.n_actions_p1

    if n_actions == 2:
        _plot_2action(all_trajectories, game, agent_name, nash_equilibria)
    else:
        _plot_naction(all_trajectories, game, agent_name, n_actions, nash_equilibria)


def _plot_2action(all_trajectories, game, agent_name, nash_equilibria):
    """
    For 2-action games: plot each run as a trajectory in the 2D unit simplex.
    X-axis = P1's P(action 0),  Y-axis = P2's P(action 0).
    The trajectory moves through this space as Q-values (and thus policies) evolve.
    Nash Equilibria are marked as red dots.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, trajectory in enumerate(all_trajectories):
        # trajectory shape: (n_episodes, 2*n_actions)
        # first 2 entries per step are p1's policy, next 2 are p2's policy
        p1_prob_action0 = trajectory[:, 0]   # P1: P(action 0)
        p2_prob_action0 = trajectory[:, 2]   # P2: P(action 0)  ← index 2 because p2 starts after p1's 2 actions

        label = "run" if i == 0 else None
        ax.plot(p1_prob_action0, p2_prob_action0,
                color="steelblue", alpha=0.35, linewidth=1.0, label=label)

        # Mark start (circle) and end (star) of each run so direction is visible
        ax.scatter(p1_prob_action0[0],  p2_prob_action0[0],
                   color="green", s=30, zorder=4, alpha=0.5)
        ax.scatter(p1_prob_action0[-1], p2_prob_action0[-1],
                   color="orange", marker="*", s=80, zorder=4, alpha=0.7)

    # Mark Nash Equilibria
    for i, ne in enumerate(nash_equilibria):
        label = "Nash Eq." if i == 0 else None
        ax.scatter(ne[0], ne[1], color="red", s=150, zorder=5,
                   marker="X", label=label, edgecolors="black", linewidths=0.5)

    # Legend entries for start/end markers
    ax.scatter([], [], color="green",  s=30,  label="Start")
    ax.scatter([], [], color="orange", s=80, marker="*", label="End")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("P1: P(action 0)", fontsize=12)
    ax.set_ylabel("P2: P(action 0)", fontsize=12)
    ax.set_title(f"{game.name} — {agent_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}.png", dpi=150)
    plt.show()


def _plot_naction(all_trajectories, game, agent_name, n_actions, nash_equilibria=None):
    """
    For 3-action games (biased RPS): plot P(each action) over time per player.
    A 2D unit-simplex plot would lose information for 3+ actions,
    so we plot probability vs episode instead.
    """
    action_labels_p1 = [f"P1: P(a{i})" for i in range(n_actions)]
    action_labels_p2 = [f"P2: P(a{i})" for i in range(n_actions)]
    colors = ["steelblue", "tomato", "seagreen"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    episodes = np.arange(all_trajectories.shape[1])

    for run_idx, trajectory in enumerate(all_trajectories):
        for a in range(n_actions):
            p1_prob = trajectory[:, a]            # P1's prob of action a
            p2_prob = trajectory[:, n_actions + a]  # P2's prob of action a
            lbl1 = action_labels_p1[a] if run_idx == 0 else None
            lbl2 = action_labels_p2[a] if run_idx == 0 else None
            axes[0].plot(episodes, p1_prob, color=colors[a], alpha=0.3, label=lbl1)
            axes[1].plot(episodes, p2_prob, color=colors[a], alpha=0.3, label=lbl2)

    if nash_equilibria and len(nash_equilibria) > 0:
        ne = nash_equilibria[0]
        for a in range(n_actions):
            # Draw the target line for each action
            axes[0].axhline(ne[a], color=colors[a], linestyle='--', alpha=0.8, linewidth=2)
            axes[1].axhline(ne[a], color=colors[a], linestyle='--', alpha=0.8, linewidth=2)

    for ax, player in zip(axes, ["Player 1", "Player 2"]):
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Policy probability", fontsize=11)
        ax.set_title(f"{game.name} — {agent_name}\n{player}", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}.png", dpi=150)
    plt.show()


def compute_lenient_expected_rewards(payoff_matrix, opp_policy, kappa):
    """
    Computes the expected reward for 2 actions if the agent takes the
    maximum reward over kappa trials (Panait et al. 2008).
    """
    expected_rewards = np.zeros(2)
    for i in range(2):
        rewards = payoff_matrix[i, :]
        # Sort rewards descending to find max and min
        sort_idx = np.argsort(rewards)[::-1]
        sorted_rewards = rewards[sort_idx]
        sorted_probs = opp_policy[sort_idx]

        # Prob of getting the minimum payoff is getting it all kappa times
        p_min_payoff = sorted_probs[1] ** kappa
        # Prob of getting the max payoff is 1 - p_min_payoff
        p_max_payoff = 1.0 - p_min_payoff

        expected_rewards[i] = (sorted_rewards[0] * p_max_payoff) + (sorted_rewards[1] * p_min_payoff)

    return expected_rewards


def plot_directional_field(game_name, payoffs_P1, payoffs_P2, nash_equilibria, tau=0.3, kappa=5):
    """
    Plots the analytical replicator dynamics (streamlines) for
    Boltzmann and Lenient Boltzmann Q-learning.
    """
    grid_resolution = 25
    # Use 0.01 to 0.99 to avoid np.log(0) errors at the pure strategy boundaries
    x = np.linspace(0.01, 0.99, grid_resolution)
    y = np.linspace(0.01, 0.99, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Arrays for Boltzmann expected change
    dX_B, dY_B = np.zeros(X.shape), np.zeros(Y.shape)
    # Arrays for Lenient Boltzmann expected change
    dX_L, dY_L = np.zeros(X.shape), np.zeros(Y.shape)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            p1_0, p2_0 = X[i, j], Y[i, j]
            pi_1 = np.array([p1_0, 1 - p1_0])
            pi_2 = np.array([p2_0, 1 - p2_0])

            # --- 1. Standard Expected Rewards ---
            E_P1 = payoffs_P1 @ pi_2
            E_P2 = payoffs_P2.T @ pi_1

            # Entropy terms (sum(x * ln(x)))
            ent_P1 = np.sum(pi_1 * np.log(pi_1))
            ent_P2 = np.sum(pi_2 * np.log(pi_2))

            # Boltzmann Dynamics (Tuyls et al. 2003)
            dX_B[i, j] = pi_1[0] * ((E_P1[0] - np.dot(pi_1, E_P1)) / tau + ent_P1 - np.log(pi_1[0]))
            dY_B[i, j] = pi_2[0] * ((E_P2[0] - np.dot(pi_2, E_P2)) / tau + ent_P2 - np.log(pi_2[0]))

            # --- 2. Lenient Expected Rewards ---
            E_P1_len = compute_lenient_expected_rewards(payoffs_P1, pi_2, kappa)
            E_P2_len = compute_lenient_expected_rewards(payoffs_P2.T, pi_1, kappa)

            # Lenient Boltzmann Dynamics
            dX_L[i, j] = pi_1[0] * ((E_P1_len[0] - np.dot(pi_1, E_P1_len)) / tau + ent_P1 - np.log(pi_1[0]))
            dY_L[i, j] = pi_2[0] * ((E_P2_len[0] - np.dot(pi_2, E_P2_len)) / tau + ent_P2 - np.log(pi_2[0]))

    # --- Plotting Helper ---
    def draw_plot(dX, dY, title, filename):
        fig, ax = plt.subplots(figsize=(6, 6))

        # 1. The Background Arrows (Quiver)
        # We slice the grid [::2] so the arrows aren't too crowded
        skip = (slice(None, None, 2), slice(None, None, 2))
        ax.quiver(X[skip], Y[skip], dX[skip], dY[skip],
                  color='steelblue', alpha=0.5, width=0.003, headwidth=4)

        # 2. The Thick Trajectory Lines (Streamplot)
        # Define starting coordinates near the edges to mimic the book's lines
        starts = []
        for val in np.linspace(0.05, 0.95, 7):
            starts.append([0.01, val])  # Left edge
            starts.append([val, 0.01])  # Bottom edge
            starts.append([0.99, val])  # Right edge
            starts.append([val, 0.99])  # Top edge
        start_points = np.array(starts)

        # Plot the thick black lines. arrowstyle='-' removes the arrows from the lines themselves
        ax.streamplot(X, Y, dX, dY, color='black', linewidth=1.0,
                      arrowstyle='-', start_points=start_points, density=0.5)

        # 3. Mark the Nash Equilibria
        for idx, ne in enumerate(nash_equilibria):
            label = "Nash Eq." if idx == 0 else None
            ax.scatter(ne[0], ne[1], color="red", s=150, zorder=5, marker="X",
                       label=label, edgecolors="black", linewidths=0.5)

        # Formatting
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("P1: P(action 0)", fontsize=12)
        ax.set_ylabel("P2: P(action 0)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        if nash_equilibria:
            ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)

        # WE REMOVED plt.close() HERE.
        # Now the figures will stay open until plt.show() is called at the end!

    # Generate both plots
    draw_plot(dX_B, dY_B, f"{game_name} — Analytical Boltzmann", f"{game_name}_Analytical_Boltzmann.png")
    draw_plot(dX_L, dY_L, f"{game_name} — Analytical Lenient_Boltzmann",
              f"{game_name}_Analytical_Lenient_Boltzmann.png")

    # This will now successfully display both windows
    plt.show()