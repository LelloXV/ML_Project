import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def plot_trajectories(all_trajectories, game, agent_name, nash_equilibria):
    """ Routes to the correct trajectory plotter based on the number of actions. """
    n_actions = game.n_actions_p1

    if n_actions == 2:
        _plot_2action(all_trajectories, game, agent_name, nash_equilibria)
    elif n_actions == 3:
        _plot_3action_ternary(all_trajectories, game, agent_name, nash_equilibria)
    else:
        _plot_naction(all_trajectories, game, agent_name, n_actions, nash_equilibria)


def _plot_2action(all_trajectories, game, agent_name, nash_equilibria):
    fig, ax = plt.subplots(figsize=(7, 7))

    run_colors = plt.cm.tab20(np.linspace(0, 1, len(all_trajectories)))

    for i, trajectory in enumerate(all_trajectories):
        p1_prob_action0 = trajectory[:, 0]
        p2_prob_action0 = trajectory[:, 2]

        ax.plot(p1_prob_action0, p2_prob_action0, color=run_colors[i], alpha=0.6, linewidth=1.2, zorder=2)
        ax.scatter(p1_prob_action0[0], p2_prob_action0[0], color="green", s=40, zorder=4, alpha=0.8)
        ax.scatter(p1_prob_action0[-1], p2_prob_action0[-1], color="orange", marker="*", s=100, zorder=4, alpha=0.9)

    if nash_equilibria:
        for i, ne in enumerate(nash_equilibria):
            label = "Nash Eq." if i == 0 else None
            ax.scatter(ne[0], ne[1], color="red", s=150, zorder=5, marker="X", label=label, edgecolors="black",
                       linewidths=1.0)

    handles, labels = ax.get_legend_handles_labels()
    traj_line = mlines.Line2D([], [], color='gray', alpha=0.6, label='Trajectories')
    start_marker = ax.scatter([], [], color="green", s=40, label="Start")
    end_marker = ax.scatter([], [], color="orange", s=100, marker="*", label="End")

    handles.extend([traj_line, start_marker, end_marker])
    labels.extend(['Trajectories', 'Start', 'End'])

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("P1: P(action 0)", fontsize=12)
    ax.set_ylabel("P2: P(action 0)", fontsize=12)

    ax.set_title(f"{game.name} — {agent_name}", fontsize=14, fontweight="bold", pad=20)

    # Legend pushed outside to the right
    ax.legend(handles=handles, labels=labels, fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    # bbox_inches='tight' prevents the external legend from being cut off
    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()  # Better practice to close the plot automatically


def _plot_naction(all_trajectories, game, agent_name, n_actions, nash_equilibria=None):
    action_labels_p1 = [f"P1: P(a{i})" for i in range(n_actions)]
    action_labels_p2 = [f"P2: P(a{i})" for i in range(n_actions)]
    colors = plt.cm.Set1(np.linspace(0, 1, n_actions))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    episodes = np.arange(all_trajectories.shape[1])

    for run_idx, trajectory in enumerate(all_trajectories):
        for a in range(n_actions):
            p1_prob = trajectory[:, a]
            p2_prob = trajectory[:, n_actions + a]
            lbl1 = action_labels_p1[a] if run_idx == 0 else None
            lbl2 = action_labels_p2[a] if run_idx == 0 else None

            axes[0].plot(episodes, p1_prob, color=colors[a], alpha=0.25, linewidth=1.5, label=lbl1)
            axes[1].plot(episodes, p2_prob, color=colors[a], alpha=0.25, linewidth=1.5, label=lbl2)

    if nash_equilibria and len(nash_equilibria) > 0:
        ne = nash_equilibria[0]
        for a in range(n_actions):
            axes[0].axhline(ne[a], color=colors[a], linestyle='--', alpha=0.9, linewidth=2)
            axes[1].axhline(ne[a], color=colors[a], linestyle='--', alpha=0.9, linewidth=2)

        nash_line = mlines.Line2D([], [], color='black', linestyle='--', label='Nash Eq. Target')
        # Only put the legend on the second plot to save space
        axes[1].legend(handles=axes[1].get_legend_handles_labels()[0] + [nash_line], fontsize=10, loc='center left',
                       bbox_to_anchor=(1.05, 0.5))
    else:
        axes[1].legend(fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))

    for ax, player in zip(axes, ["Player 1", "Player 2"]):
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Policy probability", fontsize=12)
        ax.set_title(f"{game.name} — {agent_name}\n{player}", fontsize=14, fontweight="bold", pad=20)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_3action_ternary(all_trajectories, game, agent_name, nash_equilibria):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    def ternary_to_cartesian(p):
        p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
        x = p1 + 0.5 * p2
        y = (np.sqrt(3) / 2) * p2
        return x, y

    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]
    run_colors = plt.cm.tab20(np.linspace(0, 1, len(all_trajectories)))

    for ax, player_idx, title in zip(axes, [0, 1], ["Player 1", "Player 2"]):
        ax.plot(triangle_x, triangle_y, color='black', linewidth=1.5, zorder=1)

        for i, trajectory in enumerate(all_trajectories):
            start_col = player_idx * 3
            end_col = start_col + 3
            p = trajectory[:, start_col:end_col]
            x, y = ternary_to_cartesian(p)

            ax.plot(x, y, color=run_colors[i], alpha=0.6, linewidth=1.2, zorder=2)
            ax.scatter(x[0], y[0], color="green", s=40, zorder=4, alpha=0.8)
            ax.scatter(x[-1], y[-1], color="orange", marker="*", s=100, zorder=4, alpha=0.9)

        if nash_equilibria:
            for idx, ne in enumerate(nash_equilibria):
                ne_arr = np.array([ne])
                ne_x, ne_y = ternary_to_cartesian(ne_arr)
                label = "Nash Eq." if idx == 0 else None
                ax.scatter(ne_x[0], ne_y[0], color="red", s=150, zorder=5,
                           marker="X", label=label, edgecolors="black", linewidths=1.0)

        ax.text(-0.02, -0.05, "Action 0", fontsize=12, ha='right', va='top')
        ax.text(1.02, -0.05, "Action 1", fontsize=12, ha='left', va='top')
        ax.text(0.5, np.sqrt(3) / 2 + 0.05, "Action 2", fontsize=12, ha='center', va='bottom')

        # Generate a unified legend only on the Player 2 axis
        if player_idx == 1:
            handles, labels = ax.get_legend_handles_labels()
            traj_line = mlines.Line2D([], [], color='gray', alpha=0.6, label='Trajectories')
            start_marker = ax.scatter([], [], color="green", s=40, label="Start")
            end_marker = ax.scatter([], [], color="orange", s=100, marker="*", label="End")

            handles.extend([traj_line, start_marker, end_marker])
            labels.extend(['Trajectories', 'Start', 'End'])
            ax.legend(handles=handles, labels=labels, fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))

        ax.set_title(f"{game.name} — {agent_name}\n{title}", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.2)
        ax.axis('off')
        ax.set_aspect('equal')

    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}_ternary.png", dpi=150, bbox_inches='tight')
    plt.close()


def compute_lenient_expected_rewards(payoff_matrix, opp_policy, kappa):
    n_actions_p1, n_actions_p2 = payoff_matrix.shape
    expected_rewards = np.zeros(n_actions_p1)

    for i in range(n_actions_p1):
        rewards = payoff_matrix[i, :]
        sort_idx = np.argsort(rewards)[::-1]
        sorted_rewards = rewards[sort_idx]
        sorted_probs = opp_policy[sort_idx]

        F = np.zeros(n_actions_p2)
        for k in range(n_actions_p2):
            F[k] = np.sum(sorted_probs[k:])

        F_kappa = F ** kappa
        F_next_kappa = np.append(F_kappa[1:], 0.0)
        prob_max_is_k = F_kappa - F_next_kappa
        expected_rewards[i] = np.sum(sorted_rewards * prob_max_is_k)

    return expected_rewards


def plot_directional_field(game_name, payoffs_P1, payoffs_P2, nash_equilibria, tau=0.3, kappa=5):
    if payoffs_P1.shape[0] == 2:
        _plot_2action_directional_field(game_name, payoffs_P1, payoffs_P2, nash_equilibria, tau, kappa)
    elif payoffs_P1.shape[0] == 3:
        _plot_3action_directional_field(game_name, payoffs_P1, nash_equilibria, tau, kappa)


def _plot_2action_directional_field(game_name, payoffs_P1, payoffs_P2, nash_equilibria, tau, kappa):
    grid_resolution = 25
    x = np.linspace(0.01, 0.99, grid_resolution)
    y = np.linspace(0.01, 0.99, grid_resolution)
    X, Y = np.meshgrid(x, y)

    dX_B, dY_B = np.zeros(X.shape), np.zeros(Y.shape)
    dX_L, dY_L = np.zeros(X.shape), np.zeros(Y.shape)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            p1_0, p2_0 = X[i, j], Y[i, j]
            pi_1 = np.array([p1_0, 1 - p1_0])
            pi_2 = np.array([p2_0, 1 - p2_0])

            E_P1 = payoffs_P1 @ pi_2
            E_P2 = payoffs_P2.T @ pi_1
            ent_P1 = np.sum(pi_1 * np.log(pi_1))
            ent_P2 = np.sum(pi_2 * np.log(pi_2))

            dX_B[i, j] = pi_1[0] * ((E_P1[0] - np.dot(pi_1, E_P1)) / tau + ent_P1 - np.log(pi_1[0]))
            dY_B[i, j] = pi_2[0] * ((E_P2[0] - np.dot(pi_2, E_P2)) / tau + ent_P2 - np.log(pi_2[0]))

            E_P1_len = compute_lenient_expected_rewards(payoffs_P1, pi_2, kappa)
            E_P2_len = compute_lenient_expected_rewards(payoffs_P2.T, pi_1, kappa)

            dX_L[i, j] = pi_1[0] * ((E_P1_len[0] - np.dot(pi_1, E_P1_len)) / tau + ent_P1 - np.log(pi_1[0]))
            dY_L[i, j] = pi_2[0] * ((E_P2_len[0] - np.dot(pi_2, E_P2_len)) / tau + ent_P2 - np.log(pi_2[0]))

    _draw_field_plot(X, Y, dX_B, dY_B, f"{game_name} — Analytical Boltzmann", nash_equilibria,
                     f"{game_name}_Analytical_Boltzmann.png")
    _draw_field_plot(X, Y, dX_L, dY_L, f"{game_name} — Analytical Lenient Boltzmann", nash_equilibria,
                     f"{game_name}_Analytical_Lenient_Boltzmann.png")


def _plot_3action_directional_field(game_name, payoffs_P1, nash_equilibria, tau, kappa):
    grid_resolution = 40
    X_vals = np.linspace(-0.05, 1.05, grid_resolution)
    Y_vals = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid_resolution)
    X, Y = np.meshgrid(X_vals, Y_vals)

    dX_B, dY_B = np.zeros(X.shape), np.zeros(Y.shape)
    dX_L, dY_L = np.zeros(X.shape), np.zeros(Y.shape)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            x_cart, y_cart = X[i, j], Y[i, j]

            p2 = y_cart / (np.sqrt(3) / 2)
            p1 = x_cart - 0.5 * p2
            p0 = 1.0 - p1 - p2

            if p0 >= -0.01 and p1 >= -0.01 and p2 >= -0.01:
                p = np.array([p0, p1, p2])
                p = np.clip(p, 0.001, 1.0)
                p = p / np.sum(p)

                E_B = payoffs_P1 @ p
                ent_P = np.sum(p * np.log(p))

                dp_B = p * ((E_B - np.dot(p, E_B)) / tau + ent_P - np.log(p))

                dX_B[i, j] = dp_B[1] + 0.5 * dp_B[2]
                dY_B[i, j] = (np.sqrt(3) / 2) * dp_B[2]

                E_L = compute_lenient_expected_rewards(payoffs_P1, p, kappa)
                dp_L = p * ((E_L - np.dot(p, E_L)) / tau + ent_P - np.log(p))

                dX_L[i, j] = dp_L[1] + 0.5 * dp_L[2]
                dY_L[i, j] = (np.sqrt(3) / 2) * dp_L[2]

    mask = (Y < 0) | (X < Y / np.sqrt(3)) | (X > 1 - Y / np.sqrt(3))
    dX_B = np.ma.masked_where(mask, dX_B)
    dY_B = np.ma.masked_where(mask, dY_B)
    dX_L = np.ma.masked_where(mask, dX_L)
    dY_L = np.ma.masked_where(mask, dY_L)

    _draw_field_plot(X, Y, dX_B, dY_B, f"{game_name} — Analytical Boltzmann (Ternary)", nash_equilibria,
                     f"{game_name}_Analytical_Boltzmann_Ternary.png", is_ternary=True)
    _draw_field_plot(X, Y, dX_L, dY_L, f"{game_name} — Analytical Lenient Boltzmann (Ternary)", nash_equilibria,
                     f"{game_name}_Analytical_Lenient_Boltzmann_Ternary.png", is_ternary=True)


def _draw_field_plot(X, Y, dX, dY, title, nash_equilibria, filename, is_ternary=False):
    fig, ax = plt.subplots(figsize=(7, 7))

    if is_ternary:
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3) / 2, 0]
        ax.plot(triangle_x, triangle_y, color='black', linewidth=1.5, zorder=1)

        skip = (slice(None, None, 2), slice(None, None, 2))
        ax.quiver(X[skip], Y[skip], dX[skip], dY[skip], color='steelblue', alpha=0.5, width=0.003, headwidth=4)
        ax.streamplot(X, Y, dX, dY, color='black', linewidth=1.0, density=1.2, arrowstyle='->')

        ax.text(-0.02, -0.05, "Action 0", fontsize=12, ha='right', va='top')
        ax.text(1.02, -0.05, "Action 1", fontsize=12, ha='left', va='top')
        ax.text(0.5, np.sqrt(3) / 2 + 0.05, "Action 2", fontsize=12, ha='center', va='bottom')

        # Expand limits to clear the title from the top vertex
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.25)
        ax.axis('off')
    else:
        starts = []
        for val in np.linspace(0.05, 0.95, 7):
            starts.extend([[0.01, val], [val, 0.01], [0.99, val], [val, 0.99]])

        skip = (slice(None, None, 2), slice(None, None, 2))
        ax.quiver(X[skip], Y[skip], dX[skip], dY[skip], color='steelblue', alpha=0.5, width=0.003, headwidth=4)
        ax.streamplot(X, Y, dX, dY, color='black', linewidth=1.0, arrowstyle='->', start_points=np.array(starts),
                      density=0.5)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("P1: P(action 0)", fontsize=12)
        ax.set_ylabel("P2: P(action 0)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx, ne in enumerate(nash_equilibria):
        if is_ternary:
            ne_x = ne[1] + 0.5 * ne[2]
            ne_y = (np.sqrt(3) / 2) * ne[2]
        else:
            ne_x, ne_y = ne[0], ne[1]

        label = "Nash Eq." if idx == 0 else None
        ax.scatter(ne_x, ne_y, color="red", s=150, zorder=5, marker="X", label=label, edgecolors="black",
                   linewidths=1.0)

    # Added pad=25 to ensure the title never hits the top boundary
    ax.set_title(title, fontsize=14, fontweight="bold", pad=25)
    ax.set_aspect("equal")

    if nash_equilibria:
        # Pushed legend outside
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Using bbox_inches='tight' again to prevent cropping
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()