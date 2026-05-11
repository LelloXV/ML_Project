import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def plot_trajectories(all_trajectories, game, agent_name, nash_equilibria):
    n_actions = game.n_actions_p1
    if n_actions == 2:
        _plot_2action(all_trajectories, game, agent_name, nash_equilibria)
    elif n_actions == 3:
        _plot_3action_ternary(all_trajectories, game, agent_name, nash_equilibria)
    else:
        _plot_naction(all_trajectories, game, agent_name, n_actions, nash_equilibria)


def _plot_2action(all_trajectories, game, agent_name, nash_equilibria):
    fig, ax = plt.subplots(figsize=(7, 7))
    run_colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_trajectories), 1)))

    for i, trajectory in enumerate(all_trajectories):
        p1 = trajectory[:, 0]
        p2 = trajectory[:, 2]
        # Subsample to avoid dense zigzag lines
        step = max(1, len(p1) // 500)
        p1_s, p2_s = p1[::step], p2[::step]
        ax.plot(p1_s, p2_s, color=run_colors[i], alpha=0.6, linewidth=1.2, zorder=2)
        ax.scatter(p1[0], p2[0], color="green", s=40, zorder=4, alpha=0.8)
        ax.scatter(p1[-1], p2[-1], color="orange", marker="*", s=100, zorder=4, alpha=0.9)

    for i, ne in enumerate(nash_equilibria):
        ax.scatter(ne[0], ne[1], color="red", s=150, zorder=5, marker="X",
                   label="Nash Eq." if i == 0 else None, edgecolors="black", linewidths=1.0)

    handles, labels = ax.get_legend_handles_labels()
    handles += [mlines.Line2D([], [], color='gray', alpha=0.6),
                ax.scatter([], [], color="green", s=40),
                ax.scatter([], [], color="orange", s=100, marker="*")]
    labels += ['Trajectories', 'Start', 'End']

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("P1: P(action 0)", fontsize=12)
    ax.set_ylabel("P2: P(action 0)", fontsize=12)
    ax.set_title(f"{game.name} — {agent_name}", fontsize=14, fontweight="bold", pad=20)
    ax.legend(handles=handles, labels=labels, fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_naction(all_trajectories, game, agent_name, n_actions, nash_equilibria=None):
    action_labels_p1 = [f"P1: P(a{i})" for i in range(n_actions)]
    action_labels_p2 = [f"P2: P(a{i})" for i in range(n_actions)]
    colors = plt.cm.Set1(np.linspace(0, 1, n_actions))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    episodes = np.arange(all_trajectories.shape[1])

    for run_idx, trajectory in enumerate(all_trajectories):
        for a in range(n_actions):
            axes[0].plot(episodes, trajectory[:, a], color=colors[a], alpha=0.25, linewidth=1.5,
                         label=action_labels_p1[a] if run_idx == 0 else None)
            axes[1].plot(episodes, trajectory[:, n_actions + a], color=colors[a], alpha=0.25, linewidth=1.5,
                         label=action_labels_p2[a] if run_idx == 0 else None)

    if nash_equilibria and len(nash_equilibria) > 0:
        ne = nash_equilibria[0]
        for a in range(n_actions):
            axes[0].axhline(ne[a], color=colors[a], linestyle='--', alpha=0.9, linewidth=2)
            axes[1].axhline(ne[a], color=colors[a], linestyle='--', alpha=0.9, linewidth=2)
        nash_line = mlines.Line2D([], [], color='black', linestyle='--', label='Nash Eq. Target')
        axes[1].legend(handles=axes[1].get_legend_handles_labels()[0] + [nash_line],
                       fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))
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
        return p[:, 1] + 0.5 * p[:, 2], (np.sqrt(3) / 2) * p[:, 2]

    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]
    run_colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_trajectories), 1)))

    for ax, player_idx, title in zip(axes, [0, 1], ["Player 1", "Player 2"]):
        ax.plot(triangle_x, triangle_y, color='black', linewidth=1.5, zorder=1)

        for i, trajectory in enumerate(all_trajectories):
            start_col = player_idx * 3
            p = trajectory[:, start_col:start_col + 3]
            # Subsample to at most 200 points to avoid clutter
            step = max(1, len(p) // 200)
            p_s = p[::step]
            x, y = ternary_to_cartesian(p_s)
            x0, y0 = ternary_to_cartesian(p[:1])
            xf, yf = ternary_to_cartesian(p[-1:])
            ax.plot(x, y, color=run_colors[i], alpha=0.6, linewidth=1.2, zorder=2)
            ax.scatter(x0[0], y0[0], color="green", s=40, zorder=4, alpha=0.8)
            ax.scatter(xf[0], yf[0], color="orange", marker="*", s=100, zorder=4, alpha=0.9)

        if nash_equilibria:
            for idx, ne in enumerate(nash_equilibria):
                ne_x, ne_y = ternary_to_cartesian(np.array([ne]))
                ax.scatter(ne_x[0], ne_y[0], color="red", s=150, zorder=5, marker="X",
                           label="Nash Eq." if idx == 0 else None, edgecolors="black", linewidths=1.0)

        ax.text(-0.02, -0.05, "Action 0", fontsize=12, ha='right', va='top')
        ax.text(1.02, -0.05, "Action 1", fontsize=12, ha='left', va='top')
        ax.text(0.5, np.sqrt(3) / 2 + 0.05, "Action 2", fontsize=12, ha='center', va='bottom')

        if player_idx == 1:
            handles, labels = ax.get_legend_handles_labels()
            handles += [mlines.Line2D([], [], color='gray', alpha=0.6),
                        ax.scatter([], [], color="green", s=40),
                        ax.scatter([], [], color="orange", s=100, marker="*")]
            labels += ['Trajectories', 'Start', 'End']
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


def plot_combined_field_and_trajectories(game, agent_name, trajectories, nash_equilibria, tau=0.3, kappa=5):
    """Replicator dynamics field (gray) + analytical Boltzmann/Lenient field (colored) + trajectories."""
    if game.n_actions_p1 == 2:
        _plot_2action_combined(game, agent_name, trajectories, nash_equilibria, tau, kappa)
    elif game.n_actions_p1 == 3:
        _plot_3action_combined(game, agent_name, trajectories, nash_equilibria, tau, kappa)


def _plot_2action_combined(game, agent_name, trajectories, nash_equilibria, tau, kappa):
    grid = 25
    x = np.linspace(0.01, 0.99, grid)
    y = np.linspace(0.01, 0.99, grid)
    X, Y = np.meshgrid(x, y)

    dX_rep, dY_rep = np.zeros(X.shape), np.zeros(Y.shape)
    dX_ana, dY_ana = np.zeros(X.shape), np.zeros(Y.shape)

    for i in range(grid):
        for j in range(grid):
            pi_1 = np.array([X[i, j], 1 - X[i, j]])
            pi_2 = np.array([Y[i, j], 1 - Y[i, j]])

            E1 = game.payoffs_p1 @ pi_2
            E2 = game.payoffs_p2.T @ pi_1
            dX_rep[i, j] = pi_1[0] * (E1[0] - pi_1 @ E1)
            dY_rep[i, j] = pi_2[0] * (E2[0] - pi_2 @ E2)

            ent_1 = np.sum(pi_1 * np.log(pi_1))
            ent_2 = np.sum(pi_2 * np.log(pi_2))
            if agent_name == "Boltzmann":
                E1_ana, E2_ana = E1, E2
            else:
                E1_ana = compute_lenient_expected_rewards(game.payoffs_p1, pi_2, kappa)
                E2_ana = compute_lenient_expected_rewards(game.payoffs_p2.T, pi_1, kappa)
            dX_ana[i, j] = pi_1[0] * ((E1_ana[0] - pi_1 @ E1_ana) / tau + ent_1 - np.log(pi_1[0]))
            dY_ana[i, j] = pi_2[0] * ((E2_ana[0] - pi_2 @ E2_ana) / tau + ent_2 - np.log(pi_2[0]))

    is_lenient = (agent_name == "Lenient Boltzmann")
    field_color = 'darkorange' if is_lenient else 'steelblue'
    run_colors = (plt.cm.Oranges if is_lenient else plt.cm.Blues)(
        np.linspace(0.45, 0.9, max(len(trajectories), 1)))

    fig, ax = plt.subplots(figsize=(7, 7))

    starts = []
    for val in np.linspace(0.05, 0.95, 7):
        starts.extend([[0.01, val], [val, 0.01], [0.99, val], [val, 0.99]])
    ax.streamplot(X, Y, dX_rep, dY_rep, color='lightgray', linewidth=0.9, arrowstyle='->',
                  start_points=np.array(starts), density=0.5)

    skip = (slice(None, None, 2), slice(None, None, 2))
    ax.quiver(X[skip], Y[skip], dX_ana[skip], dY_ana[skip],
              color=field_color, alpha=0.5, width=0.003, headwidth=4)

    for i, traj in enumerate(trajectories):
        step = max(1, len(traj) // 500)
        traj_s = traj[::step]
        ax.plot(traj_s[:, 0], traj_s[:, 2], color=run_colors[i], alpha=0.8, linewidth=1.2, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 2], color='green', s=40, zorder=4, alpha=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 2], color=run_colors[i], marker='*', s=100, zorder=5)

    for idx, ne in enumerate(nash_equilibria):
        ax.scatter(ne[0], ne[1], color='red', s=150, zorder=6, marker='X',
                   label='Nash Eq.' if idx == 0 else None, edgecolors='black', linewidths=1.0)

    rep_line = mlines.Line2D([], [], color='lightgray', linewidth=1.5)
    ana_line = mlines.Line2D([], [], color=field_color, alpha=0.8, linewidth=2)
    traj_line = mlines.Line2D([], [], color=run_colors[len(trajectories) // 2], linewidth=1.5)
    start_dot = ax.scatter([], [], color='green', s=40)
    end_dot = ax.scatter([], [], color='gray', marker='*', s=100)
    handles, labels = ax.get_legend_handles_labels()
    handles = [rep_line, ana_line, traj_line, start_dot, end_dot] + handles
    labels = ['Replicator Dynamics', f'Analytical {agent_name}', 'Trajectories', 'Start', 'End'] + labels
    ax.legend(handles=handles, labels=labels, fontsize=9, loc='center left', bbox_to_anchor=(1.05, 0.5))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("P1: P(action 0)", fontsize=12)
    ax.set_ylabel("P2: P(action 0)", fontsize=12)
    ax.set_title(f"{game.name} — {agent_name}", fontsize=14, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_3action_combined(game, agent_name, trajectories, nash_equilibria, tau, kappa):
    grid = 40
    X_vals = np.linspace(-0.05, 1.05, grid)
    Y_vals = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid)
    X, Y = np.meshgrid(X_vals, Y_vals)

    dX_rep, dY_rep = np.zeros(X.shape), np.zeros(Y.shape)
    dX_ana, dY_ana = np.zeros(X.shape), np.zeros(Y.shape)

    for i in range(grid):
        for j in range(grid):
            xc, yc = X[i, j], Y[i, j]
            p2_coord = yc / (np.sqrt(3) / 2)
            p1_coord = xc - 0.5 * p2_coord
            p0_coord = 1.0 - p1_coord - p2_coord
            if p0_coord >= -0.01 and p1_coord >= -0.01 and p2_coord >= -0.01:
                p = np.clip([p0_coord, p1_coord, p2_coord], 0.001, 1.0)
                p = p / p.sum()
                ent_p = np.sum(p * np.log(p))

                E_rep = game.payoffs_p1 @ p
                dp_rep = p * (E_rep - p @ E_rep)
                dX_rep[i, j] = dp_rep[1] + 0.5 * dp_rep[2]
                dY_rep[i, j] = (np.sqrt(3) / 2) * dp_rep[2]

                E_ana = E_rep if agent_name == "Boltzmann" else \
                    compute_lenient_expected_rewards(game.payoffs_p1, p, kappa)
                dp_ana = p * ((E_ana - p @ E_ana) / tau + ent_p - np.log(p))
                dX_ana[i, j] = dp_ana[1] + 0.5 * dp_ana[2]
                dY_ana[i, j] = (np.sqrt(3) / 2) * dp_ana[2]

    mask = (Y < 0) | (X < Y / np.sqrt(3)) | (X > 1 - Y / np.sqrt(3))
    dX_rep = np.ma.masked_where(mask, dX_rep)
    dY_rep = np.ma.masked_where(mask, dY_rep)
    dX_ana = np.ma.masked_where(mask, dX_ana)
    dY_ana = np.ma.masked_where(mask, dY_ana)

    is_lenient = (agent_name == "Lenient Boltzmann")
    field_color = 'darkorange' if is_lenient else 'steelblue'
    run_colors = (plt.cm.Oranges if is_lenient else plt.cm.Blues)(
        np.linspace(0.45, 0.9, max(len(trajectories), 1)))

    def ternary_to_cartesian(p_arr):
        return p_arr[:, 1] + 0.5 * p_arr[:, 2], (np.sqrt(3) / 2) * p_arr[:, 2]

    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, player_idx, title in zip(axes, [0, 1], ["Player 1", "Player 2"]):
        ax.plot(triangle_x, triangle_y, color='black', linewidth=1.5, zorder=1)
        ax.streamplot(X, Y, dX_rep, dY_rep, color='lightgray', linewidth=0.9, density=1.2, arrowstyle='->')
        skip = (slice(None, None, 2), slice(None, None, 2))
        ax.quiver(X[skip], Y[skip], dX_ana[skip], dY_ana[skip],
                  color=field_color, alpha=0.5, width=0.003, headwidth=4)

        start_col = player_idx * 3
        for i, traj in enumerate(trajectories):
            p = traj[:, start_col:start_col + 3]
            # Subsample to at most 200 points
            step = max(1, len(p) // 200)
            p_s = p[::step]
            x, y = ternary_to_cartesian(p_s)
            x0, y0 = ternary_to_cartesian(p[:1])
            xf, yf = ternary_to_cartesian(p[-1:])
            ax.plot(x, y, color=run_colors[i], alpha=0.8, linewidth=1.2, zorder=3)
            ax.scatter(x0[0], y0[0], color='green', s=40, zorder=4, alpha=0.8)
            ax.scatter(xf[0], yf[0], color=run_colors[i], marker='*', s=100, zorder=5)

        if nash_equilibria:
            for idx, ne in enumerate(nash_equilibria):
                ne_x, ne_y = ternary_to_cartesian(np.array([ne]))
                ax.scatter(ne_x[0], ne_y[0], color='red', s=150, zorder=6, marker='X',
                           label='Nash Eq.' if idx == 0 else None, edgecolors='black', linewidths=1.0)

        ax.text(-0.02, -0.05, "Action 0", fontsize=12, ha='right', va='top')
        ax.text(1.02, -0.05, "Action 1", fontsize=12, ha='left', va='top')
        ax.text(0.5, np.sqrt(3) / 2 + 0.05, "Action 2", fontsize=12, ha='center', va='bottom')
        ax.set_title(f"{game.name} — {agent_name}\n{title}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.25)
        ax.axis('off')
        ax.set_aspect('equal')

        if player_idx == 1:
            rep_line = mlines.Line2D([], [], color='lightgray', linewidth=1.5)
            ana_line = mlines.Line2D([], [], color=field_color, alpha=0.8, linewidth=2)
            traj_line = mlines.Line2D([], [], color=run_colors[len(trajectories) // 2], linewidth=1.5)
            start_dot = ax.scatter([], [], color='green', s=40)
            end_dot = ax.scatter([], [], color='gray', marker='*', s=100)
            handles, labels = ax.get_legend_handles_labels()
            handles = [rep_line, ana_line, traj_line, start_dot, end_dot] + handles
            labels = ['Replicator Dynamics', f'Analytical {agent_name}', 'Trajectories', 'Start', 'End'] + labels
            ax.legend(handles=handles, labels=labels, fontsize=9, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.savefig(f"{game.name}_{agent_name.replace(' ', '_')}_ternary.png", dpi=150, bbox_inches='tight')
    plt.close()
