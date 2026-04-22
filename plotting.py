import matplotlib.pyplot as plt

def plot_trajectories(all_trajectories, game, agent_name, nash_equilibria):
    fig, ax = plt.subplots()

    # 1. plot each trajectory
    for trajectory in all_trajectories:
        p1_probs = trajectory[:, 0]  # player 1's prob of action 0 over time
        p2_probs = trajectory[:, 1]  # player 2's prob of action 0 over time
        ax.plot(p1_probs, p2_probs, color="blue", alpha =0.3)  # what color/alpha makes sense?

    # 2. mark Nash Equilibria
    for i, ne in enumerate(nash_equilibria):
        label = "Nash Eq." if i == 0 else None
        ax.scatter(ne[0], ne[1], color="red", s=100, zorder=5, label=label)
    ax.legend()

    # 3. labels and formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P1: P(action 0)")
    ax.set_ylabel("P2: P(action 0)")
    ax.set_title(f"{game.name} - {agent_name}")

    plt.show()