import numpy as np

def ema_filter(data, alpha=0.01):
    """
    Applies an Exponential Moving Average (EMA) to smooth the data.
    Lower alpha = smoother curves (more memory of the past).
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

def run_experiment(game, agent_class, agent_params, n_episodes, n_runs, ema_alpha=0.001):
    all_trajectories = []

    for run in range(n_runs):
        agent1 = agent_class(game.n_actions_p1, **agent_params)
        agent2 = agent_class(game.n_actions_p2, **agent_params)

        policy_p1 = np.empty((n_episodes, game.n_actions_p1))
        policy_p2 = np.empty((n_episodes, game.n_actions_p2))

        for episode in range(n_episodes):
            a1 = agent1.select_action()
            a2 = agent2.select_action()

            r1, r2 = game.get_rewards(a1, a2)

            agent1.update(a1, r1)
            agent2.update(a2, r2)

            policy_p1[episode] = agent1.get_policy()
            policy_p2[episode] = agent2.get_policy()

        # Apply the EMA filter
        smooth_p1 = np.stack(
            [ema_filter(policy_p1[:, a], alpha=ema_alpha) for a in range(game.n_actions_p1)],
            axis=1,
        )
        smooth_p2 = np.stack(
            [ema_filter(policy_p2[:, a], alpha=ema_alpha) for a in range(game.n_actions_p2)],
            axis=1,
        )

        all_trajectories.append(np.concatenate([smooth_p1, smooth_p2], axis=1))

    return np.array(all_trajectories)