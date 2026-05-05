from agents import Agent
import numpy as np


def run_experiment(game, agent_class, agent_params, n_episodes, n_runs):
    all_trajectories = []

    for run in range(n_runs):
        agent1 = agent_class(game.n_actions_p1, **agent_params)
        agent2 = agent_class(game.n_actions_p2, **agent_params)

        trajectory = []

        # Track counts for ALL actions to calculate time-average
        action_counts_p1 = np.zeros(game.n_actions_p1)
        action_counts_p2 = np.zeros(game.n_actions_p2)

        for episode in range(n_episodes):
            a1 = agent1.select_action()
            a2 = agent2.select_action()

            r1, r2 = game.get_rewards(a1, a2)

            agent1.update(a1, r1)
            agent2.update(a2, r2)

            # Record the time-averaged policy
            action_counts_p1[a1] += 1
            action_counts_p2[a2] += 1

            avg_policy_p1 = action_counts_p1 / (episode + 1)
            avg_policy_p2 = action_counts_p2 / (episode + 1)

            trajectory.append(np.concatenate([avg_policy_p1, avg_policy_p2]))

        all_trajectories.append(trajectory)

    return np.array(all_trajectories)