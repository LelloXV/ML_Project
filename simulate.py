from agents import Agent
import numpy as np


def run_experiment(game, agent_class, agent_params, n_episodes, n_runs):
    all_trajectories = []  # shape will be (n_runs, n_episodes, 2)
                           # last dim = policy of p1 and p2

    for run in range(n_runs):
        # 1. create two fresh agents
        agent1 = agent_class(game.n_actions_p1, **agent_params)
        agent2 = agent_class(game.n_actions_p2, **agent_params)

        trajectory = []  # stores (prob_action0_p1, prob_action0_p2) per step

        action0_counts = [0, 0]

        for episode in range(n_episodes):
            # 2a. select actions
            a1 = agent1.select_action()
            a2 = agent2.select_action()

            # 2b. get rewards
            r1, r2 = game.get_rewards(a1, a2)

            # 2c. update agents
            agent1.update(a1, r1)
            agent2.update(a2, r2)

            # 2d. record time-averaged policy (fraction of times action 0 was chosen)
            #action0_counts[0] += (a1 == 0)
            #action0_counts[1] += (a2 == 0)
            #policy_p1 = action0_counts[0] / (episode + 1)
            #policy_p2 = action0_counts[1] / (episode + 1)
            #trajectory.append((policy_p1, policy_p2))
            policy_p1 = agent1.get_policy()
            policy_p2 = agent2.get_policy()
            trajectory.append(np.concatenate([policy_p1, policy_p2]))

        all_trajectories.append(trajectory)

    return np.array(all_trajectories)  # shape: (n_runs, n_episodes, 2)