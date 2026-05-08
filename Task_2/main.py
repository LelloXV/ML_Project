from games import stag_hunt, subsidy, prisoners_dilemma, biased_rps
from simulate import run_experiment
from plotting import plot_trajectories, plot_combined_field_and_trajectories
from agents import EpsilonGreedyAgent, BoltzmannAgent, LenientBoltzmannAgent

games = [stag_hunt, subsidy, prisoners_dilemma, biased_rps]

epsilon_params = {"alpha": 0.1, "epsilon": 0.1}
boltzmann_params = {"alpha": 0.1, "tau": 0.3}
lenient_params = {"alpha": 0.1, "tau": 0.3, "kappa": 5}

algorithms = [
    (EpsilonGreedyAgent, epsilon_params, "Epsilon-Greedy"),
    (BoltzmannAgent, boltzmann_params, "Boltzmann"),
    (LenientBoltzmannAgent, lenient_params, "Lenient Boltzmann"),
]

nash_equilibria = {
    "stag_hunt":          [(1, 1), (0, 0), (2/3, 2/3)],
    "subsidy":            [(1, 1), (0, 0), (10/11, 10/11)],
    "prisoners_dilemma":  [(0, 0)],
    "biased_rps":         [(10/16, 5/16, 1/16)],
}

# 2-action games converge fast; biased_rps needs more steps to show
# the mixed-NE cycling behaviour.
n_episodes_per_game = {
    "stag_hunt":         10000,
    "subsidy":           10000,
    "prisoners_dilemma": 10000,
    "biased_rps":        50000,
}

for game in games:
    for agent_class, agent_params, agent_name in algorithms:
        print(f"Running {game.name} with {agent_name}...")

        episodes = n_episodes_per_game[game.name]
        if "kappa" in agent_params:
            episodes = episodes * agent_params["kappa"]

        trajectories = run_experiment(
            game, agent_class, agent_params,
            n_episodes=n_episodes_per_game[game.name], n_runs=5,
        )
        if agent_name == "Epsilon-Greedy":
            plot_trajectories(trajectories, game, agent_name, nash_equilibria[game.name])
        else:
            plot_combined_field_and_trajectories(
                game, agent_name, trajectories, nash_equilibria[game.name]
            )