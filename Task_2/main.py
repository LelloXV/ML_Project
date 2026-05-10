from games import stag_hunt, subsidy, prisoners_dilemma, biased_rps
from simulate import run_experiment
from plotting import plot_trajectories, plot_combined_field_and_trajectories
from agents import EpsilonGreedyAgent, BoltzmannAgent, LenientBoltzmannAgent

games = [stag_hunt, subsidy, prisoners_dilemma, biased_rps]

epsilon_params = {"alpha": 0.1, "epsilon": 0.1}

boltzmann_params = {
    "stag_hunt":         {"alpha": 0.1, "tau": 1.0},
    "subsidy":           {"alpha": 0.1, "tau": 5.0},
    "prisoners_dilemma": {"alpha": 0.1, "tau": 0.1},
    "biased_rps":        {"alpha": 0.1, "tau": 0.5},
}
lenient_params = {
    "stag_hunt":         {"alpha": 0.1, "tau": 1.0, "kappa": 25},
    "subsidy":           {"alpha": 0.1, "tau": 5.0, "kappa": 40},
    "prisoners_dilemma": {"alpha": 0.1, "tau": 0.1, "kappa": 10},
    "biased_rps":        {"alpha": 0.1, "tau": 0.5, "kappa": 10},
}

algorithms = [
    (EpsilonGreedyAgent, epsilon_params, "Epsilon-Greedy"),
    (BoltzmannAgent, boltzmann_params, "Boltzmann"),
    (LenientBoltzmannAgent, lenient_params, "Lenient Boltzmann"),
]

nash_equilibria = {
    "stag_hunt":          [(1, 1), (0, 0), (2/3, 2/3)],
    "subsidy":            [(1, 1), (0, 0), (10/11, 10/11)],
    "prisoners_dilemma":  [(1, 1)],  # action 0 = Defect, so (1,1) = both defect
    "biased_rps":         [(10/16, 5/16, 1/16)],
}

n_episodes_per_game = {
    "stag_hunt":         10000,
    "subsidy":           100000,  # increased for lenient convergence
    "prisoners_dilemma": 10000,
    "biased_rps":        50000,
}

n_runs_per_game = {
    "stag_hunt":         5,
    "subsidy":           5,
    "prisoners_dilemma": 5,
    "biased_rps":        3,
}

ema_alpha_per_game = {
    "stag_hunt":         0.01,
    "subsidy":           0.01,
    "prisoners_dilemma": 0.05,
    "biased_rps":        0.05,
}

for game in games:
    for agent_class, agent_params_dict, agent_name in algorithms:
        if isinstance(agent_params_dict, dict) and game.name in agent_params_dict:
            agent_params = agent_params_dict[game.name]
        else:
            agent_params = agent_params_dict
        print(f"Running {game.name} with {agent_name}...")

        episodes = n_episodes_per_game[game.name]
        if "kappa" in agent_params:
            episodes = episodes * agent_params["kappa"]

        trajectories = run_experiment(
            game, agent_class, agent_params,
            n_episodes=episodes,
            n_runs=n_runs_per_game[game.name],
            ema_alpha=ema_alpha_per_game[game.name],
        )

        if agent_name == "Epsilon-Greedy":
            plot_trajectories(trajectories, game, agent_name, nash_equilibria[game.name])
        else:
            tau = agent_params.get("tau", 0.3)
            kappa = agent_params.get("kappa", 5)
            print(f"  -> tau={tau}, kappa={kappa}")  # debug: verify correct params
            plot_combined_field_and_trajectories(
                game, agent_name, trajectories, nash_equilibria[game.name],
                tau=tau, kappa=kappa
            )
