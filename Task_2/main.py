from games import stag_hunt, subsidy, prisoners_dilemma, biased_rps
from simulate import run_experiment
from plotting import plot_trajectories, plot_directional_field
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
    "stag_hunt":          [(1,1), (0,0), (2/3, 2/3)],  # pure + mixed NE
    "subsidy":            [(1, 1), (0, 0), (10/11, 10/11)],
    "prisoners_dilemma":  [(0, 0)],
    "biased_rps":         [(10/16, 5/16, 1/16)]
    #"biased_rps":         []
}

for game in games:
    # 1. Plot the Analytical Directional Field (Only for 2-action games)
    if game.n_actions_p1 == 2:
        # Note: adjust 'game.payoff_p1' if your attribute names in games.py are different
        plot_directional_field(
            game_name=game.name,
            payoffs_P1=game.payoffs_p1,
            payoffs_P2=game.payoffs_p2,
            nash_equilibria=nash_equilibria[game.name]
        )

    # 2. Run Experiments and Plot Empirical Trajectories
    for agent_class, agent_params, agent_name in algorithms:
        print(f"Running {game.name} with {agent_name}...") # helpful for tracking progress
        trajectories = run_experiment(game, agent_class, agent_params, n_episodes=10000, n_runs=20)
        plot_trajectories(trajectories, game, agent_name, nash_equilibria[game.name])