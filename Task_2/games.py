import numpy as np

class Game:
    def __init__(self, name, payoffs_p1, payoffs_p2):
        self.name = name
        self.payoffs_p1 = payoffs_p1
        self.payoffs_p2 = payoffs_p2

    def get_rewards(self, action_p1, action_p2):
        reward_p1 = self.payoffs_p1[action_p1, action_p2]
        reward_p2 = self.payoffs_p2[action_p1, action_p2]
        return reward_p1, reward_p2

    @property
    def n_actions_p1(self):
        return self.payoffs_p1.shape[0]

    @property
    def n_actions_p2(self):
        return self.payoffs_p1.shape[1]


stag_hunt = Game("stag_hunt",
    payoffs_p1 = np.array([[1, 0], [2/3, 2/3]]),
    payoffs_p2 = np.array([[1, 2/3], [0, 2/3]])
)

subsidy = Game("subsidy",
    payoffs_p1 = np.array([[12, 0], [11, 10]]),
    payoffs_p2 = np.array([[12, 11], [0, 10]])
)

# action 0 = Defect (dominant strategy), action 1 = Cooperate
# Nash equilibrium = (0,0) in payoff table = both defect = (1,1) on the plot
# P(action 0) -> 1 means both agents always defect
prisoners_dilemma = Game("prisoners_dilemma",
    payoffs_p1 = np.array([[0, -3], [-4, -1]]),
    payoffs_p2 = np.array([[0, -4], [-3, -1]])
)

biased_rps = Game("biased_rps",
    payoffs_p1 = np.array([[0, -0.05, 0.25], [0.05, 0, -0.5], [-0.25, 0.5, 0]]),
    payoffs_p2 = np.array([[0, 0.05, -0.25], [-0.05, 0, 0.5], [0.25, -0.5, 0]])
)
