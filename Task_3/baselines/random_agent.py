"""Random baseline. Use this as a sanity floor for tournament scores."""
import numpy as np


class RandomAgent:
    def __init__(self, action_space, seed=None):
        self.rng = np.random.default_rng(seed)
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()