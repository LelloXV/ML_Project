"""Heuristic: stay roughly in the centre, fire whenever an arrow is ready.
Useful as a non-trivial floor — your trained agent must beat this."""
SHOOT_ACTION = 5  # check the action space in the PettingZoo docs

class ShootDiagonalAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.t = 0

    def act(self, observation):
        self.t += 1
        # alternate "rotate slightly" and "shoot"
        return SHOOT_ACTION if self.t % 3 == 0 else 1