SHOOT_ACTION = 4  # check the action space in the PettingZoo docs

class StayAndShootAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.t = 0

    def act(self, observation):
        return SHOOT_ACTION