import numpy as np

class Agent:
    def __init__(self, n_actions, alpha):
        self.n_actions = n_actions
        self.alpha = alpha
        self.Q = np.random.uniform(-0.3, 0.3, n_actions)

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        self.Q[action] = self.Q[action] + self.alpha*(reward - self.Q[action])

    def get_policy(self):
        raise NotImplementedError

class EpsilonGreedyAgent(Agent):
    def __init__(self, n_actions, alpha, epsilon):
        super().__init__(n_actions, alpha)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions) # explore
        else:
            return np.argmax(self.Q) # exploit

    def get_policy(self):
        probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
        probs[np.argmax(self.Q)] += (1 - self.epsilon)
        return probs


class BoltzmannAgent(Agent):
    def __init__(self, n_actions, alpha, tau):
        super().__init__(n_actions, alpha)
        self.tau = tau
        # Sample initial policy uniformly from the simplex (Dirichlet) and
        # back-project to Q space: if pi = softmax(Q/tau) then Q = tau*log(pi).
        # The additive constant cancels in softmax, so this is exact.
        pi0 = np.random.dirichlet(np.ones(n_actions))
        self.Q = tau * np.log(pi0)

    def select_action(self):
        #probs = np.exp(self.Q/self.tau)/np.sum(np.exp(self.Q/self.tau))
        #return np.random.choice(self.n_actions, p=probs) # selects action a_i with probability got from Boltzmann
        probs = self.get_policy()
        return np.random.choice(self.n_actions, p=probs)

    def get_policy(self):
        #probs = np.exp(self.Q / self.tau) / np.sum(np.exp(self.Q / self.tau))
        #return probs
        z = self.Q / self.tau
        z = z - np.max(z)  # log-sum-exp stabilisation
        e = np.exp(z)
        return e / np.sum(e)

class LenientBoltzmannAgent(BoltzmannAgent):
    def __init__(self, n_actions, alpha, tau, kappa):
        super().__init__(n_actions, alpha, tau)
        self.kappa = kappa
        self.reward_buffer = [[] for _ in range(n_actions)] # one list per action, so a list of n_actions lists

    def update(self, action, reward):
        self.reward_buffer[action].append(reward)
        if len(self.reward_buffer[action]) >= self.kappa:
            best_reward = np.max(self.reward_buffer[action])  # take the max of the buffer
            self.Q[action] = self.Q[action] + self.alpha*(best_reward - self.Q[action]) # standard Q update using best_reward
            self.reward_buffer[action] = []  # reset the buffer