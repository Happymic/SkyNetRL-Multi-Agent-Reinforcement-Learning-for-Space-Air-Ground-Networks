import numpy as np

class OUNoise:
    def __init__(self, action_dimension, num_agents, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.num_agents = num_agents
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.num_agents * self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.num_agents * self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state.reshape(self.num_agents, self.action_dimension) * self.scale