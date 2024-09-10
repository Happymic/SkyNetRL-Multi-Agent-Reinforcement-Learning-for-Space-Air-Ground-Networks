import numpy as np
import random

class ReplayBuffer:
    def __init__(self, buffer_size, num_agents, obs_dim, action_dim):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pointer = 0
        self.size = 0

        self.observations = np.zeros((buffer_size, num_agents, obs_dim))
        self.actions = np.zeros((buffer_size, num_agents, action_dim))
        self.rewards = np.zeros((buffer_size, num_agents))
        self.next_observations = np.zeros((buffer_size, num_agents, obs_dim))
        self.dones = np.zeros((buffer_size, 1))

    def add(self, obs, actions, rewards, next_obs, done):
        self.observations[self.pointer] = obs
        self.actions[self.pointer] = actions
        self.rewards[self.pointer] = rewards
        self.next_observations[self.pointer] = next_obs
        self.dones[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size