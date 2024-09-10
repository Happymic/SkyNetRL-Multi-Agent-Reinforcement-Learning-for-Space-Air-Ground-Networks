import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.sag_network import SAGNetwork


class SAGAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        self.network = SAGNetwork(obs_dim, action_dim, config.hidden_dim)
        self.target_network = SAGNetwork(obs_dim, action_dim, config.hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def action(self, obs, adj, epsilon=0.05):
        if np.random.random() < epsilon:
            return np.random.uniform(-1, 1, self.action_dim)

        with torch.no_grad():
            q_values = self.network(obs, adj)
            return q_values.argmax(dim=-1).cpu().numpy()

    def update(self, batch):
        obs, adj, actions, rewards, next_obs, next_adj, dones = batch

        q_values = self.network(obs, adj).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_network(next_obs, next_adj).max(dim=-1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.network.state_dict())