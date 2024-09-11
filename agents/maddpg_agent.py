import torch
import torch.nn.functional as F
import numpy as np
from agents.sag_network import ActorNetwork, CriticNetwork

class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, config, agent_id):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_id = agent_id

        self.agent_id = agent_id

        self.actor = ActorNetwork(config.individual_obs_dim, config.action_dim, config.hidden_dim)
        self.critic = CriticNetwork(config.individual_obs_dim * config.num_agents,
                                    config.action_dim * config.num_agents,
                                    config.hidden_dim)
        self.target_actor = ActorNetwork(config.individual_obs_dim, config.action_dim, config.hidden_dim)
        self.target_critic = CriticNetwork(config.individual_obs_dim * config.num_agents,
                                           config.action_dim * config.num_agents,
                                           config.hidden_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.noise = config.exploration_noise

    def act(self, obs, add_noise=True):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs).squeeze().detach().numpy()
        if add_noise:
            action += self.noise * np.random.randn(self.config.action_dim)
        return np.clip(action, -1, 1)

    def update(self, sample, other_agents):
        obs, actions, rewards, next_obs, dones = sample

        # Convert numpy arrays to tensors
        obs = torch.FloatTensor(obs)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)

        # Update critic
        all_next_actions = []
        for i, agent in enumerate([self] + other_agents):
            next_obs_i = next_obs[:, i]
            next_action_i = agent.target_actor(next_obs_i)
            all_next_actions.append(next_action_i)
        all_next_actions = torch.cat(all_next_actions, dim=1)

        target_q = self.target_critic(next_obs.view(next_obs.size(0), -1), all_next_actions)
        target_value = rewards[:, self.agent_id].unsqueeze(1) + self.config.gamma * target_q * (1 - dones)

        q_value = self.critic(obs.view(obs.size(0), -1), actions.view(actions.size(0), -1))
        critic_loss = F.mse_loss(q_value, target_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Update actor
        policy_actions = self.actor(obs[:, self.agent_id])
        all_actions = actions.clone()
        all_actions[:, self.agent_id] = policy_actions
        actor_loss = -self.critic(obs.view(obs.size(0), -1), all_actions.view(all_actions.size(0), -1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def update_targets(self):
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    def decay_noise(self):
        self.noise *= self.config.exploration_decay


    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])