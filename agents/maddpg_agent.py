import torch
import torch.nn.functional as F
import numpy as np
from agents.sag_network import ActorNetwork, CriticNetwork
import gc


class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, config, agent_id):
        self.config = config
        self.device = config.device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_id = agent_id

        # Initialize networks with memory optimization
        torch.backends.cudnn.benchmark = True
        self.actor = ActorNetwork(config.individual_obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic = CriticNetwork(config.individual_obs_dim * config.num_agents,
                                    config.action_dim * config.num_agents,
                                    config.hidden_dim).to(self.device)
        self.target_actor = ActorNetwork(config.individual_obs_dim, config.action_dim, config.hidden_dim).to(
            self.device)
        self.target_critic = CriticNetwork(config.individual_obs_dim * config.num_agents,
                                           config.action_dim * config.num_agents,
                                           config.hidden_dim).to(self.device)

        # Initialize networks and copy weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimize memory allocation for optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Set up gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        self.noise = config.exploration_noise
        self.gradient_steps = 0

    def act(self, obs, add_noise=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs).squeeze().cpu().numpy()
            if add_noise:
                action += self.noise * np.random.randn(self.config.action_dim)
            action = np.clip(action, -1, 1)
            del obs
            return action

    def update(self, sample, other_agents):
        # Convert to tensors and move to device
        obs, actions, rewards, next_obs, dones = [
            torch.FloatTensor(x).to(self.device) for x in sample
        ]

        # Gradient accumulation setup
        self.gradient_steps += 1
        accumulate_gradients = (self.gradient_steps % self.config.gradient_accumulation_steps != 0)

        # Critic update
        self.critic_optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Calculate critic loss
            with torch.no_grad():
                all_next_actions = []
                for i, agent in enumerate([self] + other_agents):
                    next_action_i = agent.target_actor(next_obs[:, i])
                    all_next_actions.append(next_action_i)
                all_next_actions = torch.cat(all_next_actions, dim=1)

                target_q = self.target_critic(next_obs.view(next_obs.size(0), -1), all_next_actions)
                target_value = rewards[:, self.agent_id].unsqueeze(1) + self.config.gamma * target_q * (1 - dones)

            q_value = self.critic(obs.view(obs.size(0), -1), actions.view(actions.size(0), -1))
            critic_loss = F.mse_loss(q_value, target_value)
            if not accumulate_gradients:
                critic_loss = critic_loss / self.config.gradient_accumulation_steps

        # Optimize critic
        self.scaler.scale(critic_loss).backward()
        if not accumulate_gradients:
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.scaler.step(self.critic_optimizer)

        # Actor update
        if not accumulate_gradients:
            self.actor_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                policy_actions = self.actor(obs[:, self.agent_id])
                all_actions = actions.clone()
                all_actions[:, self.agent_id] = policy_actions
                actor_loss = -self.critic(obs.view(obs.size(0), -1),
                                          all_actions.view(all_actions.size(0), -1)).mean()
                actor_loss = actor_loss / self.config.gradient_accumulation_steps

            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()

            # Memory cleanup
            if self.gradient_steps % self.config.memory_cleanup_freq == 0:
                self.memory_cleanup()

        # Clean up tensors
        del obs, actions, rewards, next_obs, dones, q_value, target_value
        if not accumulate_gradients:
            del all_next_actions

        return critic_loss.item(), actor_loss.item() if not accumulate_gradients else 0.0

    def memory_cleanup(self):
        """Perform memory cleanup operations"""
        torch.cuda.empty_cache()
        gc.collect()

    def update_targets(self):
        if self.gradient_steps % self.config.gradient_accumulation_steps == 0:
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target, source):
        with torch.no_grad():
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
            'scaler': self.scaler.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])