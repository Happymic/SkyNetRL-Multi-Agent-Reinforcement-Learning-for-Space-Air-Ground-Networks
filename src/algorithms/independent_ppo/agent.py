"""
Independent PPO Agent Implementation
Each agent learns independently using Proximal Policy Optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

from .networks import PPOActor, PPOCritic, EnhancedPPOActor, SharedPPONetwork


class IndependentPPOAgent:
    """Independent PPO Agent for multi-agent environments"""
    
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, config: Dict):
        """
        Initialize Independent PPO Agent
        
        Args:
            agent_id: Unique agent identifier
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Determine agent type
        self.agent_type = self._determine_agent_type(agent_id, config)
        
        # Network configuration
        network_config = {
            'hidden_dim': config.get('hidden_dim', 256),
            'dropout': config.get('dropout', 0.1)
        }
        
        # Create networks
        use_shared_network = config.get('use_shared_network', False)
        use_enhanced_network = config.get('use_enhanced_network', True)
        
        if use_shared_network:
            self.network = SharedPPONetwork(obs_dim, action_dim, network_config).to(self.device)
            self.actor = None
            self.critic = None
        else:
            if use_enhanced_network:
                self.actor = EnhancedPPOActor(obs_dim, action_dim, network_config).to(self.device)
            else:
                self.actor = PPOActor(obs_dim, action_dim, network_config).to(self.device)
            self.critic = PPOCritic(obs_dim, network_config).to(self.device)
            self.network = None
        
        # Optimizers
        if use_shared_network:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=config.get('lr', 3e-4),
                eps=config.get('adam_eps', 1e-5),
                weight_decay=config.get('weight_decay', 1e-5)
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=config.get('actor_lr', 3e-4),
                eps=config.get('adam_eps', 1e-5),
                weight_decay=config.get('weight_decay', 1e-5)
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(),
                lr=config.get('critic_lr', 1e-3),
                eps=config.get('adam_eps', 1e-5),
                weight_decay=config.get('weight_decay', 1e-5)
            )
        
        # PPO hyperparameters
        self.clip_range = config.get('clip_range', 0.2)
        self.value_clip_range = config.get('value_clip_range', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        
        # Training parameters
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        self.normalize_advantages = config.get('normalize_advantages', True)
        
        # Experience buffer
        self.buffer_size = config.get('buffer_size', 2048)
        self.buffer = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
            'advantages': [],
            'returns': []
        }
        
        # Training statistics
        self.training_step = 0
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _determine_agent_type(self, agent_id: int, config: Dict) -> str:
        """Determine agent type based on ID and configuration"""
        num_satellites = config.get('num_satellites', 1)
        num_uavs = config.get('num_uavs', 3)
        
        if agent_id < num_satellites:
            return 'satellite'
        elif agent_id < num_satellites + num_uavs:
            return 'uav'
        else:
            return 'ground_station'
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Generate action using PPO policy
        
        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Action array
            log_prob: Log probability of action
            value: State value estimate
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.network is not None:
                # Shared network
                if deterministic:
                    mean, std, value = self.network(obs_tensor)
                    action = mean
                    log_prob = torch.zeros(1)
                else:
                    action, value, log_prob = self.network.get_action_value_and_logprob(obs_tensor)
            else:
                # Separate actor-critic
                value = self.critic(obs_tensor)
                if deterministic:
                    mean, std = self.actor(obs_tensor)
                    action = mean
                    log_prob = torch.zeros(1)
                else:
                    action, log_prob = self.actor.get_action_and_logprob(obs_tensor)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0, 0]
    
    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                        done: bool, log_prob: float, value: float):
        """Store transition in experience buffer"""
        self.buffer['observations'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
    
    def compute_gae(self, next_value: float = 0.0):
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        # Add next value for bootstrap
        values = np.append(values, next_value)
        
        advantages = []
        gae = 0
        
        # Compute advantages using GAE
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_values = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = np.array(advantages) + values[:-1]
        
        self.buffer['advantages'] = advantages
        self.buffer['returns'] = returns.tolist()
    
    def update(self) -> Dict:
        """
        Update PPO networks using collected experience
        
        Returns:
            training_info: Dictionary of training statistics
        """
        if len(self.buffer['observations']) < self.batch_size:
            return {}
        
        self.training_step += 1
        
        # Convert buffer to tensors
        obs_tensor = torch.FloatTensor(np.array(self.buffer['observations'])).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        returns_tensor = torch.FloatTensor(self.buffer['returns']).to(self.device)
        advantages_tensor = torch.FloatTensor(self.buffer['advantages']).to(self.device)
        old_values_tensor = torch.FloatTensor(self.buffer['values']).to(self.device)
        
        # Normalize advantages
        if self.normalize_advantages:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        total_clip_fraction = 0
        
        for epoch in range(self.ppo_epochs):
            # Create random batches
            indices = torch.randperm(len(obs_tensor))
            
            for start_idx in range(0, len(obs_tensor), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(obs_tensor))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                
                # Forward pass
                if self.network is not None:
                    # Shared network
                    log_probs, entropy, values = self.network.evaluate_actions(batch_obs, batch_actions)
                else:
                    # Separate networks
                    log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                    values = self.critic(batch_obs).squeeze()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                if self.value_clip_range > 0:
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values, -self.value_clip_range, self.value_clip_range
                    )
                    value_loss1 = F.mse_loss(values, batch_returns)
                    value_loss2 = F.mse_loss(values_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update networks
                if self.network is not None:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else:
                    # Update actor
                    actor_loss = policy_loss + self.entropy_coef * entropy_loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    
                    # Update critic
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                
                # Compute statistics
                with torch.no_grad():
                    # KL divergence (approximate)
                    kl_div = 0.5 * torch.mean((log_probs - batch_old_log_probs) ** 2)
                    
                    # Clip fraction
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > self.clip_range).float()
                    )
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_kl_div += kl_div.item()
                    total_clip_fraction += clip_fraction.item()
        
        # Average losses over all updates
        num_updates = self.ppo_epochs * (len(obs_tensor) // self.batch_size + 1)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_kl_div = total_kl_div / num_updates
        avg_clip_fraction = total_clip_fraction / num_updates
        
        # Record statistics
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy_loss'].append(avg_entropy_loss)
        self.training_stats['kl_divergence'].append(avg_kl_div)
        self.training_stats['clip_fraction'].append(avg_clip_fraction)
        
        # Clear buffer
        self.clear_buffer()
        
        training_info = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + avg_value_loss + avg_entropy_loss,
            'kl_divergence': avg_kl_div,
            'clip_fraction': avg_clip_fraction,
            'training_step': self.training_step
        }
        
        return training_info
    
    def clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key] = []
    
    def is_buffer_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer['observations']) >= self.buffer_size
    
    def save(self, filepath: str):
        """Save agent networks and statistics"""
        if self.network is not None:
            state_dict = {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
        else:
            state_dict = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
            }
        
        state_dict.update({
            'training_stats': self.training_stats,
            'training_step': self.training_step,
            'agent_type': self.agent_type,
            'config': self.config
        })
        
        torch.save(state_dict, filepath)
    
    def load(self, filepath: str):
        """Load agent networks and statistics"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.network is not None:
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_stats = checkpoint['training_stats']
        self.training_step = checkpoint['training_step']
    
    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        if self.network is not None:
            self.network.eval()
        else:
            self.actor.eval()
            self.critic.eval()
    
    def set_train_mode(self):
        """Set networks to training mode"""
        if self.network is not None:
            self.network.train()
        else:
            self.actor.train()
            self.critic.train()
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics for analysis"""
        if not self.training_stats['policy_loss']:
            return {}
        
        return {
            'avg_policy_loss': np.mean(self.training_stats['policy_loss'][-100:]),
            'avg_value_loss': np.mean(self.training_stats['value_loss'][-100:]),
            'avg_entropy_loss': np.mean(self.training_stats['entropy_loss'][-100:]),
            'avg_kl_divergence': np.mean(self.training_stats['kl_divergence'][-100:]),
            'avg_clip_fraction': np.mean(self.training_stats['clip_fraction'][-100:]),
            'total_training_steps': self.training_step
        }