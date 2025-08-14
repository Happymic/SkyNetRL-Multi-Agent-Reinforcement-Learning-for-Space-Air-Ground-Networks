"""
Attention-Enhanced MADDPG Agent
Complete implementation with proper attention mechanisms
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
import copy

from .networks import AttentionEnhancedActor, AttentionEnhancedCritic, AttentionAnalyzer
from .attention_modules import AttentionRegularizer


class AEMADDPGAgent:
    """Attention-Enhanced Multi-Agent DDPG Agent"""
    
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, config: Dict):
        """
        Initialize AE-MADDPG agent
        
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
        self.agent_type_id = {'satellite': 0, 'uav': 1, 'ground_station': 2}[self.agent_type]
        
        # Network configuration
        network_config = {
            'embed_dim': config.get('embed_dim', 256),
            'num_heads': config.get('num_heads', 8),
            'dropout': config.get('dropout', 0.1)
        }
        
        # Create networks
        self.actor = AttentionEnhancedActor(obs_dim, action_dim, network_config).to(self.device)
        self.critic = AttentionEnhancedCritic(
            obs_dim, action_dim, config['num_agents'], network_config
        ).to(self.device)
        
        # Target networks
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # Freeze target networks
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.get('actor_lr', 3e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.get('critic_lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        
        # Attention-specific parameters
        self.attention_reg_weight = config.get('attention_reg_weight', 0.01)
        self.entropy_reg_weight = config.get('entropy_reg_weight', 0.001)
        
        # Exploration noise
        self.exploration_noise = config.get('exploration_noise', 0.1)
        self.noise_decay = config.get('noise_decay', 0.995)
        
        # Training statistics
        self.training_step = 0
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'attention_entropy': [],
            'q_values': []
        }
        
        # Attention analyzer
        self.attention_analyzer = AttentionAnalyzer()
        
        # Performance metrics
        self.episode_rewards = []
        self.episode_coverage = []
        
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
    
    def act(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Generate action using attention-enhanced actor
        
        Args:
            obs: Observation array
            add_noise: Whether to add exploration noise
            
        Returns:
            action: Action array
        """
        self.actor.eval()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor, self.agent_type_id).cpu().numpy()[0]
            
            # Add exploration noise
            if add_noise:
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action + noise, -1, 1)
        
        return action
    
    def update_critic(self, batch: Dict, other_agents: List['AEMADDPGAgent']) -> float:
        """
        Update critic network
        
        Args:
            batch: Training batch
            other_agents: List of other agents for centralized training
            
        Returns:
            critic_loss: Critic loss value
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        batch_size = states.shape[0]
        
        # Compute target Q-values
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = []
            for i, agent in enumerate([self] + other_agents):
                if i == self.agent_id:
                    next_action = self.target_actor(next_states[:, i], self.agent_type_id)
                else:
                    next_action = agent.target_actor(
                        next_states[:, i], agent.agent_type_id
                    )
                next_actions.append(next_action)
            
            next_actions_tensor = torch.stack(next_actions, dim=1)
            
            # Compute target Q-values
            target_q_values = self.target_critic(next_states, next_actions_tensor)
            target_q_values = rewards[:, self.agent_id:self.agent_id+1] + \
                             self.gamma * target_q_values * (1 - dones[:, self.agent_id:self.agent_id+1])
        
        # Current Q-values
        current_q_values = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, batch: Dict, other_agents: List['AEMADDPGAgent']) -> Tuple[float, Dict]:
        """
        Update actor network with attention regularization
        
        Args:
            batch: Training batch
            other_agents: List of other agents
            
        Returns:
            actor_loss: Actor loss value
            attention_stats: Attention statistics
        """
        states = batch['states'].to(self.device)
        
        # Generate actions from current actor
        current_actions = []
        for i, agent in enumerate([self] + other_agents):
            if i == self.agent_id:
                action = self.actor(states[:, i], self.agent_type_id)
            else:
                with torch.no_grad():
                    action = agent.actor(states[:, i], agent.agent_type_id)
            current_actions.append(action)
        
        current_actions_tensor = torch.stack(current_actions, dim=1)
        
        # Actor loss (policy gradient)
        actor_loss = -self.critic(states, current_actions_tensor).mean()
        
        # Get attention weights for regularization
        attention_weights = self.actor.get_attention_weights()
        
        # Attention regularization
        attention_reg_loss = 0.0
        attention_stats = {}
        
        for attention_type, weights in attention_weights.items():
            # Entropy regularization
            entropy_loss = AttentionRegularizer.entropy_regularization(
                weights.mean(dim=1), target_entropy=1.0
            )
            attention_reg_loss += entropy_loss
            
            # Record statistics
            attention_stats[f'{attention_type}_entropy'] = \
                self.attention_analyzer.compute_attention_entropy(weights)
        
        # Total loss
        total_loss = actor_loss + self.attention_reg_weight * attention_reg_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        
        # Record attention patterns
        self.attention_analyzer.record_attention(attention_weights)
        
        return actor_loss.item(), attention_stats
    
    def update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update(self, batch: Dict, other_agents: List['AEMADDPGAgent']) -> Dict:
        """
        Complete update step
        
        Args:
            batch: Training batch
            other_agents: List of other agents
            
        Returns:
            training_info: Dictionary of training statistics
        """
        self.training_step += 1
        
        # Update critic
        critic_loss = self.update_critic(batch, other_agents)
        
        # Update actor
        actor_loss, attention_stats = self.update_actor(batch, other_agents)
        
        # Update target networks
        self.update_target_networks()
        
        # Decay exploration noise
        self.exploration_noise = max(0.01, self.exploration_noise * self.noise_decay)
        
        # Record statistics
        self.training_stats['actor_loss'].append(actor_loss)
        self.training_stats['critic_loss'].append(critic_loss)
        
        training_info = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'exploration_noise': self.exploration_noise,
            'training_step': self.training_step,
            **attention_stats
        }
        
        return training_info
    
    def get_attention_analysis(self) -> Dict:
        """Get comprehensive attention analysis"""
        return self.attention_analyzer.get_attention_statistics()
    
    def save(self, filepath: str):
        """Save agent networks and statistics"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'training_step': self.training_step,
            'agent_type': self.agent_type,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent networks and statistics"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_stats = checkpoint['training_stats']
        self.training_step = checkpoint['training_step']
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'attention_entropy': [],
            'q_values': []
        }
        self.attention_analyzer.reset()
        self.episode_rewards = []
        self.episode_coverage = []
    
    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()
    
    def set_train_mode(self):
        """Set networks to training mode"""
        self.actor.train()
        self.critic.train()
        # Target networks stay in eval mode