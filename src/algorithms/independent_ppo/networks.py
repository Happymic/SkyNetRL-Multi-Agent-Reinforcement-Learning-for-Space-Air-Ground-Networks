"""
Independent PPO Networks
Each agent trains independently using Proximal Policy Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from torch.distributions import Normal


class PPOActor(nn.Module):
    """PPO Actor Network with continuous action space"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Policy mean
        self.mean_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Policy standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Initialize final layer with smaller weights
        nn.init.xavier_uniform_(self.mean_net[-2].weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network
        
        Args:
            obs: [batch_size, obs_dim] - observations
        
        Returns:
            mean: [batch_size, action_dim] - action means
            std: [batch_size, action_dim] - action standard deviations
        """
        features = self.feature_net(obs)
        mean = self.mean_net(features)
        std = torch.exp(self.log_std.expand_as(mean))
        
        return mean, std
    
    def get_action_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and log probability for given observation
        
        Args:
            obs: [batch_size, obs_dim] - observations
        
        Returns:
            action: [batch_size, action_dim] - sampled actions
            log_prob: [batch_size] - log probabilities
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given observations and actions
        
        Args:
            obs: [batch_size, obs_dim] - observations
            actions: [batch_size, action_dim] - actions to evaluate
        
        Returns:
            log_probs: [batch_size] - log probabilities
            entropy: [batch_size] - action entropy
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy


class PPOCritic(nn.Module):
    """PPO Critic Network (Value Function)"""
    
    def __init__(self, obs_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network
        
        Args:
            obs: [batch_size, obs_dim] - observations
        
        Returns:
            values: [batch_size, 1] - state values
        """
        return self.value_net(obs)


class EnhancedPPOActor(nn.Module):
    """Enhanced PPO Actor with residual connections and attention"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Parse structured observation (similar to attention networks)
        self.self_obs_dim = 9
        self.spatial_obs_dim = 80
        self.agent_obs_dim = 90
        self.task_obs_dim = 5
        
        # Feature encoders for different observation components
        self.self_encoder = nn.Sequential(
            nn.Linear(self.self_obs_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.spatial_obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        self.task_encoder = nn.Sequential(
            nn.Linear(self.task_obs_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 4)
        )
        
        # Feature fusion
        fusion_input_dim = self.hidden_dim // 2 * 3 + self.hidden_dim // 4  # Sum of encoder outputs
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Policy head
        self.mean_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Initialize policy head with smaller weights
        nn.init.xavier_uniform_(self.mean_net[-2].weight, gain=0.01)
    
    def parse_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Parse structured observation into components"""
        idx = 0
        
        # Self observation [batch_size, 9]
        self_obs = obs[:, idx:idx + self.self_obs_dim]
        idx += self.self_obs_dim
        
        # Spatial observation [batch_size, 80]  
        spatial_obs = obs[:, idx:idx + self.spatial_obs_dim]
        idx += self.spatial_obs_dim
        
        # Agent observation [batch_size, 90]
        agent_obs = obs[:, idx:idx + self.agent_obs_dim]
        idx += self.agent_obs_dim
        
        # Task observation [batch_size, 5]
        task_obs = obs[:, idx:idx + self.task_obs_dim]
        
        return self_obs, spatial_obs, agent_obs, task_obs
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with structured observation processing
        
        Args:
            obs: [batch_size, obs_dim] - structured observations
        
        Returns:
            mean: [batch_size, action_dim] - action means
            std: [batch_size, action_dim] - action standard deviations
        """
        # Parse observation components
        self_obs, spatial_obs, agent_obs, task_obs = self.parse_observation(obs)
        
        # Encode each component
        self_features = self.self_encoder(self_obs)
        spatial_features = self.spatial_encoder(spatial_obs)
        agent_features = self.agent_encoder(agent_obs)
        task_features = self.task_encoder(task_obs)
        
        # Fuse features
        fused_features = torch.cat([self_features, spatial_features, agent_features, task_features], dim=-1)
        enhanced_features = self.fusion_net(fused_features)
        
        # Generate policy
        mean = self.mean_net(enhanced_features)
        std = torch.exp(self.log_std.expand_as(mean))
        
        return mean, std
    
    def get_action_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy


class SharedPPONetwork(nn.Module):
    """Shared Actor-Critic Network for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Learnable log std for policy
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layers
        nn.init.xavier_uniform_(self.actor_head[-2].weight, gain=0.01)
        nn.init.xavier_uniform_(self.critic_head[-1].weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared network
        
        Args:
            obs: [batch_size, obs_dim] - observations
        
        Returns:
            mean: [batch_size, action_dim] - policy mean
            std: [batch_size, action_dim] - policy std
            value: [batch_size, 1] - state value
        """
        shared_features = self.shared_net(obs)
        
        # Policy output
        mean = self.actor_head(shared_features)
        std = torch.exp(self.log_std.expand_as(mean))
        
        # Value output  
        value = self.critic_head(shared_features)
        
        return mean, std, value
    
    def get_action_value_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, value, and log probability"""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, value, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions and get values"""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy, value