"""
Baseline MADDPG Networks (without attention mechanisms)
Standard implementation for comparison with AE-MADDPG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class BaselineActor(nn.Module):
    """Standard actor network without attention mechanisms"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            
            nn.Linear(self.hidden_dim, action_dim),
            nn.Tanh()  # Bounded actions [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through baseline actor
        
        Args:
            obs: [batch_size, obs_dim] - observation
        
        Returns:
            actions: [batch_size, action_dim]
        """
        return self.network(obs)


class BaselineCritic(nn.Module):
    """Standard critic network without attention mechanisms"""
    
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim * num_agents, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Action processing
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * num_agents, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Value function
        self.value_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
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
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through baseline critic
        
        Args:
            states: [batch_size, num_agents, obs_dim] - all agents' observations
            actions: [batch_size, num_agents, action_dim] - all agents' actions
        
        Returns:
            q_values: [batch_size, 1]
        """
        batch_size = states.shape[0]
        
        # Flatten states and actions
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        
        # Encode states and actions
        state_features = self.state_encoder(states_flat)
        action_features = self.action_encoder(actions_flat)
        
        # Combine features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Compute Q-value
        q_value = self.value_network(combined_features)
        
        return q_value


class EnhancedBaselineActor(nn.Module):
    """Enhanced baseline actor with better architecture but no attention"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Policy layers with residual connections
        self.policy_layer1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.policy_layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, action_dim)
        
        # Layer normalization and activations
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layer
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections
        
        Args:
            obs: [batch_size, obs_dim] - observation
        
        Returns:
            actions: [batch_size, action_dim]
        """
        # Extract features
        features = self.feature_extractor(obs)
        
        # First policy layer with residual connection
        x1 = self.policy_layer1(features)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x1 = features + x1  # Residual connection
        x1 = self.dropout(x1)
        
        # Second policy layer with residual connection
        x2 = self.policy_layer2(x1)
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        x2 = x1 + x2  # Residual connection
        
        # Output layer
        actions = torch.tanh(self.output_layer(x2))
        
        return actions


class EnhancedBaselineCritic(nn.Module):
    """Enhanced baseline critic with better architecture but no attention"""
    
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # State and action encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim * num_agents, self.hidden_dim * 3),
            nn.LayerNorm(self.hidden_dim * 3),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * num_agents, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced baseline critic
        
        Args:
            states: [batch_size, num_agents, obs_dim]
            actions: [batch_size, num_agents, action_dim]
        
        Returns:
            q_values: [batch_size, 1]
        """
        batch_size = states.shape[0]
        
        # Flatten and encode
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        
        state_features = self.state_encoder(states_flat)
        action_features = self.action_encoder(actions_flat)
        
        # Combine features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Dueling architecture
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)
        
        # Combine value and advantage
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_value