"""
QMIX Algorithm Networks
Implements QMIX with value function factorization for cooperative multi-agent RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class QMixAgent(nn.Module):
    """Individual Q-network for each agent in QMIX"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Agent Q-network
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(self.hidden_dim // 2, action_dim)
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
        Forward pass through Q-network
        
        Args:
            obs: [batch_size, obs_dim] - agent observation
        
        Returns:
            q_values: [batch_size, action_dim] - Q-values for each action
        """
        return self.q_network(obs)


class QMixMixer(nn.Module):
    """QMIX Mixing Network for combining individual Q-values"""
    
    def __init__(self, num_agents: int, state_dim: int, config: Dict):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = config.get('mixer_hidden_dim', 64)
        self.hypernet_hidden_dim = config.get('hypernet_hidden_dim', 128)
        
        # Hypernetworks for generating mixing network weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, num_agents * self.hidden_dim)
        )
        
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, self.hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, self.hidden_dim)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize hypernetwork weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values using hypernetworks
        
        Args:
            agent_qs: [batch_size, num_agents] - Individual Q-values
            states: [batch_size, state_dim] - Global state
        
        Returns:
            qtot: [batch_size, 1] - Mixed Q-value
        """
        batch_size = agent_qs.shape[0]
        
        # Generate mixing network weights and biases
        w1 = torch.abs(self.hyper_w1(states))  # Ensure positive weights
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        
        b1 = self.hyper_b1(states)
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        b2 = self.hyper_b2(states)
        b2 = b2.view(batch_size, 1, 1)
        
        # Forward pass through mixing network
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        qtot = torch.bmm(hidden, w2) + b2
        
        return qtot.view(batch_size, 1)


class QMixRNN(nn.Module):
    """RNN-based Q-network for QMIX with memory"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        self.rnn_hidden_dim = config.get('rnn_hidden_dim', 128)
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.rnn_hidden_dim)
        )
        
        # RNN core
        self.rnn = nn.GRU(self.rnn_hidden_dim, self.rnn_hidden_dim, batch_first=True)
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RNN Q-network
        
        Args:
            obs: [batch_size, seq_len, obs_dim] - observation sequence
            hidden_state: [1, batch_size, rnn_hidden_dim] - RNN hidden state
        
        Returns:
            q_values: [batch_size, seq_len, action_dim]
            new_hidden: [1, batch_size, rnn_hidden_dim]
        """
        batch_size, seq_len = obs.shape[:2]
        
        # Encode observations
        encoded_obs = self.obs_encoder(obs.view(-1, self.obs_dim))
        encoded_obs = encoded_obs.view(batch_size, seq_len, self.rnn_hidden_dim)
        
        # RNN forward pass
        rnn_out, new_hidden = self.rnn(encoded_obs, hidden_state)
        
        # Compute Q-values
        q_values = self.q_head(rnn_out.view(-1, self.rnn_hidden_dim))
        q_values = q_values.view(batch_size, seq_len, self.action_dim)
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize RNN hidden state"""
        return torch.zeros(1, batch_size, self.rnn_hidden_dim)


class DoubleQMixMixer(nn.Module):
    """Double QMIX Mixer for reduced overestimation bias"""
    
    def __init__(self, num_agents: int, state_dim: int, config: Dict):
        super().__init__()
        
        self.mixer1 = QMixMixer(num_agents, state_dim, config)
        self.mixer2 = QMixMixer(num_agents, state_dim, config)
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor, 
                use_mixer1: bool = True) -> torch.Tensor:
        """
        Forward pass through one of the mixers
        
        Args:
            agent_qs: [batch_size, num_agents] - Individual Q-values
            states: [batch_size, state_dim] - Global state
            use_mixer1: Whether to use first mixer
        
        Returns:
            qtot: [batch_size, 1] - Mixed Q-value
        """
        if use_mixer1:
            return self.mixer1(agent_qs, states)
        else:
            return self.mixer2(agent_qs, states)
    
    def get_both_values(self, agent_qs: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both mixers"""
        q1 = self.mixer1(agent_qs, states)
        q2 = self.mixer2(agent_qs, states)
        return q1, q2


class WeightedQMixMixer(nn.Module):
    """Weighted QMIX Mixer with attention-based weighting"""
    
    def __init__(self, num_agents: int, state_dim: int, config: Dict):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = config.get('mixer_hidden_dim', 64)
        
        # Standard QMIX mixer
        self.base_mixer = QMixMixer(num_agents, state_dim, config)
        
        # Attention mechanism for agent importance weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=config.get('attention_dim', 64),
            num_heads=config.get('attention_heads', 4),
            batch_first=True
        )
        
        # State and agent feature encoders
        self.state_encoder = nn.Linear(state_dim, config.get('attention_dim', 64))
        self.agent_encoder = nn.Linear(1, config.get('attention_dim', 64))  # Single Q-value per agent
        
        # Weight combination network
        self.weight_net = nn.Sequential(
            nn.Linear(config.get('attention_dim', 64), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_agents),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mix Q-values with attention-based weighting
        
        Args:
            agent_qs: [batch_size, num_agents] - Individual Q-values
            states: [batch_size, state_dim] - Global state
        
        Returns:
            qtot: [batch_size, 1] - Weighted mixed Q-value
        """
        batch_size = agent_qs.shape[0]
        
        # Get base QMIX output
        base_qtot = self.base_mixer(agent_qs, states)
        
        # Encode state and agent Q-values
        state_features = self.state_encoder(states).unsqueeze(1)  # [batch, 1, dim]
        agent_features = self.agent_encoder(agent_qs.unsqueeze(-1))  # [batch, num_agents, dim]
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            state_features, agent_features, agent_features
        )
        
        # Compute importance weights
        importance_weights = self.weight_net(attended_features.squeeze(1))  # [batch, num_agents]
        
        # Weighted combination
        weighted_qs = torch.sum(agent_qs * importance_weights, dim=1, keepdim=True)
        
        # Combine with base QMIX (learnable interpolation)
        alpha = torch.sigmoid(torch.sum(attended_features.squeeze(1), dim=-1, keepdim=True))
        qtot = alpha * base_qtot + (1 - alpha) * weighted_qs
        
        return qtot