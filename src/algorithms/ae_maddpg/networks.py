"""
Enhanced Actor-Critic Networks for AE-MADDPG
Implements attention-enhanced networks with proper feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .attention_modules import (
    SpatialAttentionModule,
    AgentAttentionModule,
    TaskAttentionModule,
    AttentionFusion,
    AttentionRegularizer
)


class AttentionEnhancedActor(nn.Module):
    """Actor network with three attention mechanisms"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = config.get('embed_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        
        # Parse observation dimensions
        self.self_obs_dim = 9
        self.spatial_obs_dim = 80  # 20 objects × 4 features
        self.agent_obs_dim = 90    # 10 agents × 9 features
        self.task_obs_dim = 5
        
        # Attention modules
        self.spatial_attention = SpatialAttentionModule(self.embed_dim, self.num_heads)
        self.agent_attention = AgentAttentionModule(self.embed_dim, self.num_heads)
        self.task_attention = TaskAttentionModule(self.embed_dim, self.num_heads // 2)
        
        # Attention fusion
        self.attention_fusion = AttentionFusion(self.embed_dim)
        
        # Agent type embedding
        self.agent_type_embedding = nn.Embedding(3, self.embed_dim)  # 3 agent types
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, action_dim),
            nn.Tanh()  # Bounded actions
        )
        
        # Store attention weights for analysis
        self.attention_weights = {}
        
    def parse_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Parse the enhanced observation into components"""
        # obs shape: [batch_size, 184]
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
    
    def forward(self, obs: torch.Tensor, agent_type: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through attention-enhanced actor
        
        Args:
            obs: [batch_size, obs_dim] - enhanced observation
            agent_type: int - agent type ID (0: satellite, 1: UAV, 2: ground station)
        
        Returns:
            actions: [batch_size, action_dim]
        """
        # Parse observation
        self_obs, spatial_obs, agent_obs, task_obs = self.parse_observation(obs)
        
        # Apply attention mechanisms
        spatial_features, spatial_weights = self.spatial_attention(self_obs, spatial_obs)
        agent_features, agent_weights = self.agent_attention(self_obs, agent_obs)
        task_features, task_weights = self.task_attention(self_obs, task_obs)
        
        # Store attention weights for analysis
        self.attention_weights = {
            'spatial': spatial_weights,
            'agent': agent_weights,
            'task': task_weights
        }
        
        # Fuse attention features
        fused_features = self.attention_fusion(spatial_features, agent_features, task_features)
        
        # Add agent type embedding if provided
        if agent_type is not None:
            batch_size = obs.shape[0]
            type_embed = self.agent_type_embedding(
                torch.tensor([agent_type] * batch_size, device=obs.device)
            )
            fused_features = fused_features + type_embed
        
        # Generate actions
        actions = self.policy_network(fused_features)
        
        return actions
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return stored attention weights for analysis"""
        return self.attention_weights


class AttentionEnhancedCritic(nn.Module):
    """Critic network with global attention mechanisms"""
    
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, config: Dict):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.embed_dim = config.get('embed_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        
        # Global state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim * num_agents, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * num_agents, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Global attention for state-action pairs (combined features have 2*embed_dim)
        self.global_attention = nn.MultiheadAttention(
            self.embed_dim * 2, self.num_heads, batch_first=True
        )
        
        # Value network (final features have 4*embed_dim after concatenation)
        self.value_network = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        
        # Advantage network (for dueling architecture)
        self.advantage_network = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention-enhanced critic
        
        Args:
            states: [batch_size, num_agents, obs_dim] - all agents' observations
            actions: [batch_size, num_agents, action_dim] - all agents' actions
        
        Returns:
            q_values: [batch_size, 1]
        """
        batch_size = states.shape[0]
        
        # Flatten states and actions for encoding
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        
        # Encode states and actions
        state_features = self.state_encoder(states_flat)
        action_features = self.action_encoder(actions_flat)
        
        # Combine state and action features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Apply global attention (self-attention on combined features)
        combined_features_unsqueezed = combined_features.unsqueeze(1)
        attended_features, _ = self.global_attention(
            combined_features_unsqueezed,
            combined_features_unsqueezed,
            combined_features_unsqueezed
        )
        attended_features = attended_features.squeeze(1)
        
        # Concatenate original and attended features
        final_features = torch.cat([combined_features, attended_features], dim=-1)
        
        # Compute value and advantage (dueling architecture)
        value = self.value_network(final_features)
        advantage = self.advantage_network(final_features)
        
        # Combine value and advantage
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_value


class AttentionAnalyzer:
    """Analyzes and visualizes attention patterns"""
    
    def __init__(self):
        self.attention_history = {
            'spatial': [],
            'agent': [],
            'task': []
        }
        
    def record_attention(self, attention_weights: Dict[str, torch.Tensor]):
        """Record attention weights for analysis"""
        for key, weights in attention_weights.items():
            if key in self.attention_history:
                self.attention_history[key].append(weights.detach().cpu().numpy())
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution"""
        # Flatten attention weights
        weights_flat = attention_weights.view(-1, attention_weights.shape[-1])
        # Compute entropy
        entropy = -torch.sum(weights_flat * torch.log(weights_flat + 1e-9), dim=-1)
        return entropy.mean().item()
    
    def get_attention_statistics(self) -> Dict:
        """Compute statistics of recorded attention patterns"""
        stats = {}
        
        for key, history in self.attention_history.items():
            if history:
                history_array = np.array(history)
                stats[key] = {
                    'mean': np.mean(history_array),
                    'std': np.std(history_array),
                    'max': np.max(history_array),
                    'min': np.min(history_array),
                    'entropy': self._compute_entropy_stats(history_array)
                }
        
        return stats
    
    def _compute_entropy_stats(self, attention_array: np.ndarray) -> Dict:
        """Compute entropy statistics for attention patterns"""
        # Compute entropy for each sample
        entropies = []
        for sample in attention_array:
            # Flatten and compute entropy
            flat_sample = sample.reshape(-1, sample.shape[-1])
            entropy = -np.sum(flat_sample * np.log(flat_sample + 1e-9), axis=-1)
            entropies.append(np.mean(entropy))
        
        return {
            'mean': np.mean(entropies),
            'std': np.std(entropies),
            'trend': np.polyfit(range(len(entropies)), entropies, 1)[0]  # Linear trend
        }
    
    def reset(self):
        """Reset attention history"""
        for key in self.attention_history:
            self.attention_history[key] = []