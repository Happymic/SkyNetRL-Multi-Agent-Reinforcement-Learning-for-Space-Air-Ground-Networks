"""
Hierarchical Attention Network Architecture for SAGIN Multi-Agent Systems
Implements sophisticated attention mechanisms for spatial, agent, and task contexts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with configurable heads and dimensions"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled dot-product attention"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v), attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attended, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attended), attention_weights


class SpatialAttentionModule(nn.Module):
    """Spatial attention for POIs, obstacles, and charging stations"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention for spatial features
        self.spatial_attention = MultiHeadAttention(d_model, n_heads)
        
        # Position encoding for spatial relationships
        self.position_encoder = nn.Linear(2, d_model // 4)  # (x, y) -> embedding
        self.distance_encoder = nn.Linear(1, d_model // 4)  # distance -> embedding
        self.type_embedding = nn.Embedding(4, d_model // 4)  # POI, obstacle, station, empty
        self.feature_encoder = nn.Linear(2, d_model // 4)   # priority/radius, importance/availability
        
        # Layer normalization and feedforward
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
    
    def encode_spatial_objects(self, spatial_features, agent_position):
        """Encode spatial objects (POIs, obstacles, etc.) with positional information"""
        batch_size, n_objects, feature_dim = spatial_features.shape
        
        # Extract spatial information (assuming structured input)
        positions = spatial_features[:, :, :2]  # x, y coordinates
        types = torch.clamp(spatial_features[:, :, 2].long(), 0, 3)  # object type, clamped to valid range
        features = spatial_features[:, :, 3:min(5, feature_dim)]  # object-specific features
        
        # Pad features if needed
        if features.shape[-1] < 2:
            padding_size = 2 - features.shape[-1]
            features = F.pad(features, (0, padding_size))
        
        # Calculate distances from agent
        agent_pos_expanded = agent_position.unsqueeze(1).expand(-1, n_objects, -1)
        distances = torch.norm(positions - agent_pos_expanded, dim=-1, keepdim=True)
        
        # Encode components
        pos_enc = self.position_encoder(positions)
        dist_enc = self.distance_encoder(distances)
        type_enc = self.type_embedding(types)
        feat_enc = self.feature_encoder(features)
        
        # Combine encodings
        encoded = torch.cat([pos_enc, dist_enc, type_enc, feat_enc], dim=-1)
        return encoded
    
    def forward(self, spatial_features, agent_position):
        """Apply spatial attention to spatial features"""
        # Encode spatial objects
        encoded_spatial = self.encode_spatial_objects(spatial_features, agent_position)
        
        # Apply self-attention to spatial features
        attended_spatial, attention_weights = self.spatial_attention(
            encoded_spatial, encoded_spatial, encoded_spatial
        )
        
        # Residual connection and layer norm
        spatial_out = self.layer_norm1(attended_spatial + encoded_spatial)
        
        # Feedforward
        ff_out = self.feed_forward(spatial_out)
        spatial_final = self.layer_norm2(ff_out + spatial_out)
        
        return spatial_final, attention_weights


class AgentAttentionModule(nn.Module):
    """Agent-to-agent attention for communication and coordination"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention for agent interactions
        self.agent_attention = MultiHeadAttention(d_model, n_heads)
        
        # Agent state encoders
        self.agent_state_encoder = nn.Linear(9, d_model // 2)  # agent state features
        self.relative_position_encoder = nn.Linear(3, d_model // 4)  # relative position
        self.communication_encoder = nn.Linear(2, d_model // 4)  # comm status, signal strength
        
        # Communication range encoding
        self.comm_range_threshold = 200.0
        
        # Layer normalization and feedforward
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
    
    def encode_agent_features(self, agent_features, self_position):
        """Encode other agents' features with communication information"""
        batch_size, n_agents, feature_dim = agent_features.shape
        
        # Extract agent information (assuming structured input)
        agent_states = agent_features[:, :, :9]  # agent state features
        agent_positions = agent_features[:, :, :3]  # x, y, z positions
        
        # Calculate relative positions
        self_pos_expanded = self_position.unsqueeze(1).expand(-1, n_agents, -1)
        relative_positions = agent_positions - self_pos_expanded
        distances = torch.norm(relative_positions, dim=-1, keepdim=True)
        
        # Communication features
        in_comm_range = (distances <= self.comm_range_threshold).float()
        signal_strength = torch.exp(-distances / (self.comm_range_threshold / 2))
        comm_features = torch.cat([in_comm_range, signal_strength], dim=-1)
        
        # Encode components
        state_enc = self.agent_state_encoder(agent_states)
        pos_enc = self.relative_position_encoder(relative_positions)
        comm_enc = self.communication_encoder(comm_features)
        
        # Combine encodings
        encoded_agents = torch.cat([state_enc, pos_enc, comm_enc], dim=-1)
        
        # Create communication mask (agents outside comm range have reduced attention)
        comm_mask = in_comm_range.squeeze(-1)
        
        return encoded_agents, comm_mask
    
    def forward(self, agent_features, self_position):
        """Apply agent attention for coordination"""
        # Encode agent features
        encoded_agents, comm_mask = self.encode_agent_features(agent_features, self_position)
        
        # Apply attention with communication mask
        attended_agents, attention_weights = self.agent_attention(
            encoded_agents, encoded_agents, encoded_agents, mask=comm_mask.unsqueeze(1).unsqueeze(2)
        )
        
        # Residual connection and layer norm
        agent_out = self.layer_norm1(attended_agents + encoded_agents)
        
        # Feedforward
        ff_out = self.feed_forward(agent_out)
        agent_final = self.layer_norm2(ff_out + agent_out)
        
        return agent_final, attention_weights, comm_mask


class TaskAttentionModule(nn.Module):
    """Task-level attention for mission objectives and priorities"""
    
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention for task features
        self.task_attention = MultiHeadAttention(d_model, n_heads)
        
        # Task encoders
        self.global_state_encoder = nn.Linear(6, d_model // 2)  # global mission state
        self.temporal_encoder = nn.Linear(2, d_model // 4)  # time and progress info
        self.priority_encoder = nn.Linear(2, d_model // 4)  # urgency and importance
        
        # Layer normalization and feedforward
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, global_features):
        """Apply task-level attention to global features"""
        batch_size = global_features.size(0)
        
        # Extract components
        global_state = global_features[:, :6]
        temporal_info = global_features[:, [1, 2]]  # episode progress, avg energy
        priority_info = global_features[:, [0, 5]]  # coverage rate, mission urgency
        
        # Encode components
        global_enc = self.global_state_encoder(global_state)
        temporal_enc = self.temporal_encoder(temporal_info)
        priority_enc = self.priority_encoder(priority_info)
        
        # Combine encodings
        task_features = torch.cat([global_enc, temporal_enc, priority_enc], dim=-1)
        task_features = task_features.unsqueeze(1)  # Add sequence dimension
        
        # Apply self-attention
        attended_task, attention_weights = self.task_attention(
            task_features, task_features, task_features
        )
        
        # Residual connection and layer norm
        task_out = self.layer_norm1(attended_task + task_features)
        
        # Feedforward
        ff_out = self.feed_forward(task_out)
        task_final = self.layer_norm2(ff_out + task_out)
        
        return task_final.squeeze(1), attention_weights


class HierarchicalAttentionNetwork(nn.Module):
    """Hierarchical attention network combining spatial, agent, and task attention"""
    
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_spatial_heads: int = 8,
        n_agent_heads: int = 8,
        n_task_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.self_state_encoder = nn.Linear(9, hidden_dim)
        
        # Attention modules
        self.spatial_attention = SpatialAttentionModule(hidden_dim, n_spatial_heads)
        self.agent_attention = AgentAttentionModule(hidden_dim, n_agent_heads)
        self.task_attention = TaskAttentionModule(hidden_dim, n_task_heads)
        
        # Attention fusion
        self.attention_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Final processing layers
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and std for continuous actions
        )
        
        # Attention weight storage for analysis
        self.last_attention_weights = {}
        
    def parse_structured_observation(self, obs):
        """Parse structured observation into components"""
        if isinstance(obs, dict):
            # Structured observation
            self_state = obs['self_state']
            spatial_features = obs['visible_pois'].reshape(-1, 8, 6)  # Assuming 8 max POIs, 6 features each
            agent_features = obs['visible_agents'].reshape(-1, 6, 8)  # Assuming 6 max agents, 8 features each
            global_features = obs['global_info']
        else:
            # Flat observation - need to parse based on expected structure
            # For 184-dim observation: [self(9) + spatial(80) + agents(90) + global(5)]
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
            
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            
            idx = 0
            self_state = obs[:, idx:idx+9]
            idx += 9
            
            # Spatial features: 80 dims = 20 objects * 4 features (simplified for now)
            spatial_raw = obs[:, idx:idx+80]
            spatial_features = spatial_raw.reshape(batch_size, 20, 4)
            # Pad to 5 features for compatibility
            spatial_features = F.pad(spatial_features, (0, 1))  # Pad last dimension
            # Take first 8 objects
            spatial_features = spatial_features[:, :8, :]
            idx += 80
            
            # Agent features: 90 dims = 15 agents * 6 features (simplified)
            agent_raw = obs[:, idx:idx+90]
            agent_features = agent_raw.reshape(batch_size, 15, 6)
            # Pad to 12 features for compatibility
            agent_features = F.pad(agent_features, (0, 6))  # Pad last dimension
            # Take first 6 agents
            agent_features = agent_features[:, :6, :]
            idx += 90
            
            # Global features: remaining dimensions
            remaining_dims = obs.shape[1] - idx
            global_features = obs[:, idx:idx+min(6, remaining_dims)]
            
            # Ensure global features has 6 dimensions
            if global_features.shape[1] < 6:
                pad_size = 6 - global_features.shape[1]
                global_features = F.pad(global_features, (0, pad_size))
        
        return self_state, spatial_features, agent_features, global_features
    
    def forward(self, obs):
        """Forward pass through hierarchical attention network"""
        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # Parse structured observation
        self_state, spatial_features, agent_features, global_features = self.parse_structured_observation(obs)
        
        # Ensure batch dimension
        if len(self_state.shape) == 1:
            self_state = self_state.unsqueeze(0)
            spatial_features = spatial_features.unsqueeze(0)
            agent_features = agent_features.unsqueeze(0)
            global_features = global_features.unsqueeze(0)
        
        # Encode self state
        self_encoding = self.self_state_encoder(self_state)
        agent_position = self_state[:, :3]  # Extract position for spatial calculations
        
        # Apply attention modules
        spatial_attended, spatial_weights = self.spatial_attention(spatial_features, agent_position[:, :2])
        agent_attended, agent_weights, comm_mask = self.agent_attention(agent_features, agent_position)
        task_attended, task_weights = self.task_attention(global_features)
        
        # Store attention weights for analysis
        self.last_attention_weights = {
            'spatial': spatial_weights,
            'agent': agent_weights,
            'task': task_weights,
            'communication_mask': comm_mask
        }
        
        # Aggregate attended features
        spatial_aggregated = torch.mean(spatial_attended, dim=1)  # Pool over spatial objects
        agent_aggregated = torch.mean(agent_attended, dim=1)     # Pool over agents
        task_aggregated = task_attended  # Already aggregated
        
        # Fuse attention outputs
        combined_attention = torch.cat([spatial_aggregated, agent_aggregated, task_aggregated], dim=-1)
        fused_features = self.attention_fusion(combined_attention)
        
        # Add self state information
        final_features = fused_features + self_encoding
        
        # Generate outputs
        value = self.value_head(final_features)
        policy_output = self.policy_head(final_features)
        
        # Split policy output into mean and std
        action_mean = torch.tanh(policy_output[:, :self.action_dim])  # Bound actions to [-1, 1]
        action_std = F.softplus(policy_output[:, self.action_dim:]) + 1e-5  # Ensure positive std
        
        if batch_size == 1:
            value = value.squeeze(0)
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        
        return {
            'value': value,
            'action_mean': action_mean,
            'action_std': action_std,
            'attention_weights': self.last_attention_weights
        }
    
    def get_attention_analysis(self):
        """Get detailed attention analysis for interpretability"""
        if not self.last_attention_weights:
            return {}
        
        analysis = {
            'spatial_attention_entropy': self._calculate_attention_entropy(
                self.last_attention_weights['spatial']
            ),
            'agent_attention_entropy': self._calculate_attention_entropy(
                self.last_attention_weights['agent']
            ),
            'task_attention_entropy': self._calculate_attention_entropy(
                self.last_attention_weights['task']
            ),
            'communication_connectivity': torch.mean(
                self.last_attention_weights['communication_mask'].float()
            ).item()
        }
        
        return analysis
    
    def _calculate_attention_entropy(self, attention_weights):
        """Calculate entropy of attention weights (higher = more distributed attention)"""
        if attention_weights is None:
            return 0.0
        
        # Average over heads and batch
        avg_attention = torch.mean(attention_weights, dim=(0, 1))
        # Calculate entropy
        entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8))
        return entropy.item()


class HierarchicalCritic(nn.Module):
    """Hierarchical critic network for value function estimation"""
    
    def __init__(
        self, 
        state_dim: int,
        hidden_dim: int = 256,
        n_agents: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        
        # Individual agent encoders
        self.agent_encoder = nn.Sequential(
            nn.Linear(state_dim // n_agents, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Global state attention
        self.global_attention = MultiHeadAttention(hidden_dim, 8)
        
        # Value estimation
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, global_state):
        """Forward pass for centralized critic"""
        batch_size = global_state.shape[0]
        
        # Reshape to separate agent states
        agent_states = global_state.view(batch_size, self.n_agents, -1)
        
        # Encode each agent's state
        encoded_agents = self.agent_encoder(agent_states)
        
        # Apply global attention
        attended_global, _ = self.global_attention(
            encoded_agents, encoded_agents, encoded_agents
        )
        
        # Aggregate and estimate value
        global_features = torch.mean(attended_global, dim=1)
        value = self.value_network(global_features)
        
        return value


def create_hierarchical_attention_network(config: Dict) -> HierarchicalAttentionNetwork:
    """Factory function to create hierarchical attention network"""
    
    network_config = config.get('hierarchical_attention', {})
    
    return HierarchicalAttentionNetwork(
        state_dim=config.get('state_dim', 184),
        action_dim=config.get('action_dim', 2),
        hidden_dim=network_config.get('hidden_dim', 256),
        n_spatial_heads=network_config.get('n_spatial_heads', 8),
        n_agent_heads=network_config.get('n_agent_heads', 6),
        n_task_heads=network_config.get('n_task_heads', 4),
        dropout=network_config.get('dropout', 0.1)
    )


def create_hierarchical_critic(config: Dict) -> HierarchicalCritic:
    """Factory function to create hierarchical critic"""
    
    critic_config = config.get('hierarchical_critic', {})
    
    return HierarchicalCritic(
        state_dim=config.get('state_dim', 184),
        hidden_dim=critic_config.get('hidden_dim', 256),
        n_agents=config.get('n_agents', 7),
        dropout=critic_config.get('dropout', 0.1)
    )