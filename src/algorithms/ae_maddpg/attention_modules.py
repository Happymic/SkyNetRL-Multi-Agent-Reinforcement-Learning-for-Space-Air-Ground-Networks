"""
Enhanced Attention Modules for AE-MADDPG
Implements spatial, agent, and task attention mechanisms with proper feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EnhancedMultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with proper scaling and regularization"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for analysis
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.shape[:2]
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store weights for analysis
        self.attention_weights = attention_weights.detach()
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(context)
        
        # Return both output and attention weights for analysis
        return output, attention_weights


class SpatialAttentionModule(nn.Module):
    """Spatial Attention for environmental awareness"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Feature encoders for different spatial elements
        self.poi_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),  # [x, y, priority, covered, importance]
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),  # [x, y, radius]
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.charging_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),  # [x, y, availability, importance]
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Self-state encoder
        self.self_encoder = nn.Sequential(
            nn.Linear(9, embed_dim),  # Agent's own state
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Multi-head attention
        self.attention = EnhancedMultiHeadAttention(embed_dim, num_heads)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, self_state: torch.Tensor, spatial_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_state: [batch_size, 9] - agent's own state
            spatial_obs: [batch_size, 80] - flattened spatial observations (20 objects × 4 features)
        
        Returns:
            attended_features: [batch_size, embed_dim]
            attention_weights: [batch_size, num_heads, 1, num_objects]
        """
        batch_size = self_state.shape[0]
        
        # Encode self state as query
        self_features = self.self_encoder(self_state).unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Parse and encode spatial observations
        spatial_obs = spatial_obs.view(batch_size, 20, 4)  # Reshape to [batch, num_objects, features]
        
        # Separate different object types (assuming first 10 are POIs, next 5 obstacles, last 5 charging)
        poi_features = spatial_obs[:, :10, :]
        obstacle_features = spatial_obs[:, 10:15, :]
        charging_features = spatial_obs[:, 15:20, :]
        
        # Add dummy features to match expected dimensions
        poi_extended = torch.cat([poi_features, torch.zeros(batch_size, 10, 1).to(poi_features.device)], dim=-1)
        obstacle_extended = torch.cat([obstacle_features[:, :, :3], torch.zeros(batch_size, 5, 2).to(obstacle_features.device)], dim=-1)
        charging_extended = torch.cat([charging_features, torch.zeros(batch_size, 5, 0).to(charging_features.device)], dim=-1)
        
        # Encode each object type
        encoded_objects = []
        
        # Encode POIs
        for i in range(10):
            if i < poi_extended.shape[1]:
                encoded = self.poi_encoder(poi_extended[:, i, :])
                encoded_objects.append(encoded.unsqueeze(1))
        
        # Encode obstacles
        for i in range(5):
            if i < obstacle_extended.shape[1]:
                encoded = self.obstacle_encoder(obstacle_extended[:, i, :3])
                encoded_objects.append(encoded.unsqueeze(1))
        
        # Encode charging stations
        for i in range(5):
            if i < charging_extended.shape[1]:
                encoded = self.charging_encoder(charging_extended[:, i, :])
                encoded_objects.append(encoded.unsqueeze(1))
        
        # Stack all encoded objects
        if encoded_objects:
            spatial_features = torch.cat(encoded_objects, dim=1)  # [batch, num_objects, embed_dim]
        else:
            # Fallback if no objects
            spatial_features = torch.zeros(batch_size, 1, self.embed_dim).to(self_state.device)
        
        # Apply attention (query: self_features, key/value: spatial_features)
        attended_output, attention_weights = self.attention(
            self_features, spatial_features, spatial_features
        )
        
        # Project output
        output = self.output_proj(attended_output.squeeze(1))
        
        return output, attention_weights


class AgentAttentionModule(nn.Module):
    """Agent Attention for multi-agent coordination"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Agent feature encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(9, embed_dim),  # Agent observation dimension
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Relation encoder (for pairwise relationships)
        self.relation_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),  # Two agent features + distance
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Multi-head attention
        self.attention = EnhancedMultiHeadAttention(embed_dim, num_heads)
        
        # Communication range embedding
        self.comm_range_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, self_state: torch.Tensor, other_agents_obs: torch.Tensor,
                agent_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_state: [batch_size, 9] - agent's own state
            other_agents_obs: [batch_size, 90] - other agents' observations (10 agents × 9 features)
            agent_positions: [batch_size, 10, 2] - positions for distance calculation (optional)
        
        Returns:
            attended_features: [batch_size, embed_dim]
            attention_weights: [batch_size, num_heads, 1, num_agents]
        """
        batch_size = self_state.shape[0]
        
        # Encode self state
        self_features = self.agent_encoder(self_state).unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Parse and encode other agents
        other_agents_obs = other_agents_obs.view(batch_size, 10, 9)  # [batch, num_agents, features]
        
        # Encode each agent
        agent_features = []
        for i in range(10):
            agent_feat = self.agent_encoder(other_agents_obs[:, i, :])
            
            # Add relational encoding if positions are provided
            if agent_positions is not None:
                self_pos = self_state[:, :2]  # Assuming first 2 dims are position
                other_pos = agent_positions[:, i, :]
                distance = torch.norm(self_pos - other_pos, dim=1, keepdim=True)
                
                # Create relation features
                relation_input = torch.cat([
                    self_features.squeeze(1),
                    agent_feat,
                    distance
                ], dim=-1)
                
                agent_feat = agent_feat + self.relation_encoder(relation_input)
            
            agent_features.append(agent_feat.unsqueeze(1))
        
        # Stack all agent features
        all_agent_features = torch.cat(agent_features, dim=1)  # [batch, num_agents, embed_dim]
        
        # Add communication range embedding
        all_agent_features = all_agent_features + self.comm_range_embed
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            self_features, all_agent_features, all_agent_features
        )
        
        # Project output
        output = self.output_proj(attended_output.squeeze(1))
        
        return output, attention_weights


class TaskAttentionModule(nn.Module):
    """Task Attention for dynamic priority adjustment"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Task feature encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),  # Task observation dimension
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Priority encoder
        self.priority_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        # Temporal encoder for dynamic priorities
        self.temporal_encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
        # Multi-head attention
        self.attention = EnhancedMultiHeadAttention(embed_dim, num_heads)
        
        # Gating mechanism for priority adjustment
        self.priority_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, self_state: torch.Tensor, task_obs: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_state: [batch_size, 9] - agent's own state
            task_obs: [batch_size, 5] - task-related observations
            temporal_context: [batch_size, seq_len, embed_dim] - temporal context (optional)
        
        Returns:
            attended_features: [batch_size, embed_dim]
            attention_weights: [batch_size, num_heads, 1, 1]
        """
        batch_size = self_state.shape[0]
        
        # Encode task features
        task_features = self.task_encoder(task_obs).unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Extract and encode priority (assuming it's part of task_obs)
        priority = task_obs[:, 2:3]  # Assuming 3rd dimension is priority
        priority_features = self.priority_encoder(priority).unsqueeze(1)
        
        # Combine task and priority features
        combined_features = task_features + priority_features
        
        # Apply temporal encoding if context is provided
        if temporal_context is not None:
            temporal_output, _ = self.temporal_encoder(temporal_context)
            combined_features = combined_features + temporal_output[:, -1:, :]  # Use last timestep
        
        # Create query from self state
        self_features = self.task_encoder(
            torch.cat([self_state[:, :5], torch.zeros(batch_size, 0).to(self_state.device)], dim=-1)
        ).unsqueeze(1)
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            self_features, combined_features, combined_features
        )
        
        # Apply gating mechanism
        gate = self.priority_gate(torch.cat([
            attended_output.squeeze(1),
            task_features.squeeze(1)
        ], dim=-1))
        
        # Gated output
        gated_output = gate * attended_output.squeeze(1) + (1 - gate) * task_features.squeeze(1)
        
        # Project output
        output = self.output_proj(gated_output)
        
        return output, attention_weights


class AttentionFusion(nn.Module):
    """Fuses outputs from all three attention mechanisms"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Feature-specific projections
        self.spatial_proj = nn.Linear(embed_dim, embed_dim)
        self.agent_proj = nn.Linear(embed_dim, embed_dim)
        self.task_proj = nn.Linear(embed_dim, embed_dim)
        
        # Adaptive fusion weights
        self.fusion_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),  # 3 weights for 3 attention types
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, spatial_features: torch.Tensor, agent_features: torch.Tensor,
                task_features: torch.Tensor) -> torch.Tensor:
        """
        Fuses features from three attention mechanisms
        
        Args:
            spatial_features: [batch_size, embed_dim]
            agent_features: [batch_size, embed_dim]
            task_features: [batch_size, embed_dim]
        
        Returns:
            fused_features: [batch_size, embed_dim]
        """
        # Project each feature type
        spatial_proj = self.spatial_proj(spatial_features)
        agent_proj = self.agent_proj(agent_features)
        task_proj = self.task_proj(task_features)
        
        # Compute adaptive fusion weights
        concat_features = torch.cat([spatial_features, agent_features, task_features], dim=-1)
        fusion_weights = self.fusion_network(concat_features)  # [batch, 3]
        
        # Apply weighted fusion
        fused = (fusion_weights[:, 0:1] * spatial_proj +
                 fusion_weights[:, 1:2] * agent_proj +
                 fusion_weights[:, 2:3] * task_proj)
        
        # Final projection
        output = self.output_proj(fused)
        
        return output


class AttentionRegularizer:
    """Provides various regularization terms for attention mechanisms"""
    
    @staticmethod
    def entropy_regularization(attention_weights: torch.Tensor, target_entropy: float = 1.0) -> torch.Tensor:
        """Encourages attention diversity through entropy regularization"""
        # Compute entropy
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
        # Regularization term (minimize difference from target entropy)
        reg_loss = torch.mean((entropy - target_entropy) ** 2)
        return reg_loss
    
    @staticmethod
    def sparsity_regularization(attention_weights: torch.Tensor, sparsity_target: float = 0.1) -> torch.Tensor:
        """Encourages sparse attention patterns"""
        # L1 regularization on attention weights
        sparsity_loss = torch.mean(torch.abs(attention_weights - sparsity_target))
        return sparsity_loss
    
    @staticmethod
    def consistency_regularization(attention_weights_1: torch.Tensor,
                                   attention_weights_2: torch.Tensor) -> torch.Tensor:
        """Encourages consistency between different attention heads/modules"""
        # KL divergence between attention distributions
        kl_div = F.kl_div(torch.log(attention_weights_1 + 1e-9),
                          attention_weights_2, reduction='batchmean')
        return kl_div