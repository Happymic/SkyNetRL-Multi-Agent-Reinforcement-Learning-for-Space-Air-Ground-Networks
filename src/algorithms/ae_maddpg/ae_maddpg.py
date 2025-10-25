"""
Attention-Enhanced Multi-Agent Deep Deterministic Policy Gradient (AE-MADDPG)
Advanced implementation with multi-head attention for SAGIN network optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from collections import deque
import random
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for agent interactions"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attention_weights.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out(context)
        return output, attention_weights


class AttentionActor(nn.Module):
    """Attention-enhanced actor network"""
    
    def __init__(self, state_dim: int, action_dim: int, num_agents: int, 
                 embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super(AttentionActor, self).__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.num_agents = num_agents
        
        # Input embedding
        self.input_embed = nn.Linear(state_dim, embed_dim)
        
        # Multi-head attention layers
        self.attention1 = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.attention2 = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward networks
        self.ff1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state, agent_states=None):
        """
        Forward pass with attention mechanism
        
        Args:
            state: Current agent's state
            agent_states: States of all agents for attention
        """
        batch_size = state.size(0)
        
        # Embed input
        x = self.input_embed(state)
        
        if agent_states is not None and agent_states.size(1) > 1:
            # Multi-agent attention
            agent_embeds = self.input_embed(agent_states)
            
            # Self-attention
            attended, _ = self.attention1(agent_embeds, agent_embeds, agent_embeds)
            attended = self.norm1(attended + agent_embeds)
            
            # Cross-attention with current agent
            x_expanded = x.unsqueeze(1)
            cross_attended, _ = self.attention2(x_expanded, attended, attended)
            x = self.norm2(cross_attended.squeeze(1) + x)
        
        # Feedforward
        ff_out = self.ff1(x)
        x = self.norm2(ff_out + x)
        
        # Output action
        action = self.output_layers(x)
        return action


class AttentionCritic(nn.Module):
    """Attention-enhanced critic network"""
    
    def __init__(self, state_dim: int, action_dim: int, num_agents: int,
                 embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super(AttentionCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_agents = num_agents
        
        # State and action embeddings
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(action_dim, embed_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Value networks
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, states, actions):
        """
        Forward pass for critic with attention
        
        Args:
            states: States of all agents [batch_size, num_agents, state_dim]
            actions: Actions of all agents [batch_size, num_agents, action_dim]
        """
        batch_size = states.size(0)
        
        # Embed states and actions
        state_embeds = self.state_embed(states)  # [batch_size, num_agents, embed_dim]
        action_embeds = self.action_embed(actions)  # [batch_size, num_agents, embed_dim]
        
        # Combine state and action embeddings
        combined_embeds = state_embeds + action_embeds
        
        # Apply attention
        attended, _ = self.attention(combined_embeds, combined_embeds, combined_embeds)
        attended = self.norm(attended + combined_embeds)
        
        # Global pooling (mean over agents)
        global_state = torch.mean(attended, dim=1)  # [batch_size, embed_dim]
        global_action = torch.mean(action_embeds, dim=1)  # [batch_size, embed_dim]
        
        # Concatenate and compute value
        combined = torch.cat([global_state, global_action], dim=1)
        value = self.value_net(combined)
        
        return value


class AEMADDPG:
    """
    Attention-Enhanced Multi-Agent Deep Deterministic Policy Gradient
    Advanced algorithm with multi-head attention for complex multi-agent scenarios
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AE-MADDPG algorithm
        
        Args:
            config: Configuration dictionary with algorithm parameters
        """
        self.config = config
        self.num_agents = config.get('num_agents', 6)
        self.obs_dim = config.get('obs_dim', 10)
        self.action_dim = config.get('action_dim', 2)
        
        # Network hyperparameters
        self.embed_dim = config.get('embed_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # Training hyperparameters
        self.actor_lr = config.get('actor_lr', 1e-4)
        self.critic_lr = config.get('critic_lr', 1e-3)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.batch_size = config.get('batch_size', 64)
        self.noise_std = config.get('exploration_noise', 0.1)
        self.noise_decay = config.get('noise_decay', 0.995)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        # Initialize networks for each agent
        for i in range(self.num_agents):
            # Actor networks
            actor = AttentionActor(
                self.obs_dim, self.action_dim, self.num_agents,
                self.embed_dim, self.num_heads, self.dropout
            ).to(self.device)
            target_actor = AttentionActor(
                self.obs_dim, self.action_dim, self.num_agents,
                self.embed_dim, self.num_heads, self.dropout
            ).to(self.device)
            target_actor.load_state_dict(actor.state_dict())
            
            # Critic networks
            critic = AttentionCritic(
                self.obs_dim, self.action_dim, self.num_agents,
                self.embed_dim, self.num_heads, self.dropout
            ).to(self.device)
            target_critic = AttentionCritic(
                self.obs_dim, self.action_dim, self.num_agents,
                self.embed_dim, self.num_heads, self.dropout
            ).to(self.device)
            target_critic.load_state_dict(critic.state_dict())
            
            # Optimizers
            actor_optimizer = optim.Adam(actor.parameters(), lr=self.actor_lr)
            critic_optimizer = optim.Adam(critic.parameters(), lr=self.critic_lr)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        # Experience replay
        self.replay_buffer = deque(maxlen=config.get('buffer_size', 100000))
        
        # Training statistics
        self.training_step = 0
        self.eval_mode = False
        self.current_noise_std = self.noise_std
        
        print(f"Initialized AE-MADDPG with {self.num_agents} agents, attention heads: {self.num_heads}")
    
    def act(self, observations, add_noise: bool = True) -> Dict[int, np.ndarray]:
        """
        Generate actions using attention-enhanced actors
        
        Args:
            observations: Agent observations
            add_noise: Whether to add exploration noise
            
        Returns:
            Dictionary of agent actions
        """
        actions = {}
        
        # Process observations
        if isinstance(observations, dict):
            obs_list = [observations[i] for i in range(self.num_agents)]
        else:
            obs_array = np.array(observations)
            if obs_array.ndim == 1:
                obs_list = [obs_array for _ in range(self.num_agents)]
            else:
                obs_list = [obs_array[i] for i in range(min(len(obs_array), self.num_agents))]
        
        # Ensure we have observations for all agents
        while len(obs_list) < self.num_agents:
            obs_list.append(obs_list[-1] if obs_list else np.zeros(self.obs_dim))
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        obs_batch = obs_tensor.unsqueeze(0)  # Add batch dimension
        
        # Generate actions for each agent
        for i in range(self.num_agents):
            agent_obs = obs_batch[0, i].unsqueeze(0)  # [1, obs_dim]
            
            with torch.no_grad():
                # Use attention mechanism with all agent states
                action = self.actors[i](agent_obs, obs_batch).cpu().numpy().flatten()
                
                # Add exploration noise
                if add_noise and not self.eval_mode:
                    noise = np.random.normal(0, self.current_noise_std, size=action.shape)
                    action = np.clip(action + noise, -1, 1)
                
                actions[i] = action
        
        return actions
    
    def store_experience(self, obs, actions, rewards, next_obs, done):
        """Store experience for training"""
        # Convert to numpy arrays
        if isinstance(obs, dict):
            obs_array = np.array([obs[i] for i in range(self.num_agents)])
        else:
            obs_array = np.array(obs)
            
        if isinstance(actions, dict):
            actions_array = np.array([actions[i] for i in range(self.num_agents)])
        else:
            actions_array = np.array(actions)
            
        if isinstance(rewards, dict):
            rewards_array = np.array([rewards[i] for i in range(self.num_agents)])
        else:
            rewards_array = np.array(rewards) if hasattr(rewards, '__len__') else np.array([rewards])
            
        if isinstance(next_obs, dict):
            next_obs_array = np.array([next_obs[i] for i in range(self.num_agents)])
        else:
            next_obs_array = np.array(next_obs)
            
        # Convert done dict to array
        if isinstance(done, dict):
            done_array = np.array([done[i] for i in range(self.num_agents)], dtype=np.float32)
        else:
            done_array = np.array([done] * self.num_agents, dtype=np.float32)
        
        self.replay_buffer.append((obs_array, actions_array, rewards_array, next_obs_array, done_array))
    
    def update(self) -> Dict[str, float]:
        """Update networks with attention-enhanced learning"""
        if len(self.replay_buffer) < self.batch_size:
            return {'buffer_size': len(self.replay_buffer)}
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        for agent_idx in range(self.num_agents):
            # Update critic
            with torch.no_grad():
                # Get next actions from target actors
                next_actions = []
                for i in range(self.num_agents):
                    agent_next_obs = next_states[:, i]
                    next_action = self.target_actors[i](agent_next_obs, next_states)
                    next_actions.append(next_action)
                next_actions = torch.stack(next_actions, dim=1)
                
                # Target Q-value
                target_q = self.target_critics[agent_idx](next_states, next_actions)
                
                # Agent reward
                if rewards.dim() == 1:
                    agent_rewards = rewards.unsqueeze(1)
                else:
                    agent_rewards = rewards[:, agent_idx:agent_idx+1] if rewards.shape[1] > agent_idx else rewards[:, 0:1]
                
                target_q = agent_rewards + self.gamma * target_q * (1 - dones.unsqueeze(1))
            
            # Current Q-value
            current_q = self.critics[agent_idx](states, actions)
            critic_loss = F.mse_loss(current_q, target_q)
            
            # Update critic
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
            self.critic_optimizers[agent_idx].step()
            
            # Update actor
            current_actions = []
            for i in range(self.num_agents):
                if i == agent_idx:
                    agent_obs = states[:, i]
                    current_actions.append(self.actors[i](agent_obs, states))
                else:
                    current_actions.append(actions[:, i])
            current_actions = torch.stack(current_actions, dim=1)
            
            actor_loss = -self.critics[agent_idx](states, current_actions).mean()
            
            # Add attention regularization
            attention_reg = 0.0
            for param in self.actors[agent_idx].parameters():
                if param.dim() > 1:
                    attention_reg += torch.norm(param, p=2)
            
            actor_loss += self.config.get('attention_reg_weight', 0.01) * attention_reg
            
            # Update actor
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
            self.actor_optimizers[agent_idx].step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
            # Soft update target networks
            self._soft_update(self.target_actors[agent_idx], self.actors[agent_idx])
            self._soft_update(self.target_critics[agent_idx], self.critics[agent_idx])
        
        # Decay exploration noise
        self.current_noise_std *= self.noise_decay
        self.current_noise_std = max(self.current_noise_std, 0.01)
        
        self.training_step += 1
        
        return {
            'actor_loss': total_actor_loss / self.num_agents,
            'critic_loss': total_critic_loss / self.num_agents,
            'buffer_size': len(self.replay_buffer),
            'training_step': self.training_step,
            'noise_std': self.current_noise_std
        }
    
    def _soft_update(self, target, source):
        """Soft update target networks"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
    
    def set_eval_mode(self):
        """Set to evaluation mode"""
        self.eval_mode = True
        for actor in self.actors:
            actor.eval()
    
    def set_train_mode(self):
        """Set to training mode"""
        self.eval_mode = False
        for actor in self.actors:
            actor.train()
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [actor.state_dict() for actor in self.target_actors],
            'target_critics': [critic.state_dict() for critic in self.target_critics],
            'training_step': self.training_step,
            'noise_std': self.current_noise_std
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
        for i, target_actor in enumerate(self.target_actors):
            target_actor.load_state_dict(checkpoint['target_actors'][i])
        for i, target_critic in enumerate(self.target_critics):
            target_critic.load_state_dict(checkpoint['target_critics'][i])
        self.training_step = checkpoint['training_step']
        self.current_noise_std = checkpoint.get('noise_std', self.noise_std)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get algorithm statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'eval_mode': self.eval_mode,
            'algorithm_type': 'ae_maddpg',
            'num_agents': self.num_agents,
            'attention_heads': self.num_heads,
            'current_noise_std': self.current_noise_std
        }