"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Implementation
Baseline implementation for SAGIN network optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from collections import deque
import random


class Actor(nn.Module):
    """Actor network for MADDPG"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """Critic network for MADDPG"""
    
    def __init__(self, state_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        # Critic takes global state and all actions
        total_input_dim = state_dim * num_agents + action_dim * num_agents
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, states, actions):
        # Concatenate all states and actions
        x = torch.cat([states.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for MADDPG"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient Algorithm
    Baseline implementation for multi-agent reinforcement learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MADDPG algorithm
        
        Args:
            config: Configuration dictionary containing:
                - obs_dim: Observation space dimension
                - action_dim: Action space dimension  
                - num_agents: Number of agents
                - actor_lr: Actor learning rate
                - critic_lr: Critic learning rate
                - gamma: Discount factor
                - tau: Target network update rate
                - batch_size: Training batch size
                - buffer_size: Replay buffer size
        """
        self.config = config
        self.num_agents = config.get('num_agents', 6)
        self.obs_dim = config.get('obs_dim', 10)
        self.action_dim = config.get('action_dim', 2)
        
        # Hyperparameters
        self.actor_lr = config.get('actor_lr', 1e-3)
        self.critic_lr = config.get('critic_lr', 1e-3)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.batch_size = config.get('batch_size', 64)
        self.noise_std = config.get('exploration_noise', 0.1)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks for each agent
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        # Initialize networks for each agent
        for i in range(self.num_agents):
            # Actor networks
            actor = Actor(self.obs_dim, self.action_dim).to(self.device)
            target_actor = Actor(self.obs_dim, self.action_dim).to(self.device)
            target_actor.load_state_dict(actor.state_dict())
            
            # Critic networks
            critic = Critic(self.obs_dim, self.action_dim, self.num_agents).to(self.device)
            target_critic = Critic(self.obs_dim, self.action_dim, self.num_agents).to(self.device)
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
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.get('buffer_size', 100000))
        
        # Training statistics
        self.training_step = 0
        self.eval_mode = False
        
        print(f"Initialized MADDPG with {self.num_agents} agents")
    
    def act(self, observations, add_noise: bool = True) -> Dict[int, np.ndarray]:
        """
        Generate actions for all agents
        
        Args:
            observations: Dictionary or array of observations
            add_noise: Whether to add exploration noise
            
        Returns:
            Dictionary mapping agent_id to action array
        """
        actions = {}
        
        # Handle different observation formats
        if isinstance(observations, dict):
            obs_list = [observations[i] for i in range(self.num_agents)]
        else:
            if isinstance(observations, np.ndarray):
                if observations.ndim == 1:
                    obs_list = [observations for _ in range(self.num_agents)]
                else:
                    obs_list = [observations[i] for i in range(min(len(observations), self.num_agents))]
            else:
                obs_list = [np.array(observations) for _ in range(self.num_agents)]
        
        # Ensure we have enough observations
        while len(obs_list) < self.num_agents:
            obs_list.append(obs_list[-1] if obs_list else np.zeros(self.obs_dim))
        
        for i in range(self.num_agents):
            obs = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](obs).cpu().numpy().flatten()
                
                # Add exploration noise during training
                if add_noise and not self.eval_mode:
                    noise = np.random.normal(0, self.noise_std, size=action.shape)
                    action = np.clip(action + noise, -1, 1)
                
                actions[i] = action
        
        return actions
    
    def store_experience(self, obs, actions, rewards, next_obs, done):
        """
        Store experience in replay buffer
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_obs: Next observations
            done: Episode termination flags
        """
        # Convert to numpy arrays if needed
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
            
        # Convert done dict to boolean
        if isinstance(done, dict):
            done_bool = any(done.values())  # Episode ends if any agent is done
        else:
            done_bool = done
        
        # Store in replay buffer
        self.replay_buffer.push(obs_array, actions_array, rewards_array, next_obs_array, done_bool)
    
    def update(self) -> Dict[str, float]:
        """
        Update algorithm parameters
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {'buffer_size': len(self.replay_buffer)}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update each agent
        total_actor_loss = 0
        total_critic_loss = 0
        
        for agent_idx in range(self.num_agents):
            # Update critic
            with torch.no_grad():
                # Get next actions from target actors
                next_actions = []
                for i in range(self.num_agents):
                    next_action = self.target_actors[i](next_states[:, i])
                    next_actions.append(next_action)
                next_actions = torch.stack(next_actions, dim=1)
                
                # Compute target Q-value
                target_q = self.target_critics[agent_idx](next_states, next_actions)
                if rewards.dim() == 1:
                    agent_rewards = rewards.unsqueeze(1)
                else:
                    agent_rewards = rewards[:, agent_idx:agent_idx+1] if rewards.shape[1] > agent_idx else rewards[:, 0:1]
                
                target_q = agent_rewards + self.gamma * target_q * (1 - dones.unsqueeze(1))
            
            # Current Q-value
            current_q = self.critics[agent_idx](states, actions)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q, target_q)
            
            # Update critic
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_idx].step()
            
            # Update actor
            # Get current actions from all actors
            current_actions = []
            for i in range(self.num_agents):
                if i == agent_idx:
                    current_actions.append(self.actors[i](states[:, i]))
                else:
                    current_actions.append(actions[:, i])
            current_actions = torch.stack(current_actions, dim=1)
            
            # Actor loss (negative Q-value to maximize)
            actor_loss = -self.critics[agent_idx](states, current_actions).mean()
            
            # Update actor
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
            # Soft update target networks
            self._soft_update(self.target_actors[agent_idx], self.actors[agent_idx])
            self._soft_update(self.target_critics[agent_idx], self.critics[agent_idx])
        
        self.training_step += 1
        
        return {
            'actor_loss': total_actor_loss / self.num_agents,
            'critic_loss': total_critic_loss / self.num_agents,
            'buffer_size': len(self.replay_buffer),
            'training_step': self.training_step
        }
    
    def _soft_update(self, target, source):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
    
    def set_eval_mode(self):
        """Set algorithm to evaluation mode"""
        self.eval_mode = True
        for actor in self.actors:
            actor.eval()
    
    def set_train_mode(self):
        """Set algorithm to training mode"""
        self.eval_mode = False
        for actor in self.actors:
            actor.train()
    
    def save(self, filepath: str):
        """Save algorithm state"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [actor.state_dict() for actor in self.target_actors],
            'target_critics': [critic.state_dict() for critic in self.target_critics],
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load algorithm state"""
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get algorithm statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'eval_mode': self.eval_mode,
            'algorithm_type': 'baseline_maddpg',
            'num_agents': self.num_agents
        }