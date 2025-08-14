"""
Replay Buffer for Multi-Agent Deep Reinforcement Learning
"""

import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional


class ReplayBuffer:
    """Experience replay buffer for MADDPG training"""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray,
            next_state: np.ndarray, done: np.ndarray):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            batch: Dictionary of batched experiences
        """
        experiences = random.sample(self.buffer, batch_size)
        
        batch = {
            'state': np.array([e['state'] for e in experiences]),
            'action': np.array([e['action'] for e in experiences]),
            'reward': np.array([e['reward'] for e in experiences]),
            'next_state': np.array([e['next_state'] for e in experiences]),
            'done': np.array([e['done'] for e in experiences])
        }
        
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer capacity
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray,
            next_state: np.ndarray, done: np.ndarray, td_error: Optional[float] = None):
        """
        Add experience with priority
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
            td_error: TD error for priority (if None, uses max priority)
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        # Set priority
        priority = self.max_priority if td_error is None else abs(td_error) + 1e-6
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample batch with importance sampling weights
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            batch: Dictionary of batched experiences
            weights: Importance sampling weights
            indices: Indices of sampled experiences
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        batch = {
            'state': np.array([e['state'] for e in experiences]),
            'action': np.array([e['action'] for e in experiences]),
            'reward': np.array([e['reward'] for e in experiences]),
            'next_state': np.array([e['next_state'] for e in experiences]),
            'done': np.array([e['done'] for e in experiences])
        }
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities of sampled experiences
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0


class MultiAgentReplayBuffer:
    """Specialized replay buffer for multi-agent environments"""
    
    def __init__(self, capacity: int, num_agents: int):
        """
        Initialize multi-agent replay buffer
        
        Args:
            capacity: Maximum buffer capacity
            num_agents: Number of agents
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = deque(maxlen=capacity)
        
    def add(self, states: Dict[int, np.ndarray], actions: Dict[int, np.ndarray],
            rewards: Dict[int, float], next_states: Dict[int, np.ndarray],
            dones: Dict[int, bool]):
        """
        Add multi-agent experience
        
        Args:
            states: Dictionary of agent states
            actions: Dictionary of agent actions
            rewards: Dictionary of agent rewards
            next_states: Dictionary of agent next states
            dones: Dictionary of agent done flags
        """
        experience = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Sample batch of multi-agent experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            batch: Dictionary of batched multi-agent experiences
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # Initialize batch dictionary
        batch = {
            'states': {agent_id: [] for agent_id in range(self.num_agents)},
            'actions': {agent_id: [] for agent_id in range(self.num_agents)},
            'rewards': {agent_id: [] for agent_id in range(self.num_agents)},
            'next_states': {agent_id: [] for agent_id in range(self.num_agents)},
            'dones': {agent_id: [] for agent_id in range(self.num_agents)}
        }
        
        # Collect experiences for each agent
        for experience in experiences:
            for agent_id in range(self.num_agents):
                if agent_id in experience['states']:
                    batch['states'][agent_id].append(experience['states'][agent_id])
                    batch['actions'][agent_id].append(experience['actions'][agent_id])
                    batch['rewards'][agent_id].append(experience['rewards'][agent_id])
                    batch['next_states'][agent_id].append(experience['next_states'][agent_id])
                    batch['dones'][agent_id].append(experience['dones'][agent_id])
        
        # Convert to numpy arrays
        for key in batch:
            for agent_id in batch[key]:
                if batch[key][agent_id]:  # Check if list is not empty
                    batch[key][agent_id] = np.array(batch[key][agent_id])
        
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class EpisodeBuffer:
    """Buffer for storing complete episodes"""
    
    def __init__(self, capacity: int):
        """
        Initialize episode buffer
        
        Args:
            capacity: Maximum number of episodes to store
        """
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        
    def add_episode(self, episode_data: Dict):
        """
        Add complete episode
        
        Args:
            episode_data: Dictionary containing episode information
        """
        self.episodes.append(episode_data)
    
    def sample_episodes(self, num_episodes: int) -> List[Dict]:
        """
        Sample random episodes
        
        Args:
            num_episodes: Number of episodes to sample
            
        Returns:
            sampled_episodes: List of episode data
        """
        if num_episodes > len(self.episodes):
            return list(self.episodes)
        
        return random.sample(self.episodes, num_episodes)
    
    def get_recent_episodes(self, num_episodes: int) -> List[Dict]:
        """
        Get most recent episodes
        
        Args:
            num_episodes: Number of recent episodes to get
            
        Returns:
            recent_episodes: List of recent episode data
        """
        return list(self.episodes)[-num_episodes:]
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.episodes)
    
    def clear(self):
        """Clear the buffer"""
        self.episodes.clear()