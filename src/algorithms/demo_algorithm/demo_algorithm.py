"""
Demo Algorithm Implementation
A simple random action algorithm for testing the framework
"""

import numpy as np
from typing import Dict, Any


class DemoAlgorithm:
    """
    Simple demo algorithm that takes random actions
    Demonstrates the required interface for algorithm implementations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the demo algorithm
        
        Args:
            config: Configuration dictionary containing:
                - obs_dim: Observation space dimension
                - action_dim: Action space dimension  
                - num_agents: Number of agents
                - seed: Random seed
        """
        self.config = config
        self.obs_dim = config.get('obs_dim', 10)
        self.action_dim = config.get('action_dim', 2)
        self.num_agents = config.get('num_agents', 6)
        self.seed = config.get('seed', 42)
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Simple state for demonstration
        self.step_count = 0
        self.total_reward = 0.0
        self.eval_mode = False
        
        print(f"Initialized DemoAlgorithm with {self.num_agents} agents")
    
    def act(self, observations, add_noise: bool = True) -> Dict[int, np.ndarray]:
        """
        Generate actions for all agents
        
        Args:
            observations: Can be array of shape (num_agents, obs_dim) or dict
            add_noise: Whether to add exploration noise (ignored in demo)
            
        Returns:
            actions: Dictionary mapping agent_id to action array
        """
        _ = add_noise  # Unused in demo algorithm
        # Handle different observation formats
        if isinstance(observations, dict):
            num_agents = len(observations)
        else:
            obs_array = np.array(observations)
            if obs_array.ndim == 1:
                num_agents = 1
            else:
                num_agents = obs_array.shape[0]
        
        # Generate random actions for each agent
        actions = {}
        for agent_id in range(num_agents):
            # Generate random action in range [-1, 1]
            action = np.random.uniform(-1, 1, size=(self.action_dim,))
            actions[agent_id] = action
        
        self.step_count += 1
        return actions
    
    def store_experience(self, obs, actions, rewards, next_obs, done):
        """
        Store experience for training (demo algorithm doesn't learn)
        
        Args:
            obs: Current observations (can be dict or array)
            actions: Actions taken (can be dict or array)
            rewards: Rewards received (can be dict or array)
            next_obs: Next observations (can be dict or array)
            done: Episode termination flags (can be dict or array)
        """
        # Mark unused parameters
        _ = obs, actions, next_obs, done
        # Demo algorithm doesn't store experience or learn
        # But we track total reward for statistics
        if isinstance(rewards, dict):
            self.total_reward += float(np.sum(list(rewards.values())))
        elif hasattr(rewards, 'ndim'):
            if rewards.ndim == 0:
                self.total_reward += float(rewards)
            else:
                self.total_reward += float(np.sum(rewards))
        else:
            self.total_reward += float(rewards)
    
    def update(self) -> Dict[str, float]:
        """
        Update algorithm parameters (demo algorithm doesn't learn)
        
        Returns:
            Dictionary of training metrics
        """
        # Demo algorithm doesn't have learnable parameters
        # Return some dummy metrics for compatibility
        return {
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'avg_reward_per_step': self.total_reward / max(1, self.step_count)
        }
    
    def set_eval_mode(self):
        """Set algorithm to evaluation mode"""
        self.eval_mode = True
    
    def set_train_mode(self):
        """Set algorithm to training mode"""  
        self.eval_mode = False
    
    def save(self, filepath: str):
        """Save algorithm state (demo doesn't need to save anything)"""
        print(f"Demo algorithm save called with filepath: {filepath}")
    
    def load(self, filepath: str):
        """Load algorithm state (demo doesn't need to load anything)"""
        print(f"Demo algorithm load called with filepath: {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get algorithm statistics"""
        return {
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'eval_mode': self.eval_mode,
            'algorithm_type': 'demo_random'
        }