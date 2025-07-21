"""Minimal trainer without visualization dependencies for quick testing"""

import numpy as np
from tqdm import tqdm
import os
import torch
from agents.maddpg_agent import MADDPGAgent
from environment.sag_env import SAGEnvironment
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise
from utils.training_metrics import TrainingMetrics


class MinimalMADDPGTrainer:
    def __init__(self, config):
        """Initialize minimal MADDPG trainer"""
        self.config = config
        self.device = config.device
        
        # Initialize environment
        self.env = SAGEnvironment(config)
        
        # Setup agent parameters
        obs_dim = config.individual_obs_dim
        action_dim = config.action_dim
        
        # Initialize agents
        self.agents = [
            MADDPGAgent(obs_dim, action_dim, config, i).to(self.device)
            for i in range(config.num_agents)
        ]
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size,
            config.num_agents,
            obs_dim,
            action_dim
        )
        
        # Initialize exploration noise
        self.noise = OUNoise(
            action_dim,
            config.num_agents,
            scale=config.exploration_noise
        )
        
        # Initialize metrics collector
        self.metrics = TrainingMetrics(config)
        
        # Episode tracking
        self.episode_rewards = []
        self.current_episode = 0
        
    def train(self):
        """Main training loop"""
        print("\nStarting minimal training without visualization...")
        
        for episode in range(self.config.num_episodes):
            self.current_episode = episode
            
            # Reset environment
            observations = self.env.reset()
            episode_reward = 0
            self.noise.reset()
            
            # Run episode
            for step in range(self.config.max_time_steps):
                # Get actions from all agents
                actions = []
                for i, agent in enumerate(self.agents):
                    obs_tensor = torch.FloatTensor(observations[i]).to(self.device)
                    action = agent.act(obs_tensor, add_noise=True)
                    actions.append(action)
                
                # Convert actions to numpy array
                actions = np.array(actions)
                
                # Environment step
                next_observations, rewards, done, info = self.env.step(actions)
                
                # Store transition in replay buffer
                self.replay_buffer.add(
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    done
                )
                
                # Update for next step
                observations = next_observations
                episode_reward += sum(rewards)
                
                # Training update
                if len(self.replay_buffer) > self.config.batch_size:
                    self._update_agents()
                
                if done:
                    break
            
            # Episode complete
            self.episode_rewards.append(episode_reward)
            
            # Update metrics (skip for quick test)
            # self.metrics.update_episode_metrics(...)
            
            # Decay exploration noise
            self.noise.scale *= self.config.exploration_decay
            
            # Log progress
            if episode % self.config.log_frequency == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                print(f"Episode {episode}/{self.config.num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg Reward (10 eps): {avg_reward:.2f} | "
                      f"Coverage: {info.get('coverage_rate', 0):.2%}")
            
            # Save models (skip for quick test)
            # if episode % self.config.save_frequency == 0 and episode > 0:
            #     self._save_models(episode)
        
        print("\nTraining completed!")
        # self._save_models(self.config.num_episodes)  # Skip save for quick test
        
    def _update_agents(self):
        """Update all agents"""
        # Sample batch from replay buffer
        sample = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors and move to device
        obs, actions, rewards, next_obs, dones = [
            torch.FloatTensor(x).to(self.device) for x in sample
        ]
        
        # Update each agent
        for i, agent in enumerate(self.agents):
            # Get other agents list
            other_agents = self.agents[:i] + self.agents[i + 1:]
            
            # Update critic
            agent.update_critic(
                obs, actions, rewards[:, i], next_obs, dones.squeeze(),
                other_agents
            )
            
            # Update actor
            agent.update_actor(
                obs, actions, i, other_agents
            )
            
            # Soft update target networks
            agent._soft_update(agent.target_actor, agent.actor)
            agent._soft_update(agent.target_critic, agent.critic)
            
        # Update training metrics
        # self.metrics.training_updates += 1  # Skip for quick test
        
    def _save_models(self, episode):
        """Save agent models"""
        save_dir = os.path.join(self.config.model_save_path, f"episode_{episode}")
        os.makedirs(save_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent_type = self._get_agent_type(i)
            agent.save(save_dir, f"{agent_type}_{i}")
            
    def _get_agent_type(self, agent_idx):
        """Get the type of agent based on index"""
        if agent_idx < self.config.num_satellites:
            return "satellite"
        elif agent_idx < self.config.num_satellites + self.config.num_uavs:
            return "uav"
        else:
            return "ground_station"