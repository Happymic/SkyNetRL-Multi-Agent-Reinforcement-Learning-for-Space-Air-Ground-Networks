"""
Training Integration System for Enhanced Visualization
Connects the enhanced visualizer with actual training processes
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
import json
import time
import os
from collections import defaultdict, deque
from datetime import datetime

from .enhanced_training_visualizer import EnhancedTrainingVisualizer


class TrainingVisualizer:
    """Main interface for training visualization integration"""
    
    def __init__(self, env_config: Dict, output_dir: str, enable_video: bool = True):
        """
        Initialize training visualizer
        
        Args:
            env_config: Environment configuration
            output_dir: Output directory for videos
            enable_video: Whether to generate videos
        """
        self.env_config = env_config
        self.output_dir = output_dir
        self.enable_video = enable_video
        
        # Initialize enhanced visualizer
        self.visualizer = EnhancedTrainingVisualizer(env_config, output_dir)
        
        # Training data collection
        self.current_episode = 0
        self.episode_data = {
            'position_history': [],
            'coverage_history': [],
            'reward_history': [],
            'action_history': [],
            'agent_states': [],
            'communication_data': []
        }
        
        # Performance tracking
        self.training_stats = {
            'episode_rewards': [],
            'coverage_rates': [],
            'energy_efficiency': [],
            'completion_times': [],
            'collision_counts': [],
            'communication_efficiency': []
        }
        
        # Real-time metrics
        self.step_count = 0
        self.episode_start_time = None
        self.total_reward = 0
        
        # Callbacks
        self.callbacks = defaultdict(list)
        
    def start_episode(self, episode_num: int, algorithm_name: str):
        """Start new episode tracking"""
        self.current_episode = episode_num
        self.algorithm_name = algorithm_name
        self.episode_start_time = time.time()
        self.step_count = 0
        self.total_reward = 0
        
        # Reset episode data
        self.episode_data = {
            'position_history': [],
            'coverage_history': [],
            'reward_history': [],
            'action_history': [],
            'agent_states': [],
            'communication_data': [],
            'metadata': {
                'episode': episode_num,
                'algorithm': algorithm_name,
                'start_time': datetime.now().isoformat()
            }
        }
        
        print(f"📊 Started tracking Episode {episode_num} ({algorithm_name})")
        
    def record_step(self, 
                   observations: Dict,
                   actions: np.ndarray,
                   rewards: float,
                   info: Dict):
        """
        Record a single training step
        
        Args:
            observations: Agent observations
            actions: Actions taken by agents
            rewards: Rewards received
            info: Additional information from environment
        """
        self.step_count += 1
        self.total_reward += rewards
        
        # Extract agent positions from observations
        agent_positions = self._extract_positions_from_obs(observations)
        
        # Extract coverage information
        coverage_status = info.get('coverage_status', [])
        
        # Extract agent states
        agent_states = self._extract_agent_states(observations, info)
        
        # Extract communication data
        communication_data = self._extract_communication_data(info)
        
        # Store step data
        self.episode_data['position_history'].append(agent_positions)
        self.episode_data['coverage_history'].append(coverage_status)
        self.episode_data['reward_history'].append(rewards)
        self.episode_data['action_history'].append(actions.tolist() if hasattr(actions, 'tolist') else actions)
        self.episode_data['agent_states'].append(agent_states)
        self.episode_data['communication_data'].append(communication_data)
        
    def end_episode(self, final_info: Dict = None):
        """
        End episode and generate visualization if needed
        
        Args:
            final_info: Final episode information
        """
        episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        # Calculate episode metrics
        coverage_rate = self._calculate_coverage_rate()
        energy_efficiency = self._calculate_energy_efficiency()
        collision_count = self._calculate_collisions()
        comm_efficiency = self._calculate_communication_efficiency()
        
        # Store episode statistics
        episode_stats = {
            'total_reward': self.total_reward,
            'steps': self.step_count,
            'time': episode_time,
            'coverage_rate': coverage_rate,
            'energy_efficiency': energy_efficiency,
            'collision_count': collision_count,
            'communication_efficiency': comm_efficiency
        }
        
        self.training_stats['episode_rewards'].append(self.total_reward)
        self.training_stats['coverage_rates'].append(coverage_rate)
        self.training_stats['energy_efficiency'].append(energy_efficiency)
        self.training_stats['completion_times'].append(episode_time)
        self.training_stats['collision_counts'].append(collision_count)
        self.training_stats['communication_efficiency'].append(comm_efficiency)
        
        print(f"📈 Episode {self.current_episode} completed:")
        print(f"   Reward: {self.total_reward:.1f}, Coverage: {coverage_rate:.1f}%, Steps: {self.step_count}")
        
        # Generate video for key episodes
        video_path = None
        if self.enable_video and self._should_generate_video():
            video_path = self.visualizer.create_training_visualization(
                self.episode_data,
                episode_stats,
                self.algorithm_name,
                self.current_episode
            )
            
        # Execute callbacks
        for callback in self.callbacks.get('episode_end', []):
            callback(self.current_episode, self.episode_data, episode_stats, video_path)
            
        return episode_stats, video_path
        
    def _extract_positions_from_obs(self, observations) -> Dict:
        """Extract agent positions from observations"""
        positions = {}
        
        if isinstance(observations, dict):
            # Multi-agent environment with agent-specific observations
            for agent_id, obs in observations.items():
                if isinstance(obs, np.ndarray) and len(obs) >= 3:
                    # Assuming first 3 elements are x, y, z positions (normalized)
                    x = float(obs[0]) * self.env_config.get('area_size', 1000)
                    y = float(obs[1]) * self.env_config.get('area_size', 1000)
                    z = float(obs[2]) * 300  # Max altitude
                    positions[agent_id] = (x, y, z)
        elif isinstance(observations, np.ndarray):
            # Single observation array for all agents
            num_agents = self.env_config.get('num_satellites', 2) + \
                        self.env_config.get('num_uavs', 3) + \
                        self.env_config.get('num_ground_stations', 2)
            
            # Assuming observations are structured as [agent1_obs, agent2_obs, ...]
            obs_per_agent = len(observations) // num_agents
            for i in range(num_agents):
                start_idx = i * obs_per_agent
                if start_idx + 2 < len(observations):
                    x = float(observations[start_idx]) * self.env_config.get('area_size', 1000)
                    y = float(observations[start_idx + 1]) * self.env_config.get('area_size', 1000)
                    z = float(observations[start_idx + 2]) * 300 if start_idx + 2 < len(observations) else 50
                    positions[f'agent_{i}'] = (x, y, z)
        
        # If we couldn't extract positions, create dummy positions for visualization
        if not positions:
            num_agents = (self.env_config.get('num_satellites', 2) + 
                         self.env_config.get('num_uavs', 3) + 
                         self.env_config.get('num_ground_stations', 2))
            area_size = self.env_config.get('area_size', 1000)
            
            for i in range(num_agents):
                # Create scattered positions
                angle = i * 2 * np.pi / num_agents
                radius = area_size * 0.3
                x = area_size/2 + radius * np.cos(angle) + np.random.randn() * 50
                y = area_size/2 + radius * np.sin(angle) + np.random.randn() * 50
                
                # Different altitudes for different agent types
                if i < self.env_config.get('num_satellites', 2):
                    z = 200  # Satellites
                elif i < self.env_config.get('num_satellites', 2) + self.env_config.get('num_uavs', 3):
                    z = 100  # UAVs
                else:
                    z = 10   # Ground stations
                    
                positions[f'agent_{i}'] = (x, y, z)
                
        return positions
        
    def _extract_agent_states(self, observations, info) -> Dict:
        """Extract detailed agent states"""
        agent_states = {}
        
        # Extract energy levels, communication status, etc.
        for i in range(self.visualizer.num_agents):
            agent_type = self.visualizer._get_agent_type(i)
            
            state = {
                'type': agent_type,
                'active': True,
                'energy': 100 - (self.step_count * 0.5),  # Simulate energy consumption
                'communication_active': True
            }
            
            # UAV specific states
            if agent_type == 'uav':
                state['energy'] = max(0, 100 - (self.step_count * 1.0))  # UAVs consume more energy
                state['battery_warning'] = state['energy'] < 20
                
            agent_states[f'agent_{i}'] = state
            
        return agent_states
        
    def _extract_communication_data(self, info) -> Dict:
        """Extract communication data between agents"""
        comm_data = {
            'active_links': [],
            'data_transmitted': 0,
            'network_efficiency': 0.85 + np.random.randn() * 0.1
        }
        
        # Simulate communication links
        num_agents = self.visualizer.num_agents
        comm_range = self.env_config.get('communication_range', 200)
        
        # This would be extracted from actual environment data
        comm_data['network_efficiency'] = max(0, min(1, comm_data['network_efficiency']))
        
        return comm_data
        
    def _calculate_coverage_rate(self) -> float:
        """Calculate final coverage rate for episode"""
        if not self.episode_data['coverage_history']:
            return 0.0
            
        final_coverage = self.episode_data['coverage_history'][-1]
        return (sum(final_coverage) / len(final_coverage) * 100) if final_coverage else 0.0
        
    def _calculate_energy_efficiency(self) -> float:
        """Calculate energy efficiency for episode"""
        # Simple metric: reward per unit energy consumed
        total_energy_used = self.step_count * 2  # Simulate energy consumption
        return (self.total_reward / total_energy_used) if total_energy_used > 0 else 0.0
        
    def _calculate_collisions(self) -> int:
        """Calculate collision count for episode"""
        # Would be extracted from environment info
        return max(0, int(np.random.poisson(0.5)))  # Simulate low collision rate
        
    def _calculate_communication_efficiency(self) -> float:
        """Calculate communication efficiency"""
        if not self.episode_data['communication_data']:
            return 0.0
            
        efficiencies = [data.get('network_efficiency', 0) for data in self.episode_data['communication_data']]
        return np.mean(efficiencies) if efficiencies else 0.0
        
    def _should_generate_video(self) -> bool:
        """Determine if video should be generated for this episode"""
        # Generate video for:
        # - First episode
        # - Every 10th episode
        # - High-performing episodes
        # - Final episode
        
        if self.current_episode == 1:
            return True
            
        if self.current_episode % 10 == 0:
            return True
            
        # Check if this is a high-performing episode
        if len(self.training_stats['episode_rewards']) >= 5:
            recent_rewards = self.training_stats['episode_rewards'][-5:]
            if self.total_reward > np.mean(recent_rewards) + np.std(recent_rewards):
                return True
                
        return False
        
    def register_callback(self, event: str, callback: Callable):
        """Register callback function for training events"""
        self.callbacks[event].append(callback)
        
    def get_training_summary(self) -> Dict:
        """Get summary of training progress"""
        if not self.training_stats['episode_rewards']:
            return {}
            
        return {
            'episodes_completed': len(self.training_stats['episode_rewards']),
            'total_reward': sum(self.training_stats['episode_rewards']),
            'avg_reward': np.mean(self.training_stats['episode_rewards']),
            'best_reward': max(self.training_stats['episode_rewards']),
            'avg_coverage': np.mean(self.training_stats['coverage_rates']),
            'best_coverage': max(self.training_stats['coverage_rates']),
            'avg_energy_efficiency': np.mean(self.training_stats['energy_efficiency']),
            'total_collisions': sum(self.training_stats['collision_counts'])
        }
        
    def save_training_data(self, filename: str = None):
        """Save all training data to file"""
        if filename is None:
            filename = f"training_data_{self.algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        data = {
            'algorithm': self.algorithm_name,
            'config': self.env_config,
            'training_stats': self.training_stats,
            'summary': self.get_training_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"💾 Training data saved to: {filepath}")
        return filepath


class TrainingCallback:
    """Callback system for training events"""
    
    def __init__(self, visualizer: TrainingVisualizer):
        self.visualizer = visualizer
        
    def on_episode_start(self, episode, algorithm_name):
        """Called at start of each episode"""
        self.visualizer.start_episode(episode, algorithm_name)
        
    def on_step(self, observations, actions, rewards, info):
        """Called after each environment step"""
        self.visualizer.record_step(observations, actions, rewards, info)
        
    def on_episode_end(self, final_info=None):
        """Called at end of each episode"""
        return self.visualizer.end_episode(final_info)
        
    def on_training_complete(self):
        """Called when training is complete"""
        summary = self.visualizer.get_training_summary()
        self.visualizer.save_training_data()
        
        print("\n" + "="*60)
        print("🎯 TRAINING COMPLETE - SUMMARY")
        print("="*60)
        print(f"Algorithm: {self.visualizer.algorithm_name}")
        print(f"Episodes: {summary.get('episodes_completed', 0)}")
        print(f"Average Reward: {summary.get('avg_reward', 0):.2f}")
        print(f"Best Reward: {summary.get('best_reward', 0):.2f}")
        print(f"Average Coverage: {summary.get('avg_coverage', 0):.1f}%")
        print(f"Best Coverage: {summary.get('best_coverage', 0):.1f}%")
        print(f"Energy Efficiency: {summary.get('avg_energy_efficiency', 0):.3f}")
        print("="*60)
        
        return summary