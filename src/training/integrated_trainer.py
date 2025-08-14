"""
Integrated Training System with Professional Visualization
=========================================================
Combines training pipeline with professional visualizer for complete solution.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from src.utils.training_pipeline import StandardizedTrainingPipeline, TrainingOutputConfig
from src.visualization.professional_visualizer import ProfessionalVisualizer, VisualizationConfig
from src.environments.enhanced_sagin_env import EnhancedSAGINEnvironment


@dataclass
class IntegratedTrainerConfig:
    """Configuration for integrated training system"""
    # Algorithm settings
    algorithm: str = "maddpg"
    episodes: int = 1000
    max_steps: int = 200
    
    # Visualization settings
    visualize_every: int = 10  # Visualize every N episodes
    save_final_gif: bool = True
    gif_fps: int = 10
    
    # Training pipeline settings
    experiment_name: str = "sagin_training"
    checkpoint_frequency: int = 100
    report_frequency: int = 50
    
    # Environment settings
    n_satellites: int = 2
    n_uavs: int = 3
    n_ground: int = 2
    area_size: tuple = (500, 500)


class IntegratedTrainer:
    """Integrated training system with professional outputs"""
    
    def __init__(self, config: Optional[IntegratedTrainerConfig] = None):
        self.config = config or IntegratedTrainerConfig()
        self.logger = self._setup_logger()
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize training pipeline
        pipeline_config = TrainingOutputConfig(
            experiment_name=self.config.experiment_name,
            checkpoint_frequency=self.config.checkpoint_frequency,
            report_frequency=self.config.report_frequency,
            gif_fps=self.config.gif_fps,
            save_gif=True,
            save_mp4=False
        )
        self.pipeline = StandardizedTrainingPipeline(pipeline_config)
        
        # Initialize visualizer
        viz_config = VisualizationConfig(
            fps=self.config.gif_fps,
            background_color='white',
            trail_length=30
        )
        self.visualizer = ProfessionalVisualizer(self.env, viz_config)
        
        # Initialize algorithm
        self.algorithm = self._create_algorithm()
        
        # Training tracking
        self.episode_count = 0
        self.best_reward = -float('inf')
        
    def _setup_logger(self) -> logging.Logger:
        """Setup training logger"""
        logger = logging.getLogger('IntegratedTrainer')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _create_environment(self):
        """Create SAGIN environment"""
        env_config = {
            'n_satellites': self.config.n_satellites,
            'n_uavs': self.config.n_uavs,
            'n_ground_stations': self.config.n_ground,
            'area_size': self.config.area_size,
            'max_velocity': 10.0,
            'communication_range': 100.0
        }
        
        return EnhancedSAGINEnvironment(**env_config)
    
    def _create_algorithm(self):
        """Create RL algorithm"""
        if self.config.algorithm == "maddpg":
            from src.algorithms.ae_maddpg.agent import AttentionEnhancedMADDPG
            
            n_agents = (self.config.n_satellites + 
                       self.config.n_uavs + 
                       self.config.n_ground)
            
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0]
            
            return AttentionEnhancedMADDPG(
                n_agents=n_agents,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=1e-3,
                tau=0.01
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def train_episode(self, episode: int) -> Dict[str, Any]:
        """Train a single episode"""
        states = self.env.reset()
        episode_frames = []
        episode_reward = 0
        episode_metrics = {
            'coverage': [],
            'communication': [],
            'collisions': 0
        }
        
        for step in range(self.config.max_steps):
            # Get actions from algorithm
            actions = self.algorithm.select_actions(states)
            
            # Environment step
            next_states, rewards, dones, info = self.env.step(actions)
            
            # Store frame data for visualization
            frame_data = {
                'step': step,
                'positions': self.env.get_agent_positions(),
                'reward': np.mean(rewards),
                'coverage': info.get('coverage_rate', 0),
                'communications': self._get_communication_links()
            }
            episode_frames.append(frame_data)
            
            # Update metrics
            episode_reward += np.mean(rewards)
            episode_metrics['coverage'].append(info.get('coverage_rate', 0))
            episode_metrics['communication'].append(info.get('comm_success', 0))
            if info.get('collision', False):
                episode_metrics['collisions'] += 1
            
            # Train algorithm
            self.algorithm.update(states, actions, rewards, next_states, dones)
            
            states = next_states
            
            if all(dones):
                break
        
        # Prepare episode data
        episode_data = {
            'total_reward': episode_reward,
            'avg_coverage': np.mean(episode_metrics['coverage']),
            'communication_success': np.mean(episode_metrics['communication']),
            'collision_rate': episode_metrics['collisions'],
            'frames': episode_frames,
            'model_state': self.algorithm.get_state_dict() if hasattr(self.algorithm, 'get_state_dict') else None
        }
        
        return episode_data
    
    def _get_communication_links(self) -> List[Dict]:
        """Get current communication links between agents"""
        links = []
        positions = self.env.get_agent_positions()
        n_agents = len(positions)
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                distance = np.linalg.norm(positions[i] - positions[j])
                connected = distance < self.env.communication_range
                
                links.append({
                    'agents': (i, j),
                    'distance': distance,
                    'connected': connected
                })
        
        return links
    
    def visualize_episode(self, episode: int, episode_data: Dict):
        """Generate visualization for episode"""
        if episode % self.config.visualize_every != 0:
            return
        
        self.logger.info(f"Generating visualization for episode {episode}")
        
        # Generate episode animation
        self.pipeline.generate_episode_video(
            episode=episode,
            frames=episode_data['frames'],
            visualizer=self.visualizer
        )
        
        # Generate static overview
        overview_path = (self.pipeline.paths['videos'] / 
                        f"episode_{episode:04d}_overview.png")
        self.visualizer.create_static_overview(
            episode_data['frames'],
            str(overview_path)
        )
    
    def train(self):
        """Run complete training"""
        self.logger.info(f"Starting training for {self.config.episodes} episodes")
        
        for episode in range(self.config.episodes):
            # Train episode
            episode_data = self.train_episode(episode)
            
            # Log to pipeline
            self.pipeline.log_episode(episode, episode_data)
            
            # Generate visualization
            self.visualize_episode(episode, episode_data)
            
            # Update best reward
            if episode_data['total_reward'] > self.best_reward:
                self.best_reward = episode_data['total_reward']
                self.logger.info(f"New best reward: {self.best_reward:.2f} at episode {episode}")
            
            # Progress logging
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}/{self.config.episodes} | "
                    f"Reward: {episode_data['total_reward']:.2f} | "
                    f"Coverage: {episode_data['avg_coverage']:.1f}% | "
                    f"Comm: {episode_data['communication_success']:.1f}%"
                )
        
        # Finalize training
        output_path = self.pipeline.finalize_training()
        self.logger.info(f"Training complete! Outputs saved to: {output_path}")
        
        return output_path


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SkyNetRL Integrated Training')
    parser.add_argument('--algorithm', type=str, default='maddpg',
                       help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--experiment-name', type=str, default='sagin_training',
                       help='Experiment name for outputs')
    parser.add_argument('--visualize-every', type=int, default=10,
                       help='Generate visualization every N episodes')
    parser.add_argument('--gif-fps', type=int, default=10,
                       help='FPS for GIF animations')
    
    args = parser.parse_args()
    
    # Create configuration
    config = IntegratedTrainerConfig(
        algorithm=args.algorithm,
        episodes=args.episodes,
        experiment_name=args.experiment_name,
        visualize_every=args.visualize_every,
        gif_fps=args.gif_fps
    )
    
    # Create and run trainer
    trainer = IntegratedTrainer(config)
    output_path = trainer.train()
    
    print(f"\nTraining complete! Results saved to: {output_path}")
    print(f"View dashboard at: {output_path}/reports/latest_dashboard.html")


if __name__ == "__main__":
    main()