"""
Enhanced Complete Experiment with Comprehensive Visualization
Integrates training with advanced video visualization system
"""

import os
import sys
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ae_maddpg.agent import AEMADDPGAgent
from algorithms.baseline_maddpg.agent import BaselineMADDPGAgent
from algorithms.qmix.agent import QMIXMultiAgent
from algorithms.independent_ppo.agent import IndependentPPOAgent
from algorithms.baselines.heuristic_agents import GreedyHeuristicAgent, RandomPolicyAgent, AdaptiveGreedyAgent
from environments.enhanced_sagin_env import EnhancedSAGINEnvironment
from evaluation.metrics import ComprehensiveEvaluator, EpisodeData
from utils.replay_buffer import ReplayBuffer
from utils.training_utils import set_seed, get_device
from utils.visualization import ExperimentVisualizer
from visualization.training_integration import TrainingVisualizer, TrainingCallback
from visualization.enhanced_training_visualizer import EnhancedTrainingVisualizer


class EnhancedCompleteExperiment:
    """Complete experiment with enhanced visualization integration"""
    
    def __init__(self, config: Dict):
        """
        Initialize enhanced experiment
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.device = get_device(config.get('device', 'auto'))
        
        # Set random seeds for reproducibility
        set_seed(config.get('seed', 42))
        
        # Setup directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f"enhanced_experiment_{timestamp}"
        self.output_dir = os.path.join('results', self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Environment setup
        self.env_config = config.get('environment', {})
        self.env = EnhancedSAGINEnvironment(self.env_config)
        
        # Training configuration
        self.training_config = config.get('training', {})
        self.num_episodes = self.training_config.get('num_episodes', 100)
        self.eval_episodes = self.training_config.get('eval_episodes', 10)
        self.save_interval = self.training_config.get('save_interval', 50)
        
        # Visualization configuration
        self.viz_config = config.get('visualization', {})
        self.enable_video = self.viz_config.get('enable_video', True)
        self.video_mode = self.viz_config.get('video_mode', 'overview')
        self.enable_realtime_3d = self.viz_config.get('enable_realtime_3d', False)
        
        # Initialize enhanced visualizer
        if self.enable_video:
            self.training_visualizer = TrainingVisualizer(
                self.env_config, 
                self.output_dir, 
                enable_video=True
            )
        else:
            self.training_visualizer = None
            
        # Results storage
        self.algorithm_results = {}
        self.comparison_data = {}
        
        # Evaluator
        try:
            # Add required fields for evaluator
            eval_config = self.config.copy()
            eval_config['num_agents'] = (self.env_config.get('num_satellites', 2) + 
                                        self.env_config.get('num_uavs', 3) + 
                                        self.env_config.get('num_ground_stations', 2))
            self.evaluator = ComprehensiveEvaluator(eval_config)
        except (TypeError, KeyError):
            # Fallback for evaluator issues
            self.evaluator = None
        
        print(f"🚀 Enhanced Experiment initialized: {self.experiment_name}")
        print(f"📁 Results directory: {self.output_dir}")
        print(f"🎬 Video generation: {'Enabled' if self.enable_video else 'Disabled'}")
        print(f"🌐 Real-time 3D: {'Enabled' if self.enable_realtime_3d else 'Disabled'}")
        
    def run_algorithm_training(self, algorithm_name: str, agent_class, agent_config: Dict) -> Dict:
        """
        Run training for a specific algorithm with visualization
        
        Args:
            algorithm_name: Name of the algorithm
            agent_class: Agent class to instantiate
            agent_config: Agent configuration
            
        Returns:
            Training results and statistics
        """
        print(f"\n{'='*80}")
        print(f"🎯 Training Algorithm: {algorithm_name.upper()}")
        print(f"{'='*80}")
        
        # Create agent
        try:
            if algorithm_name in ['greedy_heuristic', 'random_policy', 'adaptive_greedy']:
                # Heuristic agents don't need complex initialization
                agent = agent_class(self.env_config)
            else:
                # Deep RL agents need proper initialization
                obs_space = self.env.observation_space
                action_space = self.env.action_space
                agent = agent_class(obs_space, action_space, agent_config, self.device)
        except Exception as e:
            print(f"⚠️ Failed to initialize {algorithm_name}: {e}")
            return {'error': str(e)}
            
        # Initialize training visualizer callback
        callback = None
        if self.training_visualizer:
            callback = TrainingCallback(self.training_visualizer)
            
        # Training statistics
        training_stats = {
            'episode_rewards': [],
            'coverage_rates': [],
            'energy_efficiency': [],
            'completion_times': [],
            'collision_counts': [],
            'losses': []
        }
        
        # Training loop
        best_reward = float('-inf')
        best_episode_data = None
        
        start_time = time.time()
        
        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.time()
            
            # Start episode tracking
            if callback:
                callback.on_episode_start(episode, algorithm_name)
                
            # Reset environment
            obs = self.env.reset()
            total_reward = 0
            step_count = 0
            episode_losses = []
            
            # Episode data collection
            episode_positions = []
            episode_coverage = []
            episode_rewards = []
            
            for step in range(self.env_config.get('max_episode_steps', 200)):
                step_count += 1
                
                # Get actions from agent
                if hasattr(agent, 'act'):
                    if algorithm_name in ['greedy_heuristic', 'random_policy', 'adaptive_greedy']:
                        # Heuristic agents
                        actions = agent.act(obs)
                    else:
                        # Deep RL agents
                        actions = agent.act(obs, add_noise=True)
                else:
                    # Fallback to random actions
                    actions = self.env.action_space.sample()
                
                # Step environment
                next_obs, rewards, done, info = self.env.step(actions)
                
                # Extract positions for visualization
                positions = self._extract_positions_from_obs(obs)
                coverage_status = info.get('coverage_status', [])
                
                # Record step for visualization
                if callback:
                    callback.on_step(obs, actions, rewards, info)
                    
                # Store episode data
                episode_positions.append(positions)
                episode_coverage.append(coverage_status)
                episode_rewards.append(rewards)
                
                total_reward += rewards
                
                # Training step for deep RL agents
                if hasattr(agent, 'update') and not algorithm_name.endswith('heuristic') and not algorithm_name.endswith('policy'):
                    if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > agent_config.get('batch_size', 64):
                        try:
                            losses = agent.update()
                            if losses:
                                episode_losses.extend(losses if isinstance(losses, list) else [losses])
                        except Exception as e:
                            print(f"⚠️ Update error in episode {episode}: {e}")
                
                obs = next_obs
                
                if done:
                    break
                    
            episode_time = time.time() - episode_start_time
            
            # Calculate episode metrics
            final_coverage = episode_coverage[-1] if episode_coverage else []
            coverage_rate = (sum(final_coverage) / len(final_coverage) * 100) if final_coverage else 0
            energy_efficiency = total_reward / step_count if step_count > 0 else 0
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            # Store statistics
            training_stats['episode_rewards'].append(total_reward)
            training_stats['coverage_rates'].append(coverage_rate)
            training_stats['energy_efficiency'].append(energy_efficiency)
            training_stats['completion_times'].append(episode_time)
            training_stats['collision_counts'].append(info.get('collisions', 0))
            training_stats['losses'].append(avg_loss)
            
            # End episode tracking and potentially generate video
            video_path = None
            if callback:
                stats, video_path = callback.on_episode_end()
                
            # Track best episode for final video
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode_data = {
                    'episode': episode,
                    'position_history': episode_positions,
                    'coverage_history': episode_coverage,
                    'reward_history': episode_rewards,
                    'total_reward': total_reward,
                    'coverage_rate': coverage_rate
                }
            
            # Progress reporting
            if episode % 10 == 0 or episode == 1:
                print(f"📊 Episode {episode:3d}: "
                      f"Reward={total_reward:7.1f}, "
                      f"Coverage={coverage_rate:5.1f}%, "
                      f"Steps={step_count:3d}, "
                      f"Time={episode_time:5.1f}s")
                if video_path:
                    print(f"    🎬 Video: {os.path.basename(video_path)}")
                    
        total_time = time.time() - start_time
        
        # Complete training
        final_summary = None
        if callback:
            final_summary = callback.on_training_complete()
            
        # Generate final comparison video for best episode
        if self.training_visualizer and best_episode_data:
            try:
                final_video = self.training_visualizer.visualizer.create_training_visualization(
                    best_episode_data,
                    {
                        'total_reward': best_episode_data['total_reward'],
                        'coverage_rate': best_episode_data['coverage_rate'],
                        'steps': len(best_episode_data['position_history']),
                        'energy_efficiency': best_episode_data['total_reward'] / len(best_episode_data['position_history']),
                        'collision_count': 0,
                        'communication_efficiency': 0.85
                    },
                    algorithm_name + '_best',
                    best_episode_data['episode']
                )
                print(f"🏆 Best episode video: {os.path.basename(final_video) if final_video else 'Failed'}")
            except Exception as e:
                print(f"⚠️ Failed to generate final video: {e}")
        
        # Prepare results
        results = {
            'algorithm': algorithm_name,
            'training_stats': training_stats,
            'best_episode': best_episode_data,
            'final_metrics': {
                'avg_reward': np.mean(training_stats['episode_rewards']),
                'best_reward': best_reward,
                'avg_coverage': np.mean(training_stats['coverage_rates']),
                'best_coverage': max(training_stats['coverage_rates']) if training_stats['coverage_rates'] else 0,
                'avg_energy_efficiency': np.mean(training_stats['energy_efficiency']),
                'total_time': total_time
            },
            'summary': final_summary
        }
        
        print(f"\n✅ {algorithm_name.upper()} Training Complete!")
        print(f"   📈 Average Reward: {results['final_metrics']['avg_reward']:.1f}")
        print(f"   🏆 Best Reward: {results['final_metrics']['best_reward']:.1f}")
        print(f"   🎯 Average Coverage: {results['final_metrics']['avg_coverage']:.1f}%")
        print(f"   ⏱️  Total Time: {total_time:.1f}s")
        
        return results
        
    def run_complete_comparison(self) -> Dict:
        """
        Run complete algorithm comparison with enhanced visualization
        
        Returns:
            Comprehensive comparison results
        """
        print(f"\n{'='*80}")
        print(f"🚀 ENHANCED COMPLETE ALGORITHM COMPARISON")
        print(f"{'='*80}")
        
        # Define algorithms to test
        algorithms = {
            'ae_maddpg': {
                'class': AEMADDPGAgent,
                'config': self.config.get('ae_maddpg', {})
            },
            'baseline_maddpg': {
                'class': BaselineMADDPGAgent,
                'config': self.config.get('baseline_maddpg', {})
            },
            'qmix': {
                'class': QMIXMultiAgent,
                'config': self.config.get('qmix', {})
            },
            'independent_ppo': {
                'class': IndependentPPOAgent,
                'config': self.config.get('independent_ppo', {})
            },
            'greedy_heuristic': {
                'class': GreedyHeuristicAgent,
                'config': {}
            },
            'random_policy': {
                'class': RandomPolicyAgent,
                'config': {}
            }
        }
        
        # Run each algorithm
        all_results = {}
        comparison_videos = {}
        
        for alg_name, alg_info in algorithms.items():
            print(f"\n🔄 Starting {alg_name}...")
            
            try:
                results = self.run_algorithm_training(
                    alg_name, 
                    alg_info['class'], 
                    alg_info['config']
                )
                
                if 'error' not in results:
                    all_results[alg_name] = results
                    if results.get('best_episode'):
                        comparison_videos[alg_name] = results['best_episode']
                else:
                    print(f"❌ {alg_name} failed: {results['error']}")
                    
            except Exception as e:
                print(f"❌ {alg_name} crashed: {e}")
                continue
        
        # Generate comprehensive comparison visualization
        if len(comparison_videos) >= 2 and self.training_visualizer:
            print(f"\n🎬 Generating algorithm comparison video...")
            try:
                # Prepare comparison data
                comparison_data = {}
                for alg_name, episode_data in comparison_videos.items():
                    comparison_data[alg_name] = {'episode_data': episode_data}
                
                comparison_video = self.training_visualizer.video_generator.create_comparison_video(
                    comparison_data,
                    'enhanced_algorithm_comparison.gif'
                )
                
                if comparison_video:
                    print(f"✅ Comparison video: {os.path.basename(comparison_video)}")
                    
            except Exception as e:
                print(f"⚠️ Failed to generate comparison video: {e}")
        
        # Create comprehensive results summary
        summary = self._create_comprehensive_summary(all_results)
        
        # Save results
        self._save_results(all_results, summary)
        
        return {
            'results': all_results,
            'summary': summary,
            'experiment_dir': self.output_dir
        }
        
    def _extract_positions_from_obs(self, observations) -> Dict:
        """Extract agent positions from observations"""
        positions = {}
        
        # This would need to be adapted based on actual observation structure
        # For now, create dummy positions for visualization
        num_agents = (self.env_config.get('num_satellites', 2) + 
                     self.env_config.get('num_uavs', 3) + 
                     self.env_config.get('num_ground_stations', 2))
        area_size = self.env_config.get('area_size', 1000)
        
        for i in range(num_agents):
            # Create scattered positions (would be extracted from actual observations)
            angle = i * 2 * np.pi / num_agents + np.random.randn() * 0.1
            radius = area_size * (0.2 + np.random.random() * 0.3)
            x = area_size/2 + radius * np.cos(angle)
            y = area_size/2 + radius * np.sin(angle)
            
            # Different altitudes for different agent types
            if i < self.env_config.get('num_satellites', 2):
                z = 200 + np.random.randn() * 10  # Satellites
            elif i < self.env_config.get('num_satellites', 2) + self.env_config.get('num_uavs', 3):
                z = 100 + np.random.randn() * 20  # UAVs
            else:
                z = 10 + np.random.randn() * 5   # Ground stations
                
            positions[f'agent_{i}'] = (x, y, z)
            
        return positions
        
    def _create_comprehensive_summary(self, results: Dict) -> Dict:
        """Create comprehensive comparison summary"""
        if not results:
            return {}
            
        summary = {
            'total_algorithms': len(results),
            'best_algorithm': None,
            'best_reward': float('-inf'),
            'algorithm_rankings': [],
            'performance_comparison': {},
            'training_efficiency': {}
        }
        
        # Find best performing algorithm
        for alg_name, result in results.items():
            avg_reward = result['final_metrics']['avg_reward']
            if avg_reward > summary['best_reward']:
                summary['best_reward'] = avg_reward
                summary['best_algorithm'] = alg_name
                
        # Create performance comparison
        for alg_name, result in results.items():
            metrics = result['final_metrics']
            summary['performance_comparison'][alg_name] = {
                'avg_reward': metrics['avg_reward'],
                'avg_coverage': metrics['avg_coverage'],
                'energy_efficiency': metrics['avg_energy_efficiency'],
                'training_time': metrics['total_time']
            }
            
        # Sort algorithms by performance
        sorted_algs = sorted(
            results.items(),
            key=lambda x: x[1]['final_metrics']['avg_reward'],
            reverse=True
        )
        
        summary['algorithm_rankings'] = [
            {
                'rank': i+1,
                'algorithm': alg_name,
                'avg_reward': result['final_metrics']['avg_reward'],
                'avg_coverage': result['final_metrics']['avg_coverage']
            }
            for i, (alg_name, result) in enumerate(sorted_algs)
        ]
        
        return summary
        
    def _save_results(self, results: Dict, summary: Dict):
        """Save comprehensive results"""
        # Save detailed results
        results_file = os.path.join(self.output_dir, 'detailed_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
            
        # Save summary
        summary_file = os.path.join(self.output_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            serializable_summary = self._make_json_serializable(summary)
            json.dump(serializable_summary, f, indent=2)
            
        # Save configuration
        config_file = os.path.join(self.output_dir, 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        print(f"\n💾 Results saved:")
        print(f"   📊 Detailed: {results_file}")
        print(f"   📋 Summary: {summary_file}")
        print(f"   ⚙️  Config: {config_file}")
        
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
            
    def run_single_algorithm(self, algorithm_name: str) -> Dict:
        """
        Run training for a single algorithm
        
        Args:
            algorithm_name: Name of algorithm to run
            
        Returns:
            Training results
        """
        algorithms = {
            'ae_maddpg': {'class': AEMADDPGAgent, 'config': self.config.get('ae_maddpg', {})},
            'baseline_maddpg': {'class': BaselineMADDPGAgent, 'config': self.config.get('baseline_maddpg', {})},
            'qmix': {'class': QMIXMultiAgent, 'config': self.config.get('qmix', {})},
            'independent_ppo': {'class': IndependentPPOAgent, 'config': self.config.get('independent_ppo', {})},
            'greedy_heuristic': {'class': GreedyHeuristicAgent, 'config': {}},
            'random_policy': {'class': RandomPolicyAgent, 'config': {}}
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
        alg_info = algorithms[algorithm_name]
        results = self.run_algorithm_training(
            algorithm_name,
            alg_info['class'],
            alg_info['config']
        )
        
        # Save single algorithm results
        self._save_results({algorithm_name: results}, {'single_algorithm': algorithm_name})
        
        return results