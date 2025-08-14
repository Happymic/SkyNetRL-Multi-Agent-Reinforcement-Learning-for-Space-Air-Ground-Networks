"""
Complete Experiment Framework for SAGIN Coverage Optimization
Includes training, evaluation, and comparison of all algorithms
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import argparse
import copy

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
from visualization.agent_3d_viewer import Agent3DViewer
from visualization.enhanced_3d_viewer import Enhanced3DViewer


class CompleteExperiment:
    """Complete experiment framework for SAGIN optimization"""
    
    def __init__(self, config: Dict):
        """
        Initialize experiment with configuration
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.device = get_device(config.get('device', 'auto'))
        
        # Set random seeds for reproducibility
        set_seed(config.get('seed', 42))
        
        # Create experiment directory
        self.experiment_name = config.get('experiment_name', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir = os.path.join(config.get('results_dir', 'results'), self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize environment
        self.env = EnhancedSAGINEnvironment(config['environment'])
        
        # Initialize evaluator
        self.evaluator = ComprehensiveEvaluator(config['environment'])
        
        # Initialize visualizers
        self.visualizer = ExperimentVisualizer(self.experiment_dir)
        self.agent_3d_viewer = Agent3DViewer(config['environment'], self.experiment_dir)
        self.enhanced_3d_viewer = Enhanced3DViewer(config['environment'], self.experiment_dir)
        
        # Training parameters
        self.num_episodes = config.get('num_episodes', 200)
        self.eval_frequency = config.get('eval_frequency', 10)
        self.save_frequency = config.get('save_frequency', 50)
        
        # Results storage
        self.training_results = defaultdict(list)
        self.evaluation_results = {}
        
        # Save experiment configuration
        self._save_config()
        
        print(f"Experiment initialized: {self.experiment_name}")
        print(f"Results will be saved to: {self.experiment_dir}")
    
    def run_comparison_study(self) -> Dict:
        """
        Run complete comparison study between all algorithms
        
        Returns:
            results: Dictionary of all experimental results
        """
        algorithms_to_test = self.config.get('algorithms', ['ae_maddpg', 'baseline_maddpg'])
        
        print(f"Running comparison study with algorithms: {algorithms_to_test}")
        
        results = {}
        
        for algorithm in algorithms_to_test:
            print(f"\n{'='*50}")
            print(f"Training {algorithm.upper()}")
            print(f"{'='*50}")
            
            # Train algorithm
            training_stats, final_agents, last_episode_data = self._train_algorithm(algorithm)
            
            # Evaluate algorithm
            evaluation_stats = self._evaluate_algorithm(algorithm, final_agents)
            
            # Store results
            results[algorithm] = {
                'training_stats': training_stats,
                'evaluation_stats': evaluation_stats,
                'episode_data': last_episode_data
            }
            
            # Save intermediate results
            self._save_algorithm_results(algorithm, results[algorithm])
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        # Generate visualizations
        self.visualizer.plot_training_comparison(results)
        self.visualizer.plot_performance_comparison(results)
        
        # Create 3D visualizations for each algorithm
        algorithms_3d_data = {}
        for algorithm, result in results.items():
            # Get the last episode data for 3D visualization
            if 'episode_data' in result:
                episode_data = result['episode_data']
                
                # Create both original and enhanced visualizations
                self.agent_3d_viewer.visualize_episode(episode_data, algorithm)
                self.enhanced_3d_viewer.create_detailed_movement_analysis(episode_data, algorithm)
                algorithms_3d_data[algorithm] = episode_data
        
        # Create comparative 3D view
        if len(algorithms_3d_data) > 1:
            self.agent_3d_viewer.create_comparative_view(algorithms_3d_data)
        
        return results
    
    def run_ablation_study(self) -> Dict:
        """
        Run ablation study for attention mechanisms
        
        Returns:
            ablation_results: Results of ablation study
        """
        print(f"\n{'='*50}")
        print("Running Ablation Study")
        print(f"{'='*50}")
        
        # Define ablation configurations
        ablation_configs = {
            'full_attention': {'use_spatial': True, 'use_agent': True, 'use_task': True},
            'no_spatial': {'use_spatial': False, 'use_agent': True, 'use_task': True},
            'no_agent': {'use_spatial': True, 'use_agent': False, 'use_task': True},
            'no_task': {'use_spatial': True, 'use_agent': True, 'use_task': False},
            'no_attention': {'use_spatial': False, 'use_agent': False, 'use_task': False}
        }
        
        ablation_results = {}
        
        for config_name, attention_config in ablation_configs.items():
            print(f"\nTesting configuration: {config_name}")
            
            # Create modified configuration
            modified_config = self.config.copy()
            modified_config['algorithm']['attention_config'] = attention_config
            
            # Train with this configuration
            training_stats, final_agents = self._train_algorithm('ae_maddpg', modified_config)
            
            # Evaluate
            evaluation_stats = self._evaluate_algorithm(f'ae_maddpg_{config_name}', final_agents)
            
            ablation_results[config_name] = {
                'training_stats': training_stats,
                'evaluation_stats': evaluation_stats,
                'attention_config': attention_config
            }
        
        # Generate ablation report
        self._generate_ablation_report(ablation_results)
        
        # Generate ablation visualizations
        self.visualizer.plot_ablation_study(ablation_results)
        
        return ablation_results
    
    def run_scalability_study(self) -> Dict:
        """
        Run scalability study with different environment sizes
        
        Returns:
            scalability_results: Results of scalability study
        """
        print(f"\n{'='*50}")
        print("Running Scalability Study")
        print(f"{'='*50}")
        
        # Define different scales
        scale_configs = {
            'small': {
                'area_size': 400,
                'num_agents': 6,
                'num_satellites': 1,
                'num_uavs': 3,
                'num_ground_stations': 2,
                'num_pois': 8
            },
            'medium': {
                'area_size': 600,
                'num_agents': 10,
                'num_satellites': 2,
                'num_uavs': 5,
                'num_ground_stations': 3,
                'num_pois': 15
            },
            'large': {
                'area_size': 1000,
                'num_agents': 16,
                'num_satellites': 3,
                'num_uavs': 8,
                'num_ground_stations': 5,
                'num_pois': 25
            }
        }
        
        scalability_results = {}
        
        for scale_name, scale_config in scale_configs.items():
            print(f"\nTesting scale: {scale_name}")
            
            # Create modified environment configuration
            modified_config = self.config.copy()
            modified_config['environment'].update(scale_config)
            
            # Create new environment for this scale
            env = EnhancedSAGINEnvironment(modified_config['environment'])
            evaluator = ComprehensiveEvaluator(modified_config['environment'])
            
            # Test both AE-MADDPG and baseline
            scale_results = {}
            for algorithm in ['ae_maddpg', 'baseline_maddpg']:
                print(f"  Testing {algorithm} at {scale_name} scale")
                
                training_stats, final_agents = self._train_algorithm(
                    algorithm, modified_config, env=env
                )
                evaluation_stats = self._evaluate_algorithm(
                    f'{algorithm}_{scale_name}', final_agents, env=env, evaluator=evaluator
                )
                
                scale_results[algorithm] = {
                    'training_stats': training_stats,
                    'evaluation_stats': evaluation_stats
                }
            
            scalability_results[scale_name] = {
                'config': scale_config,
                'results': scale_results
            }
        
        # Generate scalability report
        self._generate_scalability_report(scalability_results)
        
        return scalability_results
    
    def _train_algorithm(self, algorithm: str, config: Optional[Dict] = None, 
                        env: Optional[EnhancedSAGINEnvironment] = None) -> Tuple[Dict, List, Dict]:
        """
        Train a specific algorithm
        
        Args:
            algorithm: Algorithm name
            config: Optional custom configuration
            env: Optional custom environment
            
        Returns:
            training_stats: Training statistics
            final_agents: List of trained agents
            last_episode_data: Data from the last training episode for visualization
        """
        if config is None:
            config = self.config
        if env is None:
            env = self.env
        
        # Initialize agents
        agents = self._initialize_agents(algorithm, config, env)
        
        # Initialize replay buffer
        buffer_size = config.get('buffer_size', 100000)
        batch_size = config.get('batch_size', 32)
        replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        training_stats = {
            'episode_rewards': [],
            'coverage_rates': [],
            'energy_efficiency': [],
            'collision_rates': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        print(f"Starting training for {algorithm}")
        
        for episode in range(self.num_episodes):
            # Run episode
            episode_data = self._run_episode(agents, env, train=True)
            
            # Store episode data in replay buffer
            self._store_episode_data(replay_buffer, episode_data)
            
            # Train agents if buffer has enough samples
            if len(replay_buffer) >= batch_size:
                losses = self._train_agents(agents, replay_buffer, batch_size)
                training_stats['actor_losses'].append(losses.get('actor_loss', 0))
                training_stats['critic_losses'].append(losses.get('critic_loss', 0))
            
            # Record episode statistics
            if isinstance(episode_data['rewards'][0], list):
                episode_reward = sum(sum(step_rewards) for step_rewards in episode_data['rewards'])
            else:
                episode_reward = sum(episode_data['rewards'])
            coverage_rate = episode_data['final_coverage_rate']
            energy_efficiency = episode_data.get('energy_efficiency', 0)
            collision_rate = episode_data.get('collision_rate', 0)
            
            training_stats['episode_rewards'].append(episode_reward)
            training_stats['coverage_rates'].append(coverage_rate)
            training_stats['energy_efficiency'].append(energy_efficiency)
            training_stats['collision_rates'].append(collision_rate)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(training_stats['episode_rewards'][-10:])
                avg_coverage = np.mean(training_stats['coverage_rates'][-10:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Coverage = {avg_coverage:.3f}")
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_stats = self._evaluate_agents(agents, env)
                print(f"Evaluation at episode {episode}: "
                      f"Coverage = {eval_stats['coverage_rate']:.3f}, "
                      f"Energy Efficiency = {eval_stats['energy_efficiency']:.3f}")
        
        print(f"Training completed for {algorithm}")
        
        # Run one final episode for visualization data
        for agent in agents:
            if hasattr(agent, 'set_eval_mode'):
                agent.set_eval_mode()
        final_episode_data = self._run_episode(agents, env, train=False)
        
        return training_stats, agents, final_episode_data
    
    def _evaluate_algorithm(self, algorithm_name: str, agents: List,
                           env: Optional[EnhancedSAGINEnvironment] = None,
                           evaluator: Optional[ComprehensiveEvaluator] = None) -> Dict:
        """
        Evaluate trained algorithm
        
        Args:
            algorithm_name: Name for identification
            agents: List of trained agents
            env: Optional custom environment
            evaluator: Optional custom evaluator
            
        Returns:
            evaluation_stats: Comprehensive evaluation statistics
        """
        if env is None:
            env = self.env
        if evaluator is None:
            evaluator = self.evaluator
        
        print(f"Evaluating {algorithm_name}")
        
        # Reset evaluator
        evaluator.reset()
        
        # Run evaluation episodes
        num_eval_episodes = self.config.get('num_eval_episodes', 20)
        evaluation_episodes = []
        
        for episode in range(num_eval_episodes):
            # Set agents to evaluation mode
            for agent in agents:
                agent.set_eval_mode()
            
            # Run episode
            episode_data = self._run_episode(agents, env, train=False)
            
            # Convert to EpisodeData format for evaluator
            eval_episode_data = EpisodeData(
                states=episode_data.get('states', []),
                actions=episode_data.get('actions', []),
                rewards=episode_data.get('rewards', []),
                coverage_status=episode_data['coverage_history'],
                energy_levels=episode_data['energy_history'],
                collisions=episode_data['collision_history'],
                agent_positions=episode_data['position_history'],
                poi_priorities=episode_data['poi_priorities'],
                timestamps=episode_data.get('timestamps', list(range(len(episode_data['coverage_history']))))
            )
            
            # Evaluate episode
            episode_metrics = evaluator.evaluate_episode(eval_episode_data, env._get_info())
            evaluation_episodes.append(episode_metrics)
            
            if episode % 5 == 0:
                print(f"  Evaluation episode {episode}/{num_eval_episodes}")
        
        # Compute aggregate metrics
        aggregate_metrics = evaluator.get_aggregate_metrics()
        
        # Compute additional statistics
        final_metrics = self._compute_final_evaluation_stats(evaluation_episodes)
        
        evaluation_stats = {
            'aggregate_metrics': aggregate_metrics,
            'final_metrics': final_metrics,
            'episode_metrics': evaluation_episodes
        }
        
        print(f"Evaluation completed for {algorithm_name}")
        print(f"  Final Coverage Rate: {final_metrics.get('avg_coverage_rate', 0):.3f}")
        print(f"  Final Energy Efficiency: {final_metrics.get('avg_energy_efficiency', 0):.3f}")
        
        return evaluation_stats
    
    def _initialize_agents(self, algorithm: str, config: Dict, 
                          env: EnhancedSAGINEnvironment) -> List:
        """Initialize agents for specified algorithm"""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        num_agents = config['environment']['num_agents']
        
        agent_config = {
            **config.get('algorithm', {}),
            'num_agents': num_agents,
            'device': self.device,
            **config['environment']  # Include environment config
        }
        
        if algorithm == 'ae_maddpg':
            agents = []
            for agent_id in range(num_agents):
                agent = AEMADDPGAgent(agent_id, obs_dim, action_dim, agent_config)
                agents.append(agent)
            return agents
        
        elif algorithm == 'baseline_maddpg':
            agents = []
            for agent_id in range(num_agents):
                agent = BaselineMADDPGAgent(agent_id, obs_dim, action_dim, agent_config)
                agents.append(agent)
            return agents
        
        elif algorithm == 'qmix':
            # QMIX uses a single multi-agent system rather than individual agents
            state_dim = obs_dim  # Use observation dimension as state dimension
            qmix_agent = QMIXMultiAgent(num_agents, obs_dim, action_dim, state_dim, agent_config)
            return [qmix_agent]  # Wrap in list for consistent interface
        
        elif algorithm == 'independent_ppo':
            agents = []
            for agent_id in range(num_agents):
                agent = IndependentPPOAgent(agent_id, obs_dim, action_dim, agent_config)
                agents.append(agent)
            return agents
        
        elif algorithm == 'greedy_heuristic':
            agents = []
            for agent_id in range(num_agents):
                agent = GreedyHeuristicAgent(agent_id, agent_config)
                agents.append(agent)
            return agents
        
        elif algorithm == 'adaptive_greedy':
            agents = []
            for agent_id in range(num_agents):
                agent = AdaptiveGreedyAgent(agent_id, agent_config)
                agents.append(agent)
            return agents
        
        elif algorithm == 'random_policy':
            agents = []
            for agent_id in range(num_agents):
                agent = RandomPolicyAgent(agent_id, agent_config)
                agents.append(agent)
            return agents
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _run_episode(self, agents: List, env: EnhancedSAGINEnvironment, 
                    train: bool = True) -> Dict:
        """Run a single episode with given agents"""
        observations = env.reset()
        done = False
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'coverage_history': [],
            'energy_history': [],
            'collision_history': [],
            'position_history': [],
            'poi_priorities': env._get_info()['poi_priorities']
        }
        
        episode_reward = 0
        step = 0
        
        # Handle different agent types
        is_qmix = len(agents) == 1 and hasattr(agents[0], 'mixer')
        is_ppo = len(agents) > 0 and hasattr(agents[0], 'buffer')
        is_heuristic = len(agents) > 0 and not hasattr(agents[0], 'actor') and not hasattr(agents[0], 'mixer')
        
        # Reset episode-specific states
        if is_qmix:
            agents[0].reset_hidden_states()
        elif is_ppo:
            for agent in agents:
                agent.clear_buffer()
        elif is_heuristic:
            for agent in agents:
                if hasattr(agent, 'reset'):
                    agent.reset()
        
        while not done and step < env.max_episode_steps:
            # Get actions from all agents
            actions = {}
            log_probs = {}
            values = {}
            
            if is_qmix:
                # QMIX handles all agents together
                obs_array = np.array([observations[i] for i in range(len(observations))])
                qmix_actions = agents[0].act(obs_array, add_noise=train)
                actions = qmix_actions
            elif is_ppo:
                # PPO agents store transitions for batch updates
                for agent_id, agent in enumerate(agents):
                    obs = observations[agent_id]
                    action, log_prob, value = agent.act(obs, deterministic=not train)
                    actions[agent_id] = action
                    log_probs[agent_id] = log_prob
                    values[agent_id] = value
            elif is_heuristic:
                # Heuristic agents
                for agent_id, agent in enumerate(agents):
                    obs = observations[agent_id]
                    action = agent.act(obs, env._get_info())
                    actions[agent_id] = action
            else:
                # Standard RL agents (AE-MADDPG, Baseline MADDPG)
                for agent_id, agent in enumerate(agents):
                    obs = observations[agent_id]
                    action = agent.act(obs, add_noise=train)
                    actions[agent_id] = action
            
            # Step environment
            next_observations, rewards, dones, info = env.step(actions)
            
            # Store transitions for PPO
            if is_ppo and train:
                for agent_id, agent in enumerate(agents):
                    agent.store_transition(
                        observations[agent_id],
                        actions[agent_id], 
                        rewards[agent_id],
                        dones[agent_id],
                        log_probs[agent_id],
                        values[agent_id]
                    )
            
            # Update adaptive agents
            if hasattr(agents[0], 'update_memory'):
                for agent_id, agent in enumerate(agents):
                    agent.update_memory(actions[agent_id], rewards[agent_id])
            
            # Store episode data
            episode_data['states'].append(copy.deepcopy(observations))
            episode_data['actions'].append(copy.deepcopy(actions))
            episode_data['rewards'].append(list(rewards.values()))
            
            # Update for next step
            observations = next_observations
            episode_reward += sum(rewards.values())
            done = any(dones.values())
            step += 1
        
        # Get final episode information
        episode_data['coverage_history'] = env.coverage_history
        episode_data['energy_history'] = env.energy_history
        episode_data['collision_history'] = env.collision_history
        episode_data['position_history'] = env.position_history
        episode_data['final_coverage_rate'] = info['coverage_rate']
        episode_data['total_reward'] = episode_reward
        
        # Compute additional metrics
        episode_data['energy_efficiency'] = self._compute_energy_efficiency(env.energy_history, info['coverage_rate'])
        episode_data['collision_rate'] = sum(env.collision_history) / (step * len(agents)) if step > 0 else 0
        
        return episode_data
    
    def _store_episode_data(self, replay_buffer: ReplayBuffer, episode_data: Dict):
        """Store episode data in replay buffer"""
        states = episode_data['states']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        
        for i in range(len(states) - 1):
            # Convert to format expected by replay buffer
            state = np.array([states[i][agent_id] for agent_id in range(len(states[i]))])
            action = np.array([actions[i][agent_id] for agent_id in range(len(actions[i]))])
            reward = np.array(rewards[i])
            next_state = np.array([states[i+1][agent_id] for agent_id in range(len(states[i+1]))])
            done = np.zeros(len(reward))  # Not done except for last step
            
            replay_buffer.add(state, action, reward, next_state, done)
    
    def _train_agents(self, agents: List, replay_buffer: ReplayBuffer, 
                     batch_size: int) -> Dict:
        """Train agents using replay buffer"""
        # Handle different agent types
        is_qmix = len(agents) == 1 and hasattr(agents[0], 'mixer')
        is_ppo = len(agents) > 0 and hasattr(agents[0], 'buffer')
        is_heuristic = len(agents) > 0 and not hasattr(agents[0], 'actor') and not hasattr(agents[0], 'mixer')
        
        if is_heuristic:
            # Heuristic agents don't train
            return {'loss': 0.0}
        
        elif is_ppo:
            # PPO agents train when buffer is full
            total_losses = {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
            trained_agents = 0
            
            for agent in agents:
                if agent.is_buffer_full():
                    # Compute GAE
                    agent.compute_gae()
                    
                    # Update agent
                    losses = agent.update()
                    if losses:  # Only if training occurred
                        for key, value in losses.items():
                            if key in total_losses:
                                total_losses[key] += value
                        trained_agents += 1
            
            # Average losses
            if trained_agents > 0:
                for key in total_losses:
                    total_losses[key] /= trained_agents
            
            return total_losses
        
        elif is_qmix:
            # QMIX training
            if len(replay_buffer) < batch_size:
                return {'q_loss': 0.0}
            
            batch = replay_buffer.sample(batch_size)
            
            # Convert to QMIX format
            batch_formatted = {
                'states': torch.FloatTensor(batch['state']),
                'actions': torch.FloatTensor(batch['action']),
                'rewards': torch.FloatTensor(batch['reward']),
                'next_states': torch.FloatTensor(batch['next_state']),
                'dones': torch.FloatTensor(batch['done'])
            }
            
            losses = agents[0].update(batch_formatted)
            return losses
        
        else:
            # Standard RL agents (AE-MADDPG, Baseline MADDPG)
            if len(replay_buffer) < batch_size:
                return {'actor_loss': 0, 'critic_loss': 0}
            
            # Sample batch
            batch = replay_buffer.sample(batch_size)
            
            # Convert to proper format
            batch_formatted = {
                'states': torch.FloatTensor(batch['state']),
                'actions': torch.FloatTensor(batch['action']),
                'rewards': torch.FloatTensor(batch['reward']),
                'next_states': torch.FloatTensor(batch['next_state']),
                'dones': torch.FloatTensor(batch['done'])
            }
            
            # Train each agent
            total_losses = {'actor_loss': 0, 'critic_loss': 0}
            
            for i, agent in enumerate(agents):
                other_agents = agents[:i] + agents[i+1:]
                losses = agent.update(batch_formatted, other_agents)
                
                for key, value in losses.items():
                    if key in total_losses:
                        total_losses[key] += value
            
            # Average losses
            for key in total_losses:
                total_losses[key] /= len(agents)
            
            return total_losses
    
    def _evaluate_agents(self, agents: List, env: EnhancedSAGINEnvironment) -> Dict:
        """Quick evaluation of agents"""
        # Set to evaluation mode
        for agent in agents:
            agent.set_eval_mode()
        
        # Run evaluation episode
        episode_data = self._run_episode(agents, env, train=False)
        
        return {
            'coverage_rate': episode_data['final_coverage_rate'],
            'energy_efficiency': episode_data.get('energy_efficiency', 0),
            'collision_rate': episode_data.get('collision_rate', 0),
            'total_reward': episode_data['total_reward']
        }
    
    def _compute_energy_efficiency(self, energy_history: List[Dict], coverage_rate: float) -> float:
        """Compute energy efficiency metric"""
        if not energy_history:
            return 0.0
        
        # Calculate total energy consumed by UAVs
        initial_energy = sum(energy for energy in energy_history[0].values())
        final_energy = sum(energy for energy in energy_history[-1].values())
        energy_consumed = max(0, initial_energy - final_energy)
        
        # Energy efficiency = coverage per unit energy
        if energy_consumed == 0:
            return coverage_rate  # Perfect efficiency if no energy consumed
        
        return coverage_rate / (energy_consumed / 1000)  # Normalize energy consumption
    
    def _compute_final_evaluation_stats(self, evaluation_episodes: List[Dict]) -> Dict:
        """Compute final evaluation statistics from multiple episodes"""
        stats = {}
        
        # Extract key metrics from all episodes
        coverage_rates = []
        energy_efficiencies = []
        
        for episode in evaluation_episodes:
            coverage_rates.append(episode.get('coverage_rate', {}).get('final_coverage_rate', 0))
            energy_efficiencies.append(episode.get('energy_efficiency', {}).get('energy_efficiency', 0))
        
        # Compute statistics
        stats['avg_coverage_rate'] = np.mean(coverage_rates)
        stats['std_coverage_rate'] = np.std(coverage_rates)
        stats['max_coverage_rate'] = np.max(coverage_rates)
        stats['min_coverage_rate'] = np.min(coverage_rates)
        
        stats['avg_energy_efficiency'] = np.mean(energy_efficiencies)
        stats['std_energy_efficiency'] = np.std(energy_efficiencies)
        stats['max_energy_efficiency'] = np.max(energy_efficiencies)
        stats['min_energy_efficiency'] = np.min(energy_efficiencies)
        
        return stats
    
    def _save_config(self):
        """Save experiment configuration"""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def _save_algorithm_results(self, algorithm: str, results: Dict):
        """Save results for specific algorithm"""
        results_path = os.path.join(self.experiment_dir, f'{algorithm}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _generate_comparison_report(self, results: Dict):
        """Generate comprehensive comparison report"""
        report_path = os.path.join(self.experiment_dir, 'comparison_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# SAGIN Coverage Optimization - Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            
            # Create summary table
            f.write("| Algorithm | Avg Coverage Rate | Avg Energy Efficiency | Max Coverage |\n")
            f.write("|-----------|-------------------|----------------------|-------------|\n")
            
            for algorithm, result in results.items():
                final_metrics = result['evaluation_stats']['final_metrics']
                avg_coverage = final_metrics.get('avg_coverage_rate', 0)
                avg_energy = final_metrics.get('avg_energy_efficiency', 0)
                max_coverage = final_metrics.get('max_coverage_rate', 0)
                
                f.write(f"| {algorithm} | {avg_coverage:.3f} | {avg_energy:.3f} | {max_coverage:.3f} |\n")
            
            # Add detailed analysis for each algorithm
            for algorithm, result in results.items():
                f.write(f"\n## {algorithm.upper()} Results\n\n")
                
                training_stats = result['training_stats']
                evaluation_stats = result['evaluation_stats']['final_metrics']
                
                f.write(f"- Final Training Coverage Rate: {training_stats['coverage_rates'][-1]:.3f}\n")
                f.write(f"- Average Evaluation Coverage Rate: {evaluation_stats['avg_coverage_rate']:.3f} ± {evaluation_stats['std_coverage_rate']:.3f}\n")
                f.write(f"- Average Energy Efficiency: {evaluation_stats['avg_energy_efficiency']:.3f}\n")
                f.write(f"- Best Coverage Achieved: {evaluation_stats['max_coverage_rate']:.3f}\n")
        
        print(f"Comparison report saved to: {report_path}")
    
    def _generate_ablation_report(self, ablation_results: Dict):
        """Generate ablation study report"""
        report_path = os.path.join(self.experiment_dir, 'ablation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Ablation Study Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration Results\n\n")
            f.write("| Configuration | Coverage Rate | Energy Efficiency | Attention Components |\n")
            f.write("|---------------|---------------|-------------------|---------------------|\n")
            
            for config_name, result in ablation_results.items():
                final_metrics = result['evaluation_stats']['final_metrics']
                attention_config = result['attention_config']
                
                avg_coverage = final_metrics.get('avg_coverage_rate', 0)
                avg_energy = final_metrics.get('avg_energy_efficiency', 0)
                
                components = []
                if attention_config.get('use_spatial', False):
                    components.append('Spatial')
                if attention_config.get('use_agent', False):
                    components.append('Agent')
                if attention_config.get('use_task', False):
                    components.append('Task')
                
                components_str = ', '.join(components) if components else 'None'
                
                f.write(f"| {config_name} | {avg_coverage:.3f} | {avg_energy:.3f} | {components_str} |\n")
        
        print(f"Ablation report saved to: {report_path}")
    
    def _generate_scalability_report(self, scalability_results: Dict):
        """Generate scalability study report"""
        report_path = os.path.join(self.experiment_dir, 'scalability_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Scalability Study Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for scale_name, scale_result in scalability_results.items():
                f.write(f"## {scale_name.title()} Scale\n\n")
                
                config = scale_result['config']
                f.write(f"- Area Size: {config['area_size']}x{config['area_size']}\n")
                f.write(f"- Number of Agents: {config['num_agents']}\n")
                f.write(f"- Number of POIs: {config['num_pois']}\n\n")
                
                f.write("| Algorithm | Coverage Rate | Energy Efficiency |\n")
                f.write("|-----------|---------------|------------------|\n")
                
                for algorithm, result in scale_result['results'].items():
                    final_metrics = result['evaluation_stats']['final_metrics']
                    avg_coverage = final_metrics.get('avg_coverage_rate', 0)
                    avg_energy = final_metrics.get('avg_energy_efficiency', 0)
                    
                    f.write(f"| {algorithm} | {avg_coverage:.3f} | {avg_energy:.3f} |\n")
                
                f.write("\n")
        
        print(f"Scalability report saved to: {report_path}")


def main():
    """Main function to run experiments"""
    parser = argparse.ArgumentParser(description='SAGIN Coverage Optimization Experiments')
    parser.add_argument('--config', type=str, default='configs/default_config.json', 
                       help='Path to configuration file')
    parser.add_argument('--experiment_type', type=str, default='comparison',
                       choices=['comparison', 'ablation', 'scalability', 'all'],
                       help='Type of experiment to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'seed': args.seed,
            'num_episodes': 100,
            'num_eval_episodes': 20,
            'environment': {
                'area_size': 800,
                'num_agents': 8,
                'num_satellites': 2,
                'num_uavs': 4,
                'num_ground_stations': 2,
                'num_pois': 12,
                'max_episode_steps': 200
            },
            'algorithm': {
                'embed_dim': 256,
                'num_heads': 8,
                'actor_lr': 3e-4,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'tau': 0.005
            },
            'algorithms': ['ae_maddpg', 'baseline_maddpg']
        }
    
    # Override seed if provided
    if args.seed != 42:
        config['seed'] = args.seed
    
    # Initialize experiment
    experiment = CompleteExperiment(config)
    
    # Run specified experiment type
    if args.experiment_type == 'comparison' or args.experiment_type == 'all':
        print("Running comparison study...")
        experiment.run_comparison_study()
    
    if args.experiment_type == 'ablation' or args.experiment_type == 'all':
        print("Running ablation study...")
        experiment.run_ablation_study()
    
    if args.experiment_type == 'scalability' or args.experiment_type == 'all':
        print("Running scalability study...")
        experiment.run_scalability_study()
    
    print(f"\nAll experiments completed. Results saved to: {experiment.experiment_dir}")


if __name__ == "__main__":
    main()