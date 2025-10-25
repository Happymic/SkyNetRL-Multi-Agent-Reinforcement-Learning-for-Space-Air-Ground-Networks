"""
Multi-Algorithm Comparison Framework for SAGIN Networks
Statistical analysis and visualization for algorithm evaluation
"""

import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from environments.enhanced_sagin_env import EnhancedSAGINEnvironment
from environments.channel_model import ThreeGPPChannelModel
from protocols.mac_layer import MACLayer
from protocols.qos_manager import QoSManager
from evaluation.academic_metrics import StandardizedMetrics
from evaluation.statistical_analysis import AcademicStatisticalAnalyzer
from visualization.trajectory_plotter import TrajectoryVisualizer
# Algorithm imports - will be loaded dynamically as needed
# Try to import real algorithms, fall back to mock if not available
try:
    from algorithms.ae_maddpg.ae_maddpg import AEMADDPG
except ImportError:
    AEMADDPG = None
try:
    from algorithms.baseline_maddpg.maddpg import MADDPG
except ImportError:
    MADDPG = None
try:
    from algorithms.qmix.qmix import QMIX
except ImportError:
    QMIX = None
try:
    from algorithms.independent_ppo.ppo import IndependentPPO
except ImportError:
    IndependentPPO = None
try:
    from algorithms.heuristic.greedy import GreedyHeuristic
except ImportError:
    GreedyHeuristic = None
try:
    from algorithms.baseline.random_policy import RandomPolicy
except ImportError:
    RandomPolicy = None
try:
    from algorithms.demo_algorithm.demo_algorithm import DemoAlgorithm
except ImportError:
    DemoAlgorithm = None

# Algorithm imports will be available when implemented


@dataclass
class ExperimentResults:
    """Container for experiment results"""
    algorithm_name: str
    training_rewards: List[float]
    evaluation_metrics: Dict[str, float]
    training_time: float
    convergence_episode: int
    statistical_data: Dict[str, Any]


class ComparisonExperiment:
    """
    Multi-algorithm comparison framework for SAGIN networks
    Provides statistical analysis and visualization capabilities
    """
    
    def __init__(self, config: Dict):
        """Initialize comparison experiment framework"""
        self.config = config
        experiment_name = config.get('experiment_name', 'sagin_multi_algorithm_evaluation')
        self.output_dir = Path(config.get('output_dir', f'results/{experiment_name}'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment with realistic models
        self.env = self._create_enhanced_environment()
        
        # Initialize evaluation components
        self.metrics_evaluator = StandardizedMetrics()
        self.statistical_analyzer = AcademicStatisticalAnalyzer()
        
        # Initialize visualization
        if config.get('enable_visualization', True):
            area_size = config.get('environment', {}).get('area_size', 1000)
            self.visualizer = TrajectoryVisualizer(area_size, str(self.output_dir / "visualizations"))
        else:
            self.visualizer = None
        
        # Algorithm registry - only real implementations
        self.algorithm_registry = {}
        
        # Available algorithms
        available_algorithms = {
            'ae_maddpg': AEMADDPG,
            'baseline_maddpg': MADDPG, 
            'qmix': QMIX,
            'independent_ppo': IndependentPPO,
            'greedy_heuristic': GreedyHeuristic,
            'random_policy': RandomPolicy,
            'demo_algorithm': DemoAlgorithm
        }
        
        for alg_name, alg_class in available_algorithms.items():
            if alg_class is not None:
                self.algorithm_registry[alg_name] = alg_class
                print(f"  Available: {alg_name}")
        
        if not self.algorithm_registry:
            print("  Warning: No algorithm implementations found")
            print("  Please implement algorithms in src/algorithms/ directory")
        else:
            print(f"  Loaded {len(self.algorithm_registry)} algorithms")
        
        # Results storage
        self.experiment_results = {}
        
    def _create_enhanced_environment(self) -> EnhancedSAGINEnvironment:
        """Create environment with realistic channel and protocol models"""
        # Initialize channel model
        channel_config = self.config.get('channel_model', {})
        channel_model = ThreeGPPChannelModel(channel_config)
        
        # Initialize protocol layers
        mac_config = self.config.get('mac_layer', {})
        mac_layer = MACLayer(mac_config)
        
        qos_config = self.config.get('qos_manager', {})
        qos_manager = QoSManager(qos_config)
        
        # Create enhanced environment
        env_config = self.config.get('environment', {})
        env_config.update({
            'enable_3gpp_channels': self.config.get('enable_3gpp_channels', False),
            'enable_mac_protocols': self.config.get('enable_mac_protocols', False),
            'enable_performance_metrics': True
        })
        
        env = EnhancedSAGINEnvironment(env_config)
        
        # Inject realistic models
        env.set_channel_model(channel_model)
        env.set_mac_layer(mac_layer)
        env.set_qos_manager(qos_manager)
        
        return env
    
    def run_single_algorithm(self, algorithm_name: str, num_runs: int = 5) -> ExperimentResults:
        """
        Run single algorithm with multiple seeds for statistical validity
        
        Args:
            algorithm_name: Name of algorithm to evaluate
            num_runs: Number of independent runs for statistical analysis
            
        Returns:
            ExperimentResults: Comprehensive results including statistics
        """
        # Check if algorithm is available
        if not self.algorithm_registry:
            print("\n❌ No algorithms available!")
            print("The framework is ready for algorithm implementations.")
            print("\nTo implement an algorithm:")
            print("1. Create src/algorithms/{algorithm_name}/{algorithm_name}.py")
            print("2. Implement the required interface (see README.md)")
            print("3. The algorithm will be automatically detected")
            raise ValueError("No algorithms implemented")
        
        if algorithm_name not in self.algorithm_registry:
            print(f"\n❌ Algorithm '{algorithm_name}' not implemented!")
            print(f"Available algorithms: {list(self.algorithm_registry.keys())}")
            print("\nTo implement this algorithm, see the README.md implementation guide.")
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        
        print(f"Running {algorithm_name} with {num_runs} independent runs...")
        
        all_rewards = []
        all_metrics = []
        training_times = []
        convergence_episodes = []
        
        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs}")
            
            # Set unique random seed for each algorithm and run
            base_seed = self.config.get('base_seed', 42)
            algorithm_hash = hash(algorithm_name) % 1000  # Get unique hash per algorithm
            seed = base_seed + algorithm_hash * 100 + run_idx
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"    Using seed: {seed}")
            
            # Initialize algorithm
            algorithm = self._initialize_algorithm(algorithm_name, seed)
            
            # Training phase
            start_time = time.time()
            training_rewards, convergence_ep = self._train_algorithm(algorithm, seed)
            training_time = time.time() - start_time
            
            # Evaluation phase
            eval_metrics = self._evaluate_algorithm(algorithm)
            
            # Store results
            all_rewards.append(training_rewards)
            all_metrics.append(eval_metrics)
            training_times.append(training_time)
            convergence_episodes.append(convergence_ep)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # Statistical analysis
        statistical_data = {
            'num_runs': num_runs,
            'mean_final_reward': np.mean([rewards[-1] for rewards in all_rewards]),
            'std_final_reward': np.std([rewards[-1] for rewards in all_rewards]),
            'mean_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times),
            'mean_convergence_episode': np.mean(convergence_episodes),
            'std_convergence_episode': np.std(convergence_episodes),
            'confidence_intervals': self._compute_confidence_intervals(all_metrics)
        }
        
        return ExperimentResults(
            algorithm_name=algorithm_name,
            training_rewards=np.mean(all_rewards, axis=0).tolist(),
            evaluation_metrics=aggregated_metrics,
            training_time=np.mean(training_times),
            convergence_episode=int(np.mean(convergence_episodes)),
            statistical_data=statistical_data
        )
    
    def run_complete_comparison(self) -> Dict[str, ExperimentResults]:
        """
        Run complete multi-algorithm comparison study
        
        Returns:
            Dict mapping algorithm names to their results
        """
        # Check if any algorithms are available
        if not self.algorithm_registry:
            print("\n❌ No algorithms available!")
            print("The framework is ready for algorithm implementations.")
            print("\nTo implement an algorithm:")
            print("1. Create src/algorithms/{algorithm_name}/{algorithm_name}.py")
            print("2. Implement the required interface (see README.md)")
            print("3. The algorithm will be automatically detected")
            return {}
        
        algorithms_to_test = self.config.get('algorithms', list(self.algorithm_registry.keys()))
        # Filter to only implemented algorithms
        available_algorithms = [alg for alg in algorithms_to_test if alg in self.algorithm_registry]
        
        if not available_algorithms:
            print(f"\n❌ None of the requested algorithms are implemented!")
            print(f"Requested: {', '.join(algorithms_to_test)}")
            print(f"Available: {list(self.algorithm_registry.keys())}")
            print("\nTo implement algorithms, see the README.md implementation guide.")
            return {}
        
        num_runs = self.config.get('num_statistical_runs', 5)
        
        print(f"Running complete comparison of {len(available_algorithms)} algorithms...")
        print(f"Algorithms: {', '.join(available_algorithms)}")
        print(f"Independent runs per algorithm: {num_runs}")
        
        results = {}
        
        for algorithm_name in available_algorithms:
                
            results[algorithm_name] = self.run_single_algorithm(algorithm_name, num_runs)
            
            # Save intermediate results
            self._save_intermediate_results(algorithm_name, results[algorithm_name])
        
        # Perform statistical comparison
        self._perform_statistical_comparison(results)
        
        # Generate report
        self._generate_comparison_report(results)
        
        # Generate visualizations if enabled
        if self.visualizer and results:
            print("Generating visualization summary...")
            for algorithm_name in results.keys():
                self.visualizer.generate_summary_report(algorithm_name)
        
        return results
    
    def _initialize_algorithm(self, algorithm_name: str, seed: int):
        """Initialize algorithm with configuration"""
        if algorithm_name not in self.algorithm_registry:
            raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {list(self.algorithm_registry.keys())}")
        
        algorithm_class = self.algorithm_registry[algorithm_name]
        
        # Get algorithm-specific configuration
        algo_config = self.config.get('algorithms_config', {}).get(algorithm_name, {})
        algo_config.update({
            'seed': seed,
            'obs_dim': self.env.observation_space.shape[0],
            'action_dim': self.env.action_space.shape[0],
            'num_agents': self.env.num_agents
        })
        
        return algorithm_class(algo_config)
    
    def _train_algorithm(self, algorithm, seed: int) -> Tuple[List[float], int]:
        """Train algorithm and return rewards and convergence episode"""
        num_episodes = self.config.get('num_episodes', 1000)
        convergence_threshold = self.config.get('convergence_threshold', -1000)
        convergence_window = self.config.get('convergence_window', 100)
        
        episode_rewards = []
        convergence_episode = num_episodes
        
        # Initialize visualization for this training run
        if self.visualizer:
            self.visualizer.reset()
        
        for episode in range(num_episodes):
            # Environment training - use different seed for each episode
            episode_seed = seed + episode * 10000
            obs = self.env.reset(seed=episode_seed)
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                actions = algorithm.act(obs)
                next_obs, rewards, dones_dict, info = self.env.step(actions)
                
                # Convert done dict to single boolean (episode ends when any agent is done)
                done = any(dones_dict.values()) if isinstance(dones_dict, dict) else dones_dict
                
                # Record visualization data
                if self.visualizer and step % 5 == 0:  # Record every 5 steps
                    agents = {i: {'position': obs[i][:2], 'type': 'agent', 'energy': 1.0, 'active': True} 
                             for i in range(len(obs))}
                    pois = [type('POI', (), {'x': 100, 'y': 100, 'covered': False, 'priority': 1})]
                    # Handle reward format (dict or array)
                    if isinstance(rewards, dict):
                        mean_reward = np.mean(list(rewards.values()))
                    else:
                        mean_reward = np.mean(rewards)
                    self.visualizer.record_step(agents, pois, {'reward': mean_reward}, step)
                
                # Store experience and update if applicable
                if hasattr(algorithm, 'store_experience'):
                    algorithm.store_experience(obs, actions, rewards, next_obs, dones_dict)
                
                # Update algorithm (allow learning from episode 0 for short runs)
                learning_start = min(self.config.get('learning_start', 0), num_episodes // 10)
                if hasattr(algorithm, 'update') and episode >= learning_start:
                    algorithm.update()
                
                obs = next_obs
                # Handle reward format (dict or array)
                if isinstance(rewards, dict):
                    episode_reward += np.mean(list(rewards.values()))
                else:
                    episode_reward += np.mean(rewards)
                step += 1
            
            episode_rewards.append(episode_reward)
            
            # Check convergence
            if (episode >= convergence_window and 
                np.mean(episode_rewards[-convergence_window:]) > convergence_threshold and
                convergence_episode == num_episodes):
                convergence_episode = episode
            
            # Progress logging - show progress more frequently for shorter runs
            log_interval = max(1, min(100, num_episodes // 10))
            if episode % log_interval == 0 or episode == num_episodes - 1:
                window = min(episode + 1, 100)
                avg_reward = np.mean(episode_rewards[-window:])
                print(f"    Episode {episode}, Avg Reward: {avg_reward:.2f}, Steps: {step}")
        
        return episode_rewards, convergence_episode
    
    def _evaluate_algorithm(self, algorithm) -> Dict[str, float]:
        """Evaluate algorithm on academic metrics"""
        num_eval_episodes = self.config.get('num_eval_episodes', 100)
        
        # Set algorithm to evaluation mode
        if hasattr(algorithm, 'set_eval_mode'):
            algorithm.set_eval_mode()
        
        # Real algorithm evaluation
        metrics_data = {
            'rewards': [],
            'coverage_events': [],
            'throughput_values': [],
            'energy_consumption': [],
            'latency_values': [],
            'packet_loss_rates': [],
            'spectral_efficiency': [],
            'fairness_indices': []
        }
        
        for eval_episode in range(num_eval_episodes):
            eval_seed = hash(algorithm.__class__.__name__) % 10000 + eval_episode * 1000
            obs = self.env.reset(seed=eval_seed)
            episode_metrics = {key: [] for key in metrics_data.keys()}
            done = False
            
            while not done:
                actions = algorithm.act(obs, add_noise=False)
                obs, rewards, dones_dict, info = self.env.step(actions)
                
                # Convert done dict to single boolean
                done = any(dones_dict.values()) if isinstance(dones_dict, dict) else dones_dict
                
                # Extract metrics from environment info
                # Handle reward format (dict or array)
                if isinstance(rewards, dict):
                    episode_metrics['rewards'].append(np.mean(list(rewards.values())))
                else:
                    episode_metrics['rewards'].append(np.mean(rewards))
                
                if 'coverage_events' in info:
                    episode_metrics['coverage_events'].extend(info['coverage_events'])
                if 'throughput' in info:
                    episode_metrics['throughput_values'].append(info['throughput'])
                if 'energy_consumption' in info:
                    episode_metrics['energy_consumption'].append(info['energy_consumption'])
                if 'latency' in info:
                    episode_metrics['latency_values'].append(info['latency'])
                if 'packet_loss_rate' in info:
                    episode_metrics['packet_loss_rates'].append(info['packet_loss_rate'])
                if 'spectral_efficiency' in info:
                    episode_metrics['spectral_efficiency'].append(info['spectral_efficiency'])
                if 'fairness_index' in info:
                    episode_metrics['fairness_indices'].append(info['fairness_index'])
            
            # Aggregate episode metrics
            for key, values in episode_metrics.items():
                if values:
                    metrics_data[key].extend(values)
        
        # Compute standardized metrics
        return self.metrics_evaluator.compute_all_metrics(metrics_data)
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple runs"""
        if not all_metrics:
            return {}
        
        aggregated = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in all_metrics]
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_median'] = np.median(values)
        
        return aggregated
    
    def _compute_confidence_intervals(self, all_metrics: List[Dict[str, float]], confidence=0.95) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for metrics"""
        from scipy import stats
        
        confidence_intervals = {}
        if not all_metrics:
            return confidence_intervals
        
        metric_names = all_metrics[0].keys()
        alpha = 1 - confidence
        
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in all_metrics]
            mean = np.mean(values)
            sem = stats.sem(values)
            ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
            confidence_intervals[metric_name] = ci
        
        return confidence_intervals
    
    def _perform_statistical_comparison(self, results: Dict[str, ExperimentResults]):
        """Perform statistical significance testing between algorithms"""
        print("\nPerforming statistical comparison...")
        
        comparison_results = {}
        algorithm_names = list(results.keys())
        
        for i, algo1 in enumerate(algorithm_names):
            for j, algo2 in enumerate(algorithm_names):
                if i < j:  # Avoid duplicate comparisons
                    comparison_key = f"{algo1}_vs_{algo2}"
                    
                    # Extract final rewards for comparison
                    rewards1 = [results[algo1].training_rewards[-1]] * results[algo1].statistical_data['num_runs']
                    rewards2 = [results[algo2].training_rewards[-1]] * results[algo2].statistical_data['num_runs']
                    
                    # Perform statistical test
                    comparison_result = self.statistical_analyzer.compare_two_algorithms(rewards1, rewards2)
                    comparison_results[comparison_key] = comparison_result
                    
                    print(f"  {algo1} vs {algo2}: p-value = {comparison_result.p_value:.4f}, "
                          f"effect size = {comparison_result.effect_size:.4f}")
        
        # Save comparison results
        self._save_comparison_results(comparison_results)
    
    def _generate_comparison_report(self, results: Dict[str, ExperimentResults]):
        """Generate comprehensive academic report"""
        report_path = self.output_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# SAGIN Network Multi-Algorithm Comparison Report\n\n")
            f.write("## Experimental Setup\n\n")
            f.write(f"- Number of algorithms tested: {len(results)}\n")
            
            if results:
                f.write(f"- Statistical runs per algorithm: {list(results.values())[0].statistical_data['num_runs']}\n")
            else:
                f.write("- Statistical runs per algorithm: N/A (no algorithms executed)\n")
                
            f.write(f"- Training episodes: {self.config.get('num_episodes', 1000)}\n")
            f.write(f"- Evaluation episodes: {self.config.get('num_eval_episodes', 100)}\n\n")
            
            f.write("## Results Summary\n\n")
            
            if results:
                f.write("| Algorithm | Mean Final Reward | Std | Training Time (s) | Convergence Episode |\n")
                f.write("|-----------|-------------------|-----|-------------------|--------------------|\n")
                
                for algo_name, result in results.items():
                    f.write(f"| {algo_name} | {result.statistical_data['mean_final_reward']:.3f} | "
                           f"{result.statistical_data['std_final_reward']:.3f} | "
                           f"{result.training_time:.1f} | {result.convergence_episode} |\n")
                
                f.write("\n## Detailed Metrics\n\n")
                for algo_name, result in results.items():
                    f.write(f"### {algo_name}\n\n")
                    for metric_name, value in result.evaluation_metrics.items():
                        f.write(f"- {metric_name}: {value:.4f}\n")
                    f.write("\n")
            else:
                f.write("No algorithms were successfully executed.\n\n")
                f.write("**Possible Issues:**\n")
                f.write("- Algorithm implementations not found\n")
                f.write("- Missing dependencies\n")
                f.write("- Configuration errors\n\n")
        
        print(f"Evaluation report saved to: {report_path}")
    
    def _save_intermediate_results(self, algorithm_name: str, results: ExperimentResults):
        """Save intermediate results for individual algorithm"""
        results_path = self.output_dir / f"{algorithm_name}_results.json"
        
        # Convert results to serializable format
        results_dict = {
            'algorithm_name': results.algorithm_name,
            'training_rewards': results.training_rewards,
            'evaluation_metrics': results.evaluation_metrics,
            'training_time': results.training_time,
            'convergence_episode': results.convergence_episode,
            'statistical_data': results.statistical_data
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def _save_comparison_results(self, comparison_results: Dict):
        """Save statistical comparison results"""
        comparison_path = self.output_dir / "statistical_comparisons.json"
        
        # Convert to serializable format
        serializable_results = {}
        for key, result in comparison_results.items():
            serializable_results[key] = {
                'p_value': float(result.p_value),
                'effect_size': float(result.effect_size),
                'significantly_different': bool(result.significantly_different),
                'test_statistic': float(result.test_statistic),
                'algorithm_1': str(result.algorithm_1),
                'algorithm_2': str(result.algorithm_2),
                'test_type': str(result.test_type),
                'interpretation': str(result.interpretation)
            }
        
        with open(comparison_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Statistical comparisons saved to: {comparison_path}")