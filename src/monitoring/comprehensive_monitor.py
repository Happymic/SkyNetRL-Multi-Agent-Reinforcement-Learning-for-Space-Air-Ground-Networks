"""
Comprehensive Training Monitoring System
Integrates all optimization components with detailed tracking and analysis
"""

import numpy as np
import torch
import json
import time
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import warnings
from pathlib import Path


@dataclass
class MonitoringConfig:
    """Configuration for comprehensive monitoring"""
    
    # Logging settings
    log_interval: int = 10
    save_interval: int = 100
    plot_interval: int = 50
    
    # Metrics tracking
    track_gradients: bool = True
    track_attention_weights: bool = True
    track_reward_breakdown: bool = True
    track_network_stats: bool = True
    
    # Storage settings
    max_history_length: int = 10000
    save_raw_data: bool = True
    save_plots: bool = True
    
    # Analysis settings
    window_size: int = 100
    trend_analysis_window: int = 500
    anomaly_detection_threshold: float = 3.0
    
    # Output settings
    output_dir: str = "monitoring_outputs"
    experiment_name: str = "comprehensive_experiment"


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    episode: int
    timestamp: float
    
    # Performance metrics
    total_reward: float
    coverage_rate: float
    energy_efficiency: float
    collision_count: int
    
    # Learning metrics
    total_loss: float
    gradient_norm: float
    learning_rate: float
    
    # Environment metrics
    scenario_difficulty: float
    communication_efficiency: float
    
    # Reward breakdown
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Attention analysis
    attention_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Training stability
    stability_metrics: Dict[str, Any] = field(default_factory=dict)


class MetricsBuffer:
    """Circular buffer for storing metrics with efficient operations"""
    
    def __init__(self, maxlen: int = 10000):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        self.arrays = {}  # Cached numpy arrays for fast operations
        self.dirty = True  # Flag to indicate if arrays need updating
    
    def append(self, item: EpisodeMetrics):
        """Add new metrics"""
        self.data.append(item)
        self.dirty = True
    
    def get_recent(self, n: int) -> List[EpisodeMetrics]:
        """Get most recent n items"""
        return list(self.data)[-n:] if n <= len(self.data) else list(self.data)
    
    def get_array(self, attribute: str) -> np.ndarray:
        """Get numpy array of specific attribute"""
        if self.dirty:
            self._update_arrays()
        
        return self.arrays.get(attribute, np.array([]))
    
    def _update_arrays(self):
        """Update cached numpy arrays"""
        if not self.data:
            self.arrays = {}
            self.dirty = False
            return
        
        # Basic metrics
        self.arrays['episode'] = np.array([item.episode for item in self.data])
        self.arrays['timestamp'] = np.array([item.timestamp for item in self.data])
        self.arrays['total_reward'] = np.array([item.total_reward for item in self.data])
        self.arrays['coverage_rate'] = np.array([item.coverage_rate for item in self.data])
        self.arrays['energy_efficiency'] = np.array([item.energy_efficiency for item in self.data])
        self.arrays['collision_count'] = np.array([item.collision_count for item in self.data])
        self.arrays['total_loss'] = np.array([item.total_loss for item in self.data])
        self.arrays['gradient_norm'] = np.array([item.gradient_norm for item in self.data])
        self.arrays['learning_rate'] = np.array([item.learning_rate for item in self.data])
        self.arrays['scenario_difficulty'] = np.array([item.scenario_difficulty for item in self.data])
        
        # Reward breakdown arrays
        if self.data[0].reward_breakdown:
            for key in self.data[0].reward_breakdown.keys():
                self.arrays[f'reward_{key}'] = np.array([
                    item.reward_breakdown.get(key, 0.0) for item in self.data
                ])
        
        # Attention analysis arrays
        if self.data[0].attention_analysis:
            for key in self.data[0].attention_analysis.keys():
                self.arrays[f'attention_{key}'] = np.array([
                    item.attention_analysis.get(key, 0.0) for item in self.data
                ])
        
        self.dirty = False
    
    def __len__(self):
        return len(self.data)


class ComprehensiveMonitor:
    """Main monitoring system integrating all components"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.start_time = time.time()
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_buffer = MetricsBuffer(config.max_history_length)
        
        # Real-time tracking
        self.current_episode = 0
        self.last_log_time = time.time()
        self.last_save_time = time.time()
        self.last_plot_time = time.time()
        
        # Analysis caches
        self.trend_analysis_cache = {}
        self.anomaly_cache = {}
        
        # Performance tracking
        self.best_performance = {
            'reward': float('-inf'),
            'coverage': 0.0,
            'efficiency': 0.0,
            'episode': 0
        }
        
        # Integration points
        self.reward_system = None
        self.attention_network = None
        self.stability_system = None
        self.output_manager = None  # Will be set by factory function
        
        # Logging setup
        self.log_file = self.output_dir / "training_log.json"
        self.csv_file = self.output_dir / "metrics.csv"
        
        print(f"📊 Comprehensive Monitor initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def integrate_systems(
        self,
        reward_system=None,
        attention_network=None,
        stability_system=None
    ):
        """Integrate with other optimization systems"""
        self.reward_system = reward_system
        self.attention_network = attention_network
        self.stability_system = stability_system
        
        print("🔗 Integrated with optimization systems:")
        if reward_system:
            print("   🎯 Multi-objective reward system")
        if attention_network:
            print("   🧠 Hierarchical attention network")
        if stability_system:
            print("   🛡️ Training stability system")
    
    def record_episode(
        self,
        episode: int,
        total_reward: float,
        environment_info: Dict[str, Any],
        training_info: Dict[str, Any]
    ):
        """Record metrics for a completed episode"""
        
        self.current_episode = episode
        
        # Extract basic metrics
        metrics = EpisodeMetrics(
            episode=episode,
            timestamp=time.time(),
            total_reward=total_reward,
            coverage_rate=environment_info.get('coverage_rate', 0.0),
            energy_efficiency=training_info.get('energy_efficiency', 0.0),
            collision_count=environment_info.get('collisions', 0),
            total_loss=training_info.get('total_loss', 0.0),
            gradient_norm=training_info.get('gradient_norm', 0.0),
            learning_rate=training_info.get('learning_rate', 0.0),
            scenario_difficulty=environment_info.get('scenario_difficulty', 0.0),
            communication_efficiency=environment_info.get('communication_efficiency', 0.0)
        )
        
        # Extract reward breakdown from integrated systems
        if self.reward_system and 'reward_breakdown' in environment_info:
            metrics.reward_breakdown = environment_info['reward_breakdown']
        
        # Extract attention analysis
        if self.attention_network and hasattr(self.attention_network, 'get_attention_analysis'):
            try:
                metrics.attention_analysis = self.attention_network.get_attention_analysis()
            except Exception as e:
                print(f"⚠️ Could not get attention analysis: {e}")
        
        # Extract stability metrics
        if self.stability_system and hasattr(self.stability_system, 'get_stability_metrics'):
            try:
                metrics.stability_metrics = self.stability_system.get_stability_metrics()
            except Exception as e:
                print(f"⚠️ Could not get stability metrics: {e}")
        
        # Store metrics
        self.metrics_buffer.append(metrics)
        
        # Update best performance tracking
        self._update_best_performance(metrics)
        
        # Periodic operations
        current_time = time.time()
        
        if episode % self.config.log_interval == 0:
            self._log_progress(metrics)
        
        if current_time - self.last_save_time > self.config.save_interval:
            self._save_data()
            self.last_save_time = current_time
        
        if episode % self.config.plot_interval == 0:
            self._generate_plots()
            self.last_plot_time = current_time
        
        # Integrate with output manager if available
        if self.output_manager:
            # Convert metrics to dict format for output manager
            metrics_dict = {
                'total_reward': metrics.total_reward,
                'coverage_rate': metrics.coverage_rate,
                'energy_efficiency': metrics.energy_efficiency,
                'collision_count': metrics.collision_count,
                'total_loss': training_info.get('total_loss', 0.0),
                'gradient_norm': training_info.get('gradient_norm', 0.0),
                'learning_rate': training_info.get('learning_rate', 0.0)
            }
            
            # Add reward breakdown if available
            if hasattr(metrics, 'reward_breakdown') and metrics.reward_breakdown:
                for key, value in metrics.reward_breakdown.items():
                    metrics_dict[f'reward_{key}'] = value
            
            # Add stability metrics if available
            if hasattr(metrics, 'stability_metrics') and metrics.stability_metrics:
                for key, value in metrics.stability_metrics.items():
                    metrics_dict[f'stability_{key}'] = value
            
            # Record to output manager
            self.output_manager.record_episode_metrics(episode, metrics_dict)
    
    def _update_best_performance(self, metrics: EpisodeMetrics):
        """Update best performance tracking"""
        
        if metrics.total_reward > self.best_performance['reward']:
            self.best_performance['reward'] = metrics.total_reward
            self.best_performance['episode'] = metrics.episode
        
        if metrics.coverage_rate > self.best_performance['coverage']:
            self.best_performance['coverage'] = metrics.coverage_rate
        
        if metrics.energy_efficiency > self.best_performance['efficiency']:
            self.best_performance['efficiency'] = metrics.energy_efficiency
    
    def _log_progress(self, metrics: EpisodeMetrics):
        """Log training progress"""
        
        elapsed_time = time.time() - self.start_time
        eps_per_sec = self.current_episode / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate moving averages
        recent_rewards = self.metrics_buffer.get_array('total_reward')[-50:]
        recent_coverage = self.metrics_buffer.get_array('coverage_rate')[-50:]
        
        avg_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else metrics.total_reward
        avg_coverage = np.mean(recent_coverage) if len(recent_coverage) > 0 else metrics.coverage_rate
        
        # Progress message
        print(f"📊 Episode {metrics.episode:4d} | "
              f"Reward: {metrics.total_reward:7.1f} (avg: {avg_reward:7.1f}) | "
              f"Coverage: {metrics.coverage_rate:5.1f}% (avg: {avg_coverage:5.1f}%) | "
              f"Loss: {metrics.total_loss:6.3f} | "
              f"EPS: {eps_per_sec:5.1f}")
        
        # Additional details for integrated systems
        if metrics.reward_breakdown:
            breakdown_str = " | ".join([f"{k}: {v:.1f}" for k, v in metrics.reward_breakdown.items()])
            print(f"   🎯 Rewards: {breakdown_str}")
        
        if metrics.attention_analysis:
            attention_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.attention_analysis.items()])
            print(f"   🧠 Attention: {attention_str}")
        
        if metrics.stability_metrics and metrics.stability_metrics.get('training_stable') is not None:
            stability = "Stable" if metrics.stability_metrics['training_stable'] else "Unstable"
            noise = metrics.stability_metrics.get('current_noise_level', 0.0)
            print(f"   🛡️ Training: {stability} | Noise: {noise:.3f}")
    
    def _save_data(self):
        """Save metrics data to files"""
        
        if len(self.metrics_buffer) == 0:
            return
        
        # Save to JSON (recent data)
        recent_data = []
        for metrics in self.metrics_buffer.get_recent(1000):  # Last 1000 episodes
            recent_data.append(asdict(metrics))
        
        with open(self.log_file, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'best_performance': self.best_performance,
                'metrics': recent_data
            }, f, indent=2, default=str)
        
        # Save to CSV (all data)
        if self.config.save_raw_data:
            try:
                # Prepare DataFrame
                data = []
                for metrics in self.metrics_buffer.data:
                    row = {
                        'episode': metrics.episode,
                        'timestamp': metrics.timestamp,
                        'total_reward': metrics.total_reward,
                        'coverage_rate': metrics.coverage_rate,
                        'energy_efficiency': metrics.energy_efficiency,
                        'collision_count': metrics.collision_count,
                        'total_loss': metrics.total_loss,
                        'gradient_norm': metrics.gradient_norm,
                        'learning_rate': metrics.learning_rate,
                        'scenario_difficulty': metrics.scenario_difficulty,
                        'communication_efficiency': metrics.communication_efficiency
                    }
                    
                    # Add reward breakdown
                    for key, value in metrics.reward_breakdown.items():
                        row[f'reward_{key}'] = value
                    
                    # Add attention analysis
                    for key, value in metrics.attention_analysis.items():
                        row[f'attention_{key}'] = value
                    
                    # Add stability metrics
                    if isinstance(metrics.stability_metrics, dict):
                        for key, value in metrics.stability_metrics.items():
                            if isinstance(value, (int, float, bool)):
                                row[f'stability_{key}'] = value
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_csv(self.csv_file, index=False)
                
            except Exception as e:
                print(f"⚠️ Could not save CSV data: {e}")
    
    def _generate_plots(self):
        """Generate comprehensive training plots"""
        
        if len(self.metrics_buffer) < 10 or not self.config.save_plots:
            return
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'Training Progress - Episode {self.current_episode}', fontsize=16)
            
            # Plot 1: Rewards over time
            episodes = self.metrics_buffer.get_array('episode')
            rewards = self.metrics_buffer.get_array('total_reward')
            axes[0, 0].plot(episodes, rewards, alpha=0.6, color='blue')
            if len(rewards) > 50:
                # Moving average
                window = min(50, len(rewards) // 4)
                ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(episodes[window-1:], ma_rewards, color='red', linewidth=2)
            axes[0, 0].set_title('Total Reward')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Coverage rate
            coverage = self.metrics_buffer.get_array('coverage_rate')
            axes[0, 1].plot(episodes, coverage, color='green')
            axes[0, 1].set_title('Coverage Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Coverage %')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Energy efficiency
            efficiency = self.metrics_buffer.get_array('energy_efficiency')
            axes[0, 2].plot(episodes, efficiency, color='orange')
            axes[0, 2].set_title('Energy Efficiency')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Efficiency')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Training loss
            losses = self.metrics_buffer.get_array('total_loss')
            axes[1, 0].semilogy(episodes, np.maximum(losses, 1e-8), color='red')
            axes[1, 0].set_title('Training Loss (log scale)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Gradient norms
            grad_norms = self.metrics_buffer.get_array('gradient_norm')
            axes[1, 1].semilogy(episodes, np.maximum(grad_norms, 1e-8), color='purple')
            axes[1, 1].set_title('Gradient Norm (log scale)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Grad Norm')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Learning rate
            lrs = self.metrics_buffer.get_array('learning_rate')
            axes[1, 2].semilogy(episodes, np.maximum(lrs, 1e-8), color='brown')
            axes[1, 2].set_title('Learning Rate (log scale)')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Plot 7: Reward breakdown (if available)
            if hasattr(self.metrics_buffer.data[0], 'reward_breakdown') and self.metrics_buffer.data[0].reward_breakdown:
                reward_keys = list(self.metrics_buffer.data[0].reward_breakdown.keys())[:5]  # Top 5 components
                for i, key in enumerate(reward_keys):
                    reward_data = self.metrics_buffer.get_array(f'reward_{key}')
                    if len(reward_data) > 0:
                        axes[2, 0].plot(episodes, reward_data, label=key, alpha=0.8)
                axes[2, 0].set_title('Reward Breakdown')
                axes[2, 0].set_xlabel('Episode')
                axes[2, 0].set_ylabel('Reward Component')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)
            else:
                axes[2, 0].text(0.5, 0.5, 'No reward breakdown data', 
                               ha='center', va='center', transform=axes[2, 0].transAxes)
                axes[2, 0].set_title('Reward Breakdown')
            
            # Plot 8: Scenario difficulty
            difficulty = self.metrics_buffer.get_array('scenario_difficulty')
            axes[2, 1].plot(episodes, difficulty, color='gray')
            axes[2, 1].set_title('Scenario Difficulty')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Difficulty')
            axes[2, 1].grid(True, alpha=0.3)
            
            # Plot 9: Performance summary
            # Recent performance trends
            if len(rewards) > 100:
                recent_episodes = episodes[-100:]
                recent_rewards = rewards[-100:]
                recent_coverage = coverage[-100:]
                
                ax9 = axes[2, 2]
                ax9_twin = ax9.twinx()
                
                line1 = ax9.plot(recent_episodes, recent_rewards, 'b-', label='Reward')
                line2 = ax9_twin.plot(recent_episodes, recent_coverage, 'g-', label='Coverage %')
                
                ax9.set_title('Recent Performance (Last 100 episodes)')
                ax9.set_xlabel('Episode')
                ax9.set_ylabel('Reward', color='blue')
                ax9_twin.set_ylabel('Coverage %', color='green')
                ax9.grid(True, alpha=0.3)
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax9.legend(lines, labels, loc='upper left')
            else:
                axes[2, 2].text(0.5, 0.5, 'Insufficient data\nfor trend analysis', 
                               ha='center', va='center', transform=axes[2, 2].transAxes)
                axes[2, 2].set_title('Recent Performance')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"training_progress_ep_{self.current_episode}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Keep only recent plots (last 10)
            self._cleanup_old_plots()
            
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
            plt.close('all')
    
    def _cleanup_old_plots(self):
        """Keep only the most recent plot files"""
        try:
            plot_files = list(self.output_dir.glob("training_progress_ep_*.png"))
            plot_files.sort(key=lambda x: int(x.stem.split('_')[-1]))  # Sort by episode number
            
            # Keep only the last 10 plots
            if len(plot_files) > 10:
                for old_plot in plot_files[:-10]:
                    old_plot.unlink()
        except Exception as e:
            print(f"⚠️ Could not cleanup old plots: {e}")
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive training analysis"""
        
        if len(self.metrics_buffer) < 10:
            return {'error': 'Insufficient data for analysis'}
        
        analysis = {
            'training_summary': self._get_training_summary(),
            'performance_analysis': self._get_performance_analysis(),
            'learning_analysis': self._get_learning_analysis(),
            'stability_analysis': self._get_stability_analysis(),
            'trend_analysis': self._get_trend_analysis(),
            'recommendations': self._get_recommendations()
        }
        
        return analysis
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        
        total_episodes = len(self.metrics_buffer)
        elapsed_time = time.time() - self.start_time
        
        rewards = self.metrics_buffer.get_array('total_reward')
        coverage = self.metrics_buffer.get_array('coverage_rate')
        
        return {
            'total_episodes': total_episodes,
            'elapsed_time_minutes': elapsed_time / 60,
            'episodes_per_minute': total_episodes / (elapsed_time / 60) if elapsed_time > 0 else 0,
            'best_performance': self.best_performance,
            'current_performance': {
                'reward_mean': float(np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)),
                'coverage_mean': float(np.mean(coverage[-50:]) if len(coverage) >= 50 else np.mean(coverage)),
                'reward_std': float(np.std(rewards[-50:]) if len(rewards) >= 50 else np.std(rewards))
            }
        }
    
    def _get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance trends and patterns"""
        
        rewards = self.metrics_buffer.get_array('total_reward')
        coverage = self.metrics_buffer.get_array('coverage_rate')
        
        if len(rewards) < 20:
            return {'error': 'Insufficient data'}
        
        # Performance trends
        recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        early_rewards = rewards[:100] if len(rewards) >= 100 else rewards[:len(rewards)//2]
        
        improvement = np.mean(recent_rewards) - np.mean(early_rewards)
        volatility = np.std(recent_rewards) / np.mean(recent_rewards) if np.mean(recent_rewards) != 0 else float('inf')
        
        return {
            'performance_improvement': float(improvement),
            'performance_volatility': float(volatility),
            'convergence_indicator': self._calculate_convergence(rewards),
            'performance_consistency': self._calculate_consistency(rewards),
            'peak_performance': {
                'reward': float(np.max(rewards)),
                'coverage': float(np.max(coverage))
            }
        }
    
    def _get_learning_analysis(self) -> Dict[str, Any]:
        """Analyze learning dynamics"""
        
        losses = self.metrics_buffer.get_array('total_loss')
        grad_norms = self.metrics_buffer.get_array('gradient_norm')
        learning_rates = self.metrics_buffer.get_array('learning_rate')
        
        if len(losses) < 10:
            return {'error': 'Insufficient data'}
        
        return {
            'loss_trend': self._calculate_trend(losses),
            'gradient_stability': {
                'mean_gradient_norm': float(np.mean(grad_norms)),
                'gradient_variance': float(np.var(grad_norms)),
                'gradient_explosions': int(np.sum(grad_norms > 10.0))
            },
            'learning_rate_adaptation': {
                'initial_lr': float(learning_rates[0]) if len(learning_rates) > 0 else 0.0,
                'current_lr': float(learning_rates[-1]) if len(learning_rates) > 0 else 0.0,
                'lr_reductions': int(np.sum(np.diff(learning_rates) < -1e-6))
            }
        }
    
    def _get_stability_analysis(self) -> Dict[str, Any]:
        """Analyze training stability"""
        
        if len(self.metrics_buffer) < 20:
            return {'error': 'Insufficient data'}
        
        # Collect stability indicators from recent episodes
        recent_metrics = self.metrics_buffer.get_recent(100)
        stable_episodes = sum(1 for m in recent_metrics 
                             if m.stability_metrics.get('training_stable', False))
        
        return {
            'stability_rate': stable_episodes / len(recent_metrics),
            'early_stopping_risk': any(m.stability_metrics.get('early_stopping_triggered', False) 
                                     for m in recent_metrics[-10:]),
            'training_health': 'good' if stable_episodes / len(recent_metrics) > 0.8 else 'concerning'
        }
    
    def _get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends across different metrics"""
        
        metrics_to_analyze = ['total_reward', 'coverage_rate', 'energy_efficiency', 'total_loss']
        trends = {}
        
        for metric in metrics_to_analyze:
            data = self.metrics_buffer.get_array(metric)
            if len(data) > 0:
                trends[metric] = {
                    'trend_direction': self._calculate_trend(data),
                    'recent_mean': float(np.mean(data[-20:])) if len(data) >= 20 else float(np.mean(data)),
                    'overall_mean': float(np.mean(data)),
                    'improvement_rate': self._calculate_improvement_rate(data)
                }
        
        return trends
    
    def _get_recommendations(self) -> List[str]:
        """Generate training recommendations based on analysis"""
        
        recommendations = []
        
        if len(self.metrics_buffer) < 50:
            recommendations.append("Continue training to gather more data for analysis")
            return recommendations
        
        # Performance-based recommendations
        rewards = self.metrics_buffer.get_array('total_reward')
        if self._calculate_trend(rewards) < 0:
            recommendations.append("Performance declining - consider reducing learning rate or adjusting reward function")
        
        # Loss-based recommendations
        losses = self.metrics_buffer.get_array('total_loss')
        if np.mean(losses[-20:]) > np.mean(losses[:20:]) * 1.5:
            recommendations.append("Loss increasing - check for overfitting or instability")
        
        # Gradient-based recommendations
        grad_norms = self.metrics_buffer.get_array('gradient_norm')
        if np.mean(grad_norms[-10:]) > 5.0:
            recommendations.append("High gradient norms detected - consider gradient clipping or reducing learning rate")
        
        # Convergence recommendations
        if self._calculate_convergence(rewards) > 0.8:
            recommendations.append("Training appears to be converging - consider early stopping or exploration adjustments")
        
        if not recommendations:
            recommendations.append("Training progressing well - continue with current settings")
        
        return recommendations
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        try:
            slope, _ = np.polyfit(x, data, 1)
            return float(np.tanh(slope))  # Normalize to [-1, 1]
        except:
            return 0.0
    
    def _calculate_convergence(self, data: np.ndarray) -> float:
        """Calculate convergence indicator (0 to 1)"""
        if len(data) < 50:
            return 0.0
        
        recent_var = np.var(data[-25:])
        overall_var = np.var(data)
        
        if overall_var == 0:
            return 1.0
        
        convergence = 1.0 - (recent_var / overall_var)
        return max(0.0, min(1.0, convergence))
    
    def _calculate_consistency(self, data: np.ndarray) -> float:
        """Calculate performance consistency (0 to 1)"""
        if len(data) < 10:
            return 0.0
        
        mean_val = np.mean(data)
        if mean_val == 0:
            return 0.0
        
        cv = np.std(data) / abs(mean_val)  # Coefficient of variation
        consistency = 1.0 / (1.0 + cv)  # Higher consistency = lower CV
        return consistency
    
    def _calculate_improvement_rate(self, data: np.ndarray) -> float:
        """Calculate rate of improvement"""
        if len(data) < 20:
            return 0.0
        
        early_mean = np.mean(data[:len(data)//4])
        recent_mean = np.mean(data[-len(data)//4:])
        
        if early_mean == 0:
            return 0.0
        
        return (recent_mean - early_mean) / abs(early_mean)
    
    def finalize(self):
        """Finalize monitoring and save final analysis"""
        
        print(f"\n🏁 Finalizing monitoring for experiment: {self.config.experiment_name}")
        
        # Save final data
        self._save_data()
        
        # Generate final plots
        if self.config.save_plots:
            self._generate_plots()
        
        # Generate comprehensive analysis
        final_analysis = self.get_comprehensive_analysis()
        
        # Save analysis
        analysis_file = self.output_dir / "final_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(final_analysis, f, indent=2, default=str)
        
        # Print summary
        print(f"📊 Final Results:")
        if 'training_summary' in final_analysis:
            summary = final_analysis['training_summary']
            print(f"   Episodes: {summary['total_episodes']}")
            print(f"   Training time: {summary['elapsed_time_minutes']:.1f} minutes")
            print(f"   Best reward: {summary['best_performance']['reward']:.1f}")
            print(f"   Best coverage: {summary['best_performance']['coverage']:.1f}%")
        
        print(f"📁 All results saved to: {self.output_dir}")
        
        return final_analysis


def create_comprehensive_monitor(config: Dict, output_manager=None) -> ComprehensiveMonitor:
    """Factory function to create comprehensive monitoring system"""
    
    # Extract monitoring config and remove non-config keys
    monitor_dict = config.get('monitoring', {})
    # Remove keys that are not part of MonitoringConfig
    monitor_dict.pop('enabled', None)
    monitor_dict.pop('create_videos', None)
    monitor_dict.pop('comprehensive_analysis', None)
    monitor_dict.pop('real_time_plotting', None)
    monitor_dict.pop('save_plots', None)
    
    monitoring_config = MonitoringConfig(**monitor_dict)
    monitor = ComprehensiveMonitor(monitoring_config)
    
    # Integrate with output manager if provided
    if output_manager:
        monitor.output_manager = output_manager
        print("🔗 Monitor integrated with output manager")
    
    return monitor