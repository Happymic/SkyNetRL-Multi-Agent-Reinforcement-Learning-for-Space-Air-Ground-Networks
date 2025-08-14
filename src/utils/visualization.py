"""
Visualization Utilities for SAGIN Experiments
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ExperimentVisualizer:
    """Comprehensive visualization for SAGIN experiments"""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create subdirectories
        self.figures_dir = os.path.join(output_dir, 'figures')
        self.interactive_dir = os.path.join(output_dir, 'interactive')
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.interactive_dir, exist_ok=True)
    
    def plot_training_comparison(self, results: Dict[str, Dict], 
                                save_name: str = 'training_comparison.png'):
        """
        Plot comparison of training curves across algorithms
        
        Args:
            results: Dictionary of algorithm results
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['episode_rewards', 'coverage_rates', 'energy_efficiency', 'actor_losses']
        titles = ['Episode Rewards', 'Coverage Rates', 'Energy Efficiency', 'Actor Losses']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for algorithm, result in results.items():
                training_stats = result.get('training_stats', {})
                if metric in training_stats and training_stats[metric]:
                    data = training_stats[metric]
                    
                    # Smooth the curve
                    smoothed_data = self._smooth_curve(data, window=10)
                    episodes = range(len(smoothed_data))
                    
                    ax.plot(episodes, smoothed_data, label=algorithm.replace('_', ' ').title(), 
                           linewidth=2, alpha=0.8)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel(title.split()[0])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training comparison plot saved: {save_name}")
    
    def plot_performance_comparison(self, results: Dict[str, Dict],
                                   save_name: str = 'performance_comparison.png'):
        """
        Plot bar chart comparison of final performance metrics
        
        Args:
            results: Dictionary of algorithm results
            save_name: Filename to save plot
        """
        # Extract final performance metrics
        algorithms = []
        coverage_rates = []
        energy_efficiencies = []
        completion_times = []
        
        for algorithm, result in results.items():
            eval_stats = result.get('evaluation_stats', {})
            final_metrics = eval_stats.get('final_metrics', {})
            
            algorithms.append(algorithm.replace('_', ' ').title())
            coverage_rates.append(final_metrics.get('avg_coverage_rate', 0))
            energy_efficiencies.append(final_metrics.get('avg_energy_efficiency', 0))
            completion_times.append(final_metrics.get('avg_completion_time', 0))
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Coverage rates
        bars1 = axes[0].bar(algorithms, coverage_rates, alpha=0.8)
        axes[0].set_title('Average Coverage Rate', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Coverage Rate')
        axes[0].set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars1, coverage_rates):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Energy efficiency
        bars2 = axes[1].bar(algorithms, energy_efficiencies, alpha=0.8, color='orange')
        axes[1].set_title('Average Energy Efficiency', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Energy Efficiency')
        
        for bar, value in zip(bars2, energy_efficiencies):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Completion times (if available)
        if any(ct > 0 for ct in completion_times):
            bars3 = axes[2].bar(algorithms, completion_times, alpha=0.8, color='green')
            axes[2].set_title('Average Completion Time', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Steps to 80% Coverage')
            
            for bar, value in zip(bars3, completion_times):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved: {save_name}")
    
    def plot_ablation_study(self, ablation_results: Dict[str, Dict],
                           save_name: str = 'ablation_study.png'):
        """
        Plot ablation study results
        
        Args:
            ablation_results: Dictionary of ablation study results
            save_name: Filename to save plot
        """
        # Extract data
        configurations = []
        coverage_rates = []
        energy_efficiencies = []
        attention_components = []
        
        for config_name, result in ablation_results.items():
            final_metrics = result.get('evaluation_stats', {}).get('final_metrics', {})
            attention_config = result.get('attention_config', {})
            
            configurations.append(config_name.replace('_', ' ').title())
            coverage_rates.append(final_metrics.get('avg_coverage_rate', 0))
            energy_efficiencies.append(final_metrics.get('avg_energy_efficiency', 0))
            
            # Count active attention components
            components = sum([
                attention_config.get('use_spatial', False),
                attention_config.get('use_agent', False),
                attention_config.get('use_task', False)
            ])
            attention_components.append(components)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Coverage rate by configuration
        bars = axes[0].bar(configurations, coverage_rates, alpha=0.8)
        axes[0].set_title('Coverage Rate by Attention Configuration', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Coverage Rate')
        axes[0].set_ylim(0, 1.0)
        
        # Color bars by number of attention components
        colors = plt.cm.viridis(np.array(attention_components) / 3.0)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for bar, value in zip(bars, coverage_rates):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Scatter plot: attention components vs performance
        scatter = axes[1].scatter(attention_components, coverage_rates, 
                                 c=energy_efficiencies, s=100, alpha=0.8, cmap='plasma')
        axes[1].set_title('Attention Components vs Performance', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Attention Components')
        axes[1].set_ylabel('Coverage Rate')
        axes[1].set_xticks([0, 1, 2, 3])
        
        # Add colorbar for energy efficiency
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Energy Efficiency')
        
        # Add configuration labels
        for i, config in enumerate(configurations):
            axes[1].annotate(config, (attention_components[i], coverage_rates[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Rotate x-axis labels
        axes[0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Ablation study plot saved: {save_name}")
    
    def plot_scalability_analysis(self, scalability_results: Dict[str, Dict],
                                 save_name: str = 'scalability_analysis.png'):
        """
        Plot scalability analysis
        
        Args:
            scalability_results: Dictionary of scalability study results
            save_name: Filename to save plot
        """
        # Extract data
        scales = []
        num_agents_list = []
        area_sizes = []
        
        ae_maddpg_coverage = []
        baseline_coverage = []
        ae_maddpg_efficiency = []
        baseline_efficiency = []
        
        for scale_name, scale_result in scalability_results.items():
            scales.append(scale_name.title())
            config = scale_result['config']
            num_agents_list.append(config['num_agents'])
            area_sizes.append(config['area_size'])
            
            results = scale_result['results']
            
            # AE-MADDPG results
            ae_metrics = results.get('ae_maddpg', {}).get('evaluation_stats', {}).get('final_metrics', {})
            ae_maddpg_coverage.append(ae_metrics.get('avg_coverage_rate', 0))
            ae_maddpg_efficiency.append(ae_metrics.get('avg_energy_efficiency', 0))
            
            # Baseline results
            base_metrics = results.get('baseline_maddpg', {}).get('evaluation_stats', {}).get('final_metrics', {})
            baseline_coverage.append(base_metrics.get('avg_coverage_rate', 0))
            baseline_efficiency.append(base_metrics.get('avg_energy_efficiency', 0))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Coverage rate by scale
        x = np.arange(len(scales))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, ae_maddpg_coverage, width, label='AE-MADDPG', alpha=0.8)
        axes[0, 0].bar(x + width/2, baseline_coverage, width, label='Baseline MADDPG', alpha=0.8)
        axes[0, 0].set_title('Coverage Rate by Scale', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Coverage Rate')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scales)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy efficiency by scale
        axes[0, 1].bar(x - width/2, ae_maddpg_efficiency, width, label='AE-MADDPG', alpha=0.8)
        axes[0, 1].bar(x + width/2, baseline_efficiency, width, label='Baseline MADDPG', alpha=0.8)
        axes[0, 1].set_title('Energy Efficiency by Scale', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Energy Efficiency')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scales)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Coverage vs number of agents
        axes[1, 0].plot(num_agents_list, ae_maddpg_coverage, 'o-', label='AE-MADDPG', linewidth=2)
        axes[1, 0].plot(num_agents_list, baseline_coverage, 'o-', label='Baseline MADDPG', linewidth=2)
        axes[1, 0].set_title('Coverage Rate vs Number of Agents', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Agents')
        axes[1, 0].set_ylabel('Coverage Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Coverage vs area size
        axes[1, 1].plot(area_sizes, ae_maddpg_coverage, 'o-', label='AE-MADDPG', linewidth=2)
        axes[1, 1].plot(area_sizes, baseline_coverage, 'o-', label='Baseline MADDPG', linewidth=2)
        axes[1, 1].set_title('Coverage Rate vs Area Size', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Area Size (m)')
        axes[1, 1].set_ylabel('Coverage Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scalability analysis plot saved: {save_name}")
    
    def plot_attention_analysis(self, attention_data: Dict,
                               save_name: str = 'attention_analysis.png'):
        """
        Plot attention weight analysis
        
        Args:
            attention_data: Dictionary of attention weight data
            save_name: Filename to save plot
        """
        if not attention_data:
            print("No attention data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Attention entropy over time
        if 'spatial_entropy' in attention_data:
            axes[0, 0].plot(attention_data['spatial_entropy'], label='Spatial', linewidth=2)
        if 'agent_entropy' in attention_data:
            axes[0, 0].plot(attention_data['agent_entropy'], label='Agent', linewidth=2)
        if 'task_entropy' in attention_data:
            axes[0, 0].plot(attention_data['task_entropy'], label='Task', linewidth=2)
        
        axes[0, 0].set_title('Attention Entropy Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Attention weight heatmap (spatial attention)
        if 'spatial_weights' in attention_data:
            spatial_weights = np.array(attention_data['spatial_weights'])
            if spatial_weights.ndim >= 2:
                im1 = axes[0, 1].imshow(spatial_weights[-1], cmap='viridis', aspect='auto')
                axes[0, 1].set_title('Final Spatial Attention Weights', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Object Index')
                axes[0, 1].set_ylabel('Agent Index')
                plt.colorbar(im1, ax=axes[0, 1])
        
        # Agent attention network graph
        if 'agent_weights' in attention_data:
            agent_weights = np.array(attention_data['agent_weights'])
            if agent_weights.ndim >= 2:
                im2 = axes[1, 0].imshow(agent_weights[-1], cmap='plasma', aspect='auto')
                axes[1, 0].set_title('Final Agent Attention Matrix', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Target Agent')
                axes[1, 0].set_ylabel('Source Agent')
                plt.colorbar(im2, ax=axes[1, 0])
        
        # Task attention focus
        if 'task_weights' in attention_data:
            task_weights = attention_data['task_weights']
            axes[1, 1].plot(task_weights, linewidth=2)
            axes[1, 1].set_title('Task Attention Over Time', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Attention Weight')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention analysis plot saved: {save_name}")
    
    def create_interactive_dashboard(self, results: Dict[str, Dict]):
        """
        Create interactive dashboard using Plotly
        
        Args:
            results: Dictionary of all experimental results
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Progress', 'Performance Comparison',
                           'Energy Efficiency', 'Coverage Over Time',
                           'Attention Analysis', 'Scalability'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Add traces for each algorithm
        colors = px.colors.qualitative.Set1
        
        for i, (algorithm, result) in enumerate(results.items()):
            training_stats = result.get('training_stats', {})
            color = colors[i % len(colors)]
            
            # Training curves
            if 'episode_rewards' in training_stats:
                episodes = list(range(len(training_stats['episode_rewards'])))
                fig.add_trace(
                    go.Scatter(x=episodes, y=training_stats['episode_rewards'],
                             name=f'{algorithm} Rewards', line=dict(color=color)),
                    row=1, col=1
                )
            
            if 'coverage_rates' in training_stats:
                episodes = list(range(len(training_stats['coverage_rates'])))
                fig.add_trace(
                    go.Scatter(x=episodes, y=training_stats['coverage_rates'],
                             name=f'{algorithm} Coverage', line=dict(color=color, dash='dash'),
                             yaxis='y2'),
                    row=1, col=1, secondary_y=True
                )
        
        # Performance comparison
        algorithms = list(results.keys())
        coverage_rates = []
        energy_efficiencies = []
        
        for algorithm, result in results.items():
            eval_stats = result.get('evaluation_stats', {})
            final_metrics = eval_stats.get('final_metrics', {})
            coverage_rates.append(final_metrics.get('avg_coverage_rate', 0))
            energy_efficiencies.append(final_metrics.get('avg_energy_efficiency', 0))
        
        fig.add_trace(
            go.Bar(x=algorithms, y=coverage_rates, name='Coverage Rate',
                  marker_color='lightblue'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=algorithms, y=energy_efficiencies, name='Energy Efficiency',
                  marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="SAGIN Experiment Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save interactive dashboard
        dashboard_path = os.path.join(self.interactive_dir, 'experiment_dashboard.html')
        pyo.plot(fig, filename=dashboard_path, auto_open=False)
        
        print(f"Interactive dashboard saved: {dashboard_path}")
    
    def _smooth_curve(self, data: List[float], window: int = 10) -> List[float]:
        """Smooth curve using moving average"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            smoothed.append(np.mean(window_data))
        
        return smoothed
    
    def create_paper_figures(self, results: Dict[str, Dict]):
        """
        Create publication-ready figures matching the paper
        
        Args:
            results: Dictionary of all experimental results
        """
        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # Figure 3: Training Curves (like paper Figure 3)
        self._create_paper_training_curves(results)
        
        # Figure 4: Performance Comparison (like paper Table 2)
        self._create_paper_performance_table(results)
        
        # Figure 5: Scalability Analysis
        if any('scalability' in str(key) for key in results.keys()):
            self._create_paper_scalability_plot(results)
        
        print("Paper-style figures created")
    
    def _create_paper_training_curves(self, results: Dict[str, Dict]):
        """Create paper-style training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        for algorithm, result in results.items():
            training_stats = result.get('training_stats', {})
            
            # Average reward
            if 'episode_rewards' in training_stats:
                rewards = self._smooth_curve(training_stats['episode_rewards'], 10)
                episodes = range(len(rewards))
                axes[0, 0].plot(episodes, rewards, label=algorithm.replace('_', '-').upper(), linewidth=2)
            
            # Coverage rate
            if 'coverage_rates' in training_stats:
                coverage = self._smooth_curve(training_stats['coverage_rates'], 10)
                episodes = range(len(coverage))
                axes[0, 1].plot(episodes, coverage, label=algorithm.replace('_', '-').upper(), linewidth=2)
        
        axes[0, 0].set_title('(a) Average Reward Evolution')
        axes[0, 0].set_xlabel('Training Episodes')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('(b) Coverage Rate Evolution')
        axes[0, 1].set_xlabel('Training Episodes')
        axes[0, 1].set_ylabel('Coverage Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'paper_training_curves.pdf'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def _create_paper_performance_table(self, results: Dict[str, Dict]):
        """Create paper-style performance comparison table"""
        # This would create a formatted table similar to Table 2 in the paper
        performance_data = []
        
        for algorithm, result in results.items():
            eval_stats = result.get('evaluation_stats', {})
            final_metrics = eval_stats.get('final_metrics', {})
            
            performance_data.append({
                'Method': algorithm.replace('_', '-').upper(),
                'Coverage Rate (%)': f"{final_metrics.get('avg_coverage_rate', 0)*100:.1f}",
                'Energy Efficiency': f"{final_metrics.get('avg_energy_efficiency', 0):.2f}",
                'Completion Time': f"{final_metrics.get('avg_completion_time', 0):.0f}",
                'Collision Rate (%)': f"{final_metrics.get('avg_collision_rate', 0)*100:.1f}"
            })
        
        # Save as CSV for table creation
        df = pd.DataFrame(performance_data)
        df.to_csv(os.path.join(self.figures_dir, 'performance_table.csv'), index=False)
        
        print("Performance table data saved as CSV")
    
    def _create_paper_scalability_plot(self, results: Dict[str, Dict]):
        """Create paper-style scalability plot"""
        # This would be implemented based on scalability study results
        pass