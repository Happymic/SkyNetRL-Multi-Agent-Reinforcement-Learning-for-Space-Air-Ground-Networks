"""
Clean Training Results Visualizer
=================================
Professional visualization system for training results with comprehensive analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class CleanTrainingVisualizer:
    """Professional training results visualizer"""
    
    def __init__(self, output_dir: str):
        """Initialize visualizer with clean professional style"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set professional style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Professional color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#23CE6B',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#3B82F6',
            'dark': '#1E293B',
            'light': '#F8FAFC'
        }
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'lines.linewidth': 2.5,
            'grid.alpha': 0.3,
            'axes.grid': True
        })
    
    def create_comprehensive_training_dashboard(self, 
                                              training_data: Dict[str, Any],
                                              save_name: str = 'training_dashboard.png'):
        """
        Create comprehensive training dashboard with all key metrics
        
        Args:
            training_data: Dictionary containing training metrics and results
            save_name: Filename to save the dashboard
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main training curves (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Performance metrics (second row)  
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[1, 3])
        
        # Network architecture and attention (third row)
        ax7 = fig.add_subplot(gs[2, :2])
        ax8 = fig.add_subplot(gs[2, 2:])
        
        # Final results and analysis (bottom row)
        ax9 = fig.add_subplot(gs[3, :2])
        ax10 = fig.add_subplot(gs[3, 2:])
        
        # Extract training statistics
        stats = training_data.get('training_stats', {})
        eval_stats = training_data.get('evaluation_stats', {})
        
        # 1. Episode Rewards Over Time
        if 'episode_rewards' in stats:
            rewards = self._smooth_curve(stats['episode_rewards'], window=20)
            episodes = range(len(rewards))
            ax1.plot(episodes, rewards, color=self.colors['primary'], linewidth=3, alpha=0.8)
            ax1.fill_between(episodes, rewards, alpha=0.2, color=self.colors['primary'])
            ax1.set_title('Training Progress: Episode Rewards', fontweight='bold', pad=20)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Average Reward')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            ax1.plot(episodes, p(episodes), "--", color=self.colors['warning'], alpha=0.8)
        
        # 2. Coverage Rate Evolution
        if 'coverage_rates' in stats:
            coverage = self._smooth_curve(stats['coverage_rates'], window=20)
            episodes = range(len(coverage))
            ax2.plot(episodes, coverage, color=self.colors['success'], linewidth=3, alpha=0.8)
            ax2.fill_between(episodes, coverage, alpha=0.2, color=self.colors['success'])
            ax2.set_title('Coverage Performance', fontweight='bold', pad=20)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Coverage Rate')
            ax2.set_ylim(0, 1.0)
            ax2.grid(True, alpha=0.3)
        
        # 3. Energy Efficiency
        if 'energy_efficiency' in stats:
            efficiency = self._smooth_curve(stats['energy_efficiency'], window=20)
            episodes = range(len(efficiency))
            ax3.plot(episodes, efficiency, color=self.colors['info'], linewidth=2)
            ax3.set_title('Energy Efficiency', fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Efficiency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Actor Loss
        if 'actor_losses' in stats:
            losses = self._smooth_curve(stats['actor_losses'], window=20)
            episodes = range(len(losses))
            ax4.plot(episodes, losses, color=self.colors['danger'], linewidth=2)
            ax4.set_title('Actor Loss', fontweight='bold')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss')
            ax4.grid(True, alpha=0.3)
        
        # 5. Critic Loss
        if 'critic_losses' in stats:
            losses = self._smooth_curve(stats['critic_losses'], window=20)
            episodes = range(len(losses))
            ax5.plot(episodes, losses, color=self.colors['secondary'], linewidth=2)
            ax5.set_title('Critic Loss', fontweight='bold')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Loss')
            ax5.grid(True, alpha=0.3)
        
        # 6. Exploration Rate
        if 'exploration_rates' in stats:
            exploration = stats['exploration_rates']
            episodes = range(len(exploration))
            ax6.plot(episodes, exploration, color=self.colors['warning'], linewidth=2)
            ax6.set_title('Exploration Rate', fontweight='bold')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Epsilon')
            ax6.grid(True, alpha=0.3)
        
        # 7. Attention Analysis (if available)
        if 'attention_weights' in training_data:
            self._plot_attention_heatmap(ax7, training_data['attention_weights'])
        else:
            # Network architecture diagram
            self._plot_network_architecture(ax7)
        
        # 8. Performance Distribution
        if 'final_episode_rewards' in eval_stats:
            rewards_dist = eval_stats['final_episode_rewards']
            ax8.hist(rewards_dist, bins=20, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax8.set_title('Final Performance Distribution', fontweight='bold')
            ax8.set_xlabel('Episode Reward')
            ax8.set_ylabel('Frequency')
            ax8.grid(True, alpha=0.3)
        
        # 9. Final Performance Metrics
        self._plot_final_metrics_radar(ax9, eval_stats)
        
        # 10. Training Summary Statistics
        self._plot_training_summary(ax10, training_data)
        
        # Add main title
        fig.suptitle('Comprehensive Training Analysis Dashboard', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # Save the dashboard
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Comprehensive training dashboard saved: {save_path}")
        return save_path
    
    def create_training_animation(self, 
                                episode_data: List[Dict],
                                save_name: str = 'training_animation.gif'):
        """
        Create animated visualization of training progress
        
        Args:
            episode_data: List of episode data dictionaries
            save_name: Filename to save animation
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress Animation', fontsize=16, fontweight='bold')
        
        def animate(frame):
            # Clear all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Get data up to current frame
            current_data = episode_data[:frame+1]
            episodes = range(len(current_data))
            
            # Rewards
            rewards = [ep.get('reward', 0) for ep in current_data]
            ax1.plot(episodes, rewards, color=self.colors['primary'], linewidth=2)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            
            # Coverage
            coverage = [ep.get('coverage_rate', 0) for ep in current_data]
            ax2.plot(episodes, coverage, color=self.colors['success'], linewidth=2)
            ax2.set_title('Coverage Rate')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Coverage')
            ax2.set_ylim(0, 1.0)
            ax2.grid(True, alpha=0.3)
            
            # Energy Efficiency
            efficiency = [ep.get('energy_efficiency', 0) for ep in current_data]
            ax3.plot(episodes, efficiency, color=self.colors['info'], linewidth=2)
            ax3.set_title('Energy Efficiency')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Efficiency')
            ax3.grid(True, alpha=0.3)
            
            # Actor Loss
            losses = [ep.get('actor_loss', 0) for ep in current_data]
            ax4.plot(episodes, losses, color=self.colors['danger'], linewidth=2)
            ax4.set_title('Actor Loss')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        # Create animation
        frames = min(len(episode_data), 200)  # Limit frames for reasonable file size
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True)
        
        # Save animation
        save_path = os.path.join(self.output_dir, save_name)
        anim.save(save_path, writer='pillow', fps=10, dpi=100)
        plt.close()
        
        print(f"Training animation saved: {save_path}")
        return save_path
    
    def create_3d_performance_landscape(self, 
                                      performance_data: Dict,
                                      save_name: str = '3d_performance.png'):
        """
        Create 3D landscape visualization of performance metrics
        
        Args:
            performance_data: Dictionary with performance metrics
            save_name: Filename to save visualization
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create synthetic landscape data if real data not available
        episodes = performance_data.get('episodes', range(100))
        coverage = performance_data.get('coverage_rates', np.random.rand(100) * 0.8 + 0.2)
        efficiency = performance_data.get('energy_efficiency', np.random.rand(100) * 0.6 + 0.4)
        rewards = performance_data.get('episode_rewards', np.random.rand(100) * 200 - 100)
        
        # Create 3D scatter plot
        scatter = ax.scatter(coverage, efficiency, rewards, 
                           c=episodes, cmap='viridis', s=60, alpha=0.8)
        
        ax.set_xlabel('Coverage Rate', fontsize=12)
        ax.set_ylabel('Energy Efficiency', fontsize=12)
        ax.set_zlabel('Episode Reward', fontsize=12)
        ax.set_title('3D Performance Landscape', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Training Episode', fontsize=12)
        
        # Save visualization
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"3D performance landscape saved: {save_path}")
        return save_path
    
    def _smooth_curve(self, data: List[float], window: int = 10) -> List[float]:
        """Apply smoothing to curve data"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            smoothed.append(np.mean(window_data))
        
        return smoothed
    
    def _plot_attention_heatmap(self, ax, attention_weights):
        """Plot attention weights as heatmap"""
        if isinstance(attention_weights, list) and len(attention_weights) > 0:
            weights_matrix = np.array(attention_weights[-10:])  # Last 10 episodes
            im = ax.imshow(weights_matrix, cmap='viridis', aspect='auto')
            ax.set_title('Recent Attention Weights', fontweight='bold')
            ax.set_xlabel('Attention Head')
            ax.set_ylabel('Recent Episodes')
            plt.colorbar(im, ax=ax)
    
    def _plot_network_architecture(self, ax):
        """Plot simplified network architecture diagram"""
        # Create a simple network diagram
        ax.text(0.5, 0.8, 'Multi-Head Attention', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['primary'], alpha=0.7))
        
        ax.text(0.2, 0.5, 'Actor\nNetwork', ha='center', va='center',
                fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['success'], alpha=0.7))
        
        ax.text(0.8, 0.5, 'Critic\nNetwork', ha='center', va='center',
                fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['info'], alpha=0.7))
        
        ax.text(0.5, 0.2, 'Shared Feature Extractor', ha='center', va='center',
                fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['warning'], alpha=0.7))
        
        # Add arrows
        ax.annotate('', xy=(0.2, 0.4), xytext=(0.5, 0.7),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   transform=ax.transAxes)
        ax.annotate('', xy=(0.8, 0.4), xytext=(0.5, 0.7),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   transform=ax.transAxes)
        
        ax.set_title('Network Architecture', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _plot_final_metrics_radar(self, ax, eval_stats):
        """Plot radar chart of final performance metrics"""
        metrics = ['Coverage', 'Energy Eff.', 'Stability', 'Speed', 'Safety']
        values = [
            eval_stats.get('final_metrics', {}).get('avg_coverage_rate', 0.5) * 100,
            eval_stats.get('final_metrics', {}).get('avg_energy_efficiency', 0.5) * 100,
            eval_stats.get('stability_score', 0.5) * 100,
            eval_stats.get('convergence_speed', 0.5) * 100,
            eval_stats.get('safety_score', 0.5) * 100
        ]
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Close the plot
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['primary'])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_title('Final Performance Metrics', fontweight='bold', pad=20)
        ax.grid(True)
    
    def _plot_training_summary(self, ax, training_data):
        """Plot training summary statistics"""
        stats = training_data.get('training_stats', {})
        
        # Summary statistics
        summary_text = f"""
Training Summary:
• Total Episodes: {len(stats.get('episode_rewards', []))}
• Final Avg Reward: {np.mean(stats.get('episode_rewards', [0])[-10:]):.2f}
• Best Coverage: {max(stats.get('coverage_rates', [0])):.3f}
• Training Time: {training_data.get('training_time', 'N/A')}
• Convergence Episode: {training_data.get('convergence_episode', 'N/A')}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light'], alpha=0.8))
        
        ax.set_title('Training Summary', fontweight='bold')
        ax.axis('off')
    
    def generate_training_report(self, training_data: Dict, save_name: str = 'training_report.json'):
        """Generate comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_summary': {
                'total_episodes': len(training_data.get('training_stats', {}).get('episode_rewards', [])),
                'final_performance': training_data.get('evaluation_stats', {}).get('final_metrics', {}),
                'training_time': training_data.get('training_time', 'N/A'),
                'convergence_episode': training_data.get('convergence_episode', 'N/A')
            },
            'performance_metrics': training_data.get('evaluation_stats', {}),
            'training_configuration': training_data.get('config', {}),
            'visualizations_created': [
                'training_dashboard.png',
                '3d_performance.png',
                'training_animation.gif'
            ]
        }
        
        # Save report
        save_path = os.path.join(self.output_dir, save_name)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training report saved: {save_path}")
        return save_path