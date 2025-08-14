"""
Professional Visualizer with White Background and Legends
=========================================================
Creates publication-ready visualizations with clean white backgrounds,
agent legends, and standardized output formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont
import io
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class VisualizationConfig:
    """Configuration for professional visualizations"""
    # Display settings
    figure_size: Tuple[int, int] = (16, 10)
    dpi: int = 100
    fps: int = 10  # Slower animation for better observation
    
    # Color scheme (professional)
    background_color: str = 'white'
    grid_color: str = '#E0E0E0'
    text_color: str = '#333333'
    
    # Agent colors and symbols
    satellite_color: str = '#E74C3C'  # Professional red
    uav_color: str = '#3498DB'        # Professional blue
    ground_color: str = '#27AE60'     # Professional green
    
    # Legend settings
    legend_width: float = 0.25  # 25% of figure width for legend
    legend_bg_color: str = '#F8F9FA'
    legend_border_color: str = '#DEE2E6'
    
    # Animation settings
    trail_length: int = 30  # Shorter trails for cleaner look
    trail_alpha_decay: float = 0.02
    
    # Font settings
    title_font_size: int = 16
    label_font_size: int = 12
    legend_font_size: int = 11
    metrics_font_size: int = 10


class ProfessionalVisualizer:
    """Professional visualization system with white backgrounds and legends"""
    
    def __init__(self, env, config: Optional[VisualizationConfig] = None):
        self.env = env
        self.config = config or VisualizationConfig()
        
        # Agent type definitions
        self.agent_types = {
            'satellite': {
                'symbol': '▲',
                'marker': '^',
                'size': 200,
                'color': self.config.satellite_color,
                'description': 'Satellite (150m altitude)',
                'coverage_radius': 80,
                'coverage_alpha': 0.15
            },
            'uav': {
                'symbol': '■',
                'marker': 's',
                'size': 150,
                'color': self.config.uav_color,
                'description': 'UAV (90-110m altitude)',
                'coverage_radius': 60,
                'coverage_alpha': 0.15
            },
            'ground': {
                'symbol': '●',
                'marker': 'o',
                'size': 120,
                'color': self.config.ground_color,
                'description': 'Ground Station (0m altitude)',
                'coverage_radius': 40,
                'coverage_alpha': 0.15
            }
        }
        
        # Initialize tracking
        self.episode_data = []
        self.current_frame = 0
        
    def create_figure_with_legend(self):
        """Create figure with main plot and legend panel"""
        fig = plt.figure(figsize=self.config.figure_size, facecolor=self.config.background_color)
        
        # Create grid with legend panel on the right
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        
        # Main 3D plot
        ax_main = fig.add_subplot(gs[0], projection='3d')
        ax_main.set_facecolor(self.config.background_color)
        
        # Legend panel (2D)
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.set_facecolor(self.config.legend_bg_color)
        
        return fig, ax_main, ax_legend
    
    def setup_main_axes(self, ax):
        """Setup main 3D axes with professional styling"""
        # Set limits
        ax.set_xlim(0, self.env.area_size[0])
        ax.set_ylim(0, self.env.area_size[1])
        ax.set_zlim(0, 200)
        
        # Labels with professional fonts
        ax.set_xlabel('X Position (m)', fontsize=self.config.label_font_size, color=self.config.text_color)
        ax.set_ylabel('Y Position (m)', fontsize=self.config.label_font_size, color=self.config.text_color)
        ax.set_zlabel('Altitude (m)', fontsize=self.config.label_font_size, color=self.config.text_color)
        
        # Grid styling
        ax.grid(True, alpha=0.3, color=self.config.grid_color, linestyle='--')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
    def create_legend_panel(self, ax_legend, metrics: Optional[Dict] = None):
        """Create professional legend panel with agent descriptions and metrics"""
        ax_legend.clear()
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.axis('off')
        
        y_pos = 0.95
        
        # Title
        ax_legend.text(0.5, y_pos, 'AGENT LEGEND', 
                      fontsize=self.config.title_font_size, 
                      fontweight='bold',
                      ha='center',
                      color=self.config.text_color)
        y_pos -= 0.08
        
        # Separator line
        ax_legend.plot([0.1, 0.9], [y_pos, y_pos], 
                      color=self.config.legend_border_color, 
                      linewidth=1)
        y_pos -= 0.05
        
        # Agent types
        for agent_type, props in self.agent_types.items():
            # Symbol and description
            ax_legend.scatter(0.15, y_pos, 
                            marker=props['marker'],
                            s=props['size'],
                            color=props['color'],
                            edgecolors='white',
                            linewidth=1)
            
            ax_legend.text(0.25, y_pos, props['description'],
                          fontsize=self.config.legend_font_size,
                          va='center',
                          color=self.config.text_color)
            
            y_pos -= 0.06
            
            # Coverage info
            ax_legend.text(0.25, y_pos, 
                          f"Coverage: {props['coverage_radius']}m radius",
                          fontsize=self.config.metrics_font_size,
                          va='center',
                          color='#666666',
                          style='italic')
            y_pos -= 0.08
        
        # Separator
        y_pos -= 0.02
        ax_legend.plot([0.1, 0.9], [y_pos, y_pos], 
                      color=self.config.legend_border_color, 
                      linewidth=1)
        y_pos -= 0.05
        
        # Visual elements
        ax_legend.text(0.5, y_pos, 'VISUAL ELEMENTS', 
                      fontsize=self.config.label_font_size, 
                      fontweight='bold',
                      ha='center',
                      color=self.config.text_color)
        y_pos -= 0.06
        
        # Trail line
        ax_legend.plot([0.15, 0.25], [y_pos, y_pos], 
                      color='#888888', 
                      linewidth=2, 
                      alpha=0.5)
        ax_legend.text(0.3, y_pos, 'Movement Trail',
                      fontsize=self.config.legend_font_size,
                      va='center',
                      color=self.config.text_color)
        y_pos -= 0.05
        
        # Communication link
        ax_legend.plot([0.15, 0.25], [y_pos, y_pos], 
                      color='#FFB366', 
                      linewidth=1, 
                      linestyle='--',
                      alpha=0.7)
        ax_legend.text(0.3, y_pos, 'Communication Link',
                      fontsize=self.config.legend_font_size,
                      va='center',
                      color=self.config.text_color)
        y_pos -= 0.05
        
        # Coverage area
        circle = plt.Circle((0.2, y_pos), 0.04, 
                           color='#3498DB', 
                           alpha=0.2)
        ax_legend.add_patch(circle)
        ax_legend.text(0.3, y_pos, 'Coverage Area',
                      fontsize=self.config.legend_font_size,
                      va='center',
                      color=self.config.text_color)
        y_pos -= 0.08
        
        # Metrics section if provided
        if metrics:
            ax_legend.plot([0.1, 0.9], [y_pos, y_pos], 
                          color=self.config.legend_border_color, 
                          linewidth=1)
            y_pos -= 0.05
            
            ax_legend.text(0.5, y_pos, 'PERFORMANCE', 
                          fontsize=self.config.label_font_size, 
                          fontweight='bold',
                          ha='center',
                          color=self.config.text_color)
            y_pos -= 0.06
            
            # Display metrics
            for key, value in metrics.items():
                if y_pos < 0.1:
                    break
                ax_legend.text(0.15, y_pos, f"{key}:",
                              fontsize=self.config.metrics_font_size,
                              color=self.config.text_color)
                ax_legend.text(0.85, y_pos, f"{value:.2f}",
                              fontsize=self.config.metrics_font_size,
                              ha='right',
                              color=self.config.text_color,
                              fontweight='bold')
                y_pos -= 0.04
        
        # Frame border - use Rectangle instead of FancyBboxPatch for 2D axes
        from matplotlib.patches import Rectangle
        rect = Rectangle((0.05, 0.05), 0.9, 0.9,
                        linewidth=1,
                        edgecolor=self.config.legend_border_color,
                        facecolor='none')
        ax_legend.add_patch(rect)
    
    def visualize_frame(self, ax, frame_data: Dict, trails: Optional[List] = None):
        """Visualize a single frame with professional styling"""
        ax.clear()
        self.setup_main_axes(ax)
        
        positions = frame_data['positions']
        n_agents = len(positions)
        
        # Determine agent types based on altitude
        agent_types = []
        for pos in positions:
            if pos[2] > 120:  # Satellite
                agent_types.append('satellite')
            elif pos[2] > 50:  # UAV
                agent_types.append('uav')
            else:  # Ground
                agent_types.append('ground')
        
        # Draw trails if available
        if trails and len(trails) > 0:
            for i, agent_type in enumerate(agent_types):
                agent_trail = [t[i] for t in trails if i < len(t)]
                if len(agent_trail) > 1:
                    trail_array = np.array(agent_trail)
                    for j in range(len(trail_array) - 1):
                        alpha = (j / len(trail_array)) * 0.5
                        ax.plot(trail_array[j:j+2, 0],
                               trail_array[j:j+2, 1],
                               trail_array[j:j+2, 2],
                               color=self.agent_types[agent_type]['color'],
                               alpha=alpha,
                               linewidth=1.5)
        
        # Draw coverage areas (on ground plane)
        for i, (pos, agent_type) in enumerate(zip(positions, agent_types)):
            props = self.agent_types[agent_type]
            
            # Coverage circle on ground
            theta = np.linspace(0, 2*np.pi, 50)
            x_circle = pos[0] + props['coverage_radius'] * np.cos(theta)
            y_circle = pos[1] + props['coverage_radius'] * np.sin(theta)
            z_circle = np.zeros_like(x_circle)
            
            ax.plot(x_circle, y_circle, z_circle,
                   color=props['color'],
                   alpha=props['coverage_alpha'],
                   linewidth=1)
            # Use Poly3DCollection for 3D fill
            verts = [list(zip(x_circle, y_circle, z_circle))]
            poly = Poly3DCollection(verts, alpha=props['coverage_alpha'] * 0.5, 
                                   facecolor=props['color'], edgecolor='none')
            ax.add_collection3d(poly)
        
        # Draw communication links
        if 'communications' in frame_data:
            for link in frame_data['communications']:
                if link['connected']:
                    i, j = link['agents']
                    ax.plot([positions[i][0], positions[j][0]],
                           [positions[i][1], positions[j][1]],
                           [positions[i][2], positions[j][2]],
                           color='#FFB366',
                           alpha=0.6,
                           linewidth=1,
                           linestyle='--')
        
        # Draw agents
        for i, (pos, agent_type) in enumerate(zip(positions, agent_types)):
            props = self.agent_types[agent_type]
            
            # Agent marker
            ax.scatter(pos[0], pos[1], pos[2],
                      marker=props['marker'],
                      s=props['size'],
                      color=props['color'],
                      edgecolors='white',
                      linewidth=2,
                      alpha=0.95,
                      depthshade=True)
            
            # Agent ID label
            ax.text(pos[0], pos[1], pos[2] + 10,
                   f"A{i}",
                   fontsize=10,
                   color=self.config.text_color,
                   ha='center',
                   fontweight='bold')
        
        # Title with frame info
        if 'step' in frame_data:
            ax.text2D(0.5, 0.98, f"Step {frame_data['step']}",
                     transform=ax.transAxes,
                     fontsize=self.config.title_font_size,
                     ha='center',
                     color=self.config.text_color)
    
    def generate_episode_animation(self, episode_data: List[Dict], 
                                  output_path: str,
                                  format: str = 'gif'):
        """Generate professional animation with legend"""
        fig, ax_main, ax_legend = self.create_figure_with_legend()
        
        # Calculate metrics for legend
        metrics = self._calculate_episode_metrics(episode_data)
        
        # Setup legend (static)
        self.create_legend_panel(ax_legend, metrics)
        
        # Prepare trails
        trails = []
        
        def animate(frame_idx):
            frame_data = episode_data[frame_idx]
            
            # Update trails
            trails.append(frame_data['positions'])
            if len(trails) > self.config.trail_length:
                trails.pop(0)
            
            # Visualize frame
            self.visualize_frame(ax_main, frame_data, trails)
            
            # Update progress in legend
            progress = (frame_idx + 1) / len(episode_data)
            ax_legend.text(0.5, 0.02, f"Progress: {progress*100:.1f}%",
                          fontsize=self.config.metrics_font_size,
                          ha='center',
                          color=self.config.text_color,
                          transform=ax_legend.transAxes)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, 
                                      frames=len(episode_data),
                                      interval=1000/self.config.fps,
                                      blit=False)
        
        # Save animation
        if format == 'gif':
            writer = animation.PillowWriter(fps=self.config.fps)
            anim.save(output_path, writer=writer, dpi=self.config.dpi)
        elif format == 'mp4':
            writer = animation.FFMpegWriter(fps=self.config.fps)
            anim.save(output_path, writer=writer, dpi=self.config.dpi)
        
        plt.close(fig)
        
        return output_path
    
    def _calculate_episode_metrics(self, episode_data: List[Dict]) -> Dict:
        """Calculate performance metrics for the episode"""
        metrics = {}
        
        if len(episode_data) > 0:
            # Average reward
            rewards = [frame.get('reward', 0) for frame in episode_data]
            metrics['Avg Reward'] = np.mean(rewards)
            
            # Coverage
            coverage_rates = [frame.get('coverage', 0) for frame in episode_data]
            if coverage_rates:
                metrics['Avg Coverage'] = np.mean(coverage_rates)
            
            # Communication success
            comm_success = []
            for frame in episode_data:
                if 'communications' in frame:
                    success_rate = sum(1 for link in frame['communications'] 
                                     if link['connected']) / max(1, len(frame['communications']))
                    comm_success.append(success_rate)
            if comm_success:
                metrics['Comm Success'] = np.mean(comm_success)
            
            # Episode length
            metrics['Episode Length'] = len(episode_data)
        
        return metrics
    
    def create_static_overview(self, episode_data: List[Dict], 
                              output_path: str):
        """Create a static overview image with key frames"""
        fig = plt.figure(figsize=(20, 12), facecolor=self.config.background_color)
        
        # Select key frames
        n_frames = min(6, len(episode_data))
        frame_indices = np.linspace(0, len(episode_data)-1, n_frames, dtype=int)
        
        # Create grid
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.3], 
                             height_ratios=[1, 1],
                             wspace=0.2, hspace=0.3)
        
        # Plot key frames
        for idx, frame_idx in enumerate(frame_indices):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col], projection='3d')
            
            frame_data = episode_data[frame_idx]
            self.visualize_frame(ax, frame_data)
            ax.set_title(f"Step {frame_data.get('step', frame_idx)}", 
                        fontsize=self.config.label_font_size)
        
        # Add legend panel
        ax_legend = fig.add_subplot(gs[:, 3])
        metrics = self._calculate_episode_metrics(episode_data)
        self.create_legend_panel(ax_legend, metrics)
        
        # Overall title
        fig.suptitle('Episode Overview - Professional Visualization',
                    fontsize=self.config.title_font_size + 2,
                    fontweight='bold',
                    color=self.config.text_color)
        
        plt.savefig(output_path, dpi=self.config.dpi, 
                   facecolor=self.config.background_color,
                   bbox_inches='tight')
        plt.close(fig)
        
        return output_path