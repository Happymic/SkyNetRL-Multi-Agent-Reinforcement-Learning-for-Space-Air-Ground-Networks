"""
Agent Movement and Coverage Visualizer
=====================================
Creates vivid visualizations of agent movements, trajectories, and coverage areas during training.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.collections import LineCollection
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import io
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import colorsys


@dataclass
class AgentState:
    """State of an agent at a specific time"""
    x: float
    y: float
    agent_type: str
    agent_id: int
    coverage_radius: float
    energy_level: float = 1.0
    is_active: bool = True
    targets_covered: List[int] = None


@dataclass
class CoverageTarget:
    """Target to be covered"""
    x: float
    y: float
    target_id: int
    priority: float = 1.0
    is_covered: bool = False
    coverage_time: int = 0


class AgentMovementVisualizer:
    """Professional agent movement and coverage visualizer"""
    
    def __init__(self, output_dir: str, area_size: int = 1000):
        """Initialize the movement visualizer"""
        self.output_dir = output_dir
        self.area_size = area_size
        os.makedirs(output_dir, exist_ok=True)
        
        # Professional color scheme
        self.colors = {
            'satellite': '#E74C3C',     # Red
            'uav': '#3498DB',           # Blue  
            'ground_station': '#27AE60', # Green
            'target_uncovered': '#95A5A6', # Gray
            'target_covered': '#F39C12',   # Orange
            'coverage_area': '#ECF0F1',    # Light gray
            'trajectory': '#34495E',       # Dark gray
            'background': '#FFFFFF',       # White
            'grid': '#BDC3C7'             # Light gray
        }
        
        # Agent symbols and sizes
        self.agent_symbols = {
            'satellite': '◆',  # Diamond
            'uav': '▲',        # Triangle
            'ground_station': '■'  # Square
        }
        
        self.agent_sizes = {
            'satellite': 150,
            'uav': 120,
            'ground_station': 100
        }
        
        # Coverage radii (default values)
        self.coverage_radii = {
            'satellite': 200,
            'uav': 100,
            'ground_station': 150
        }
        
        # Animation settings
        self.trail_length = 30
        self.trail_alpha_decay = 0.03
        
        # Set professional matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 11,
            'lines.linewidth': 2,
            'grid.alpha': 0.3
        })
    
    def create_realtime_movement_animation(self, 
                                         episode_data: List[Dict],
                                         save_name: str = 'agent_movements.gif'):
        """
        Create real-time animation of agent movements and coverage
        
        Args:
            episode_data: List of episode states with agent positions
            save_name: Filename for the animation
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Initialize tracking for trajectories
        agent_trajectories = {}
        coverage_history = []
        
        def animate(frame):
            ax.clear()
            
            # Set up the environment
            ax.set_xlim(0, self.area_size)
            ax.set_ylim(0, self.area_size)
            ax.set_aspect('equal')
            ax.set_facecolor(self.colors['background'])
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            
            # Get current frame data
            if frame < len(episode_data):
                current_data = episode_data[frame]
                agents = current_data.get('agents', [])
                targets = current_data.get('targets', [])
                episode_num = current_data.get('episode', frame)
                step = current_data.get('step', 0)
                total_coverage = current_data.get('coverage_rate', 0)
                
                # Update agent trajectories
                for agent in agents:
                    agent_id = agent['id']
                    if agent_id not in agent_trajectories:
                        agent_trajectories[agent_id] = {'x': [], 'y': [], 'type': agent['type']}
                    
                    agent_trajectories[agent_id]['x'].append(agent['x'])
                    agent_trajectories[agent_id]['y'].append(agent['y'])
                    
                    # Keep only recent trajectory points
                    if len(agent_trajectories[agent_id]['x']) > self.trail_length:
                        agent_trajectories[agent_id]['x'] = agent_trajectories[agent_id]['x'][-self.trail_length:]
                        agent_trajectories[agent_id]['y'] = agent_trajectories[agent_id]['y'][-self.trail_length:]
                
                # Draw targets first (background)
                for target in targets:
                    color = self.colors['target_covered'] if target.get('covered', False) else self.colors['target_uncovered']
                    size = 80 if target.get('covered', False) else 60
                    alpha = 0.9 if target.get('covered', False) else 0.6
                    
                    ax.scatter(target['x'], target['y'], 
                             c=color, s=size, alpha=alpha, 
                             marker='*', edgecolors='black', linewidths=1,
                             zorder=2)
                
                # Draw coverage areas
                for agent in agents:
                    if agent.get('active', True):
                        coverage_radius = agent.get('coverage_radius', self.coverage_radii[agent['type']])
                        
                        # Coverage circle
                        coverage_circle = Circle(
                            (agent['x'], agent['y']), 
                            coverage_radius,
                            fill=True, 
                            facecolor=self.colors[agent['type']],
                            alpha=0.15,
                            edgecolor=self.colors[agent['type']],
                            linewidth=1.5,
                            zorder=1
                        )
                        ax.add_patch(coverage_circle)
                
                # Draw agent trajectories
                for agent_id, trajectory in agent_trajectories.items():
                    if len(trajectory['x']) > 1:
                        # Create line segments with fading alpha
                        points = np.array(list(zip(trajectory['x'], trajectory['y'])))
                        
                        # Create line collection with varying alpha
                        line_segments = []
                        colors_list = []
                        
                        for i in range(len(points) - 1):
                            line_segments.append([points[i], points[i + 1]])
                            alpha = max(0.1, 1.0 - (len(points) - i) * self.trail_alpha_decay)
                            color = list(plt.cm.viridis(i / len(points))[:3]) + [alpha]
                            colors_list.append(color)
                        
                        if line_segments:
                            lc = LineCollection(line_segments, colors=colors_list, linewidths=2, zorder=3)
                            ax.add_collection(lc)
                
                # Draw agents
                for agent in agents:
                    if agent.get('active', True):
                        # Agent color and size
                        color = self.colors[agent['type']]
                        size = self.agent_sizes[agent['type']]
                        symbol = self.agent_symbols[agent['type']]
                        
                        # Energy level affects opacity
                        energy = agent.get('energy_level', 1.0)
                        alpha = max(0.4, min(1.0, energy))
                        
                        # Draw agent
                        if symbol == '◆':  # Satellite - diamond
                            ax.scatter(agent['x'], agent['y'], 
                                     c=color, s=size, alpha=alpha,
                                     marker='D', edgecolors='black', linewidths=2,
                                     zorder=5)
                        elif symbol == '▲':  # UAV - triangle
                            ax.scatter(agent['x'], agent['y'], 
                                     c=color, s=size, alpha=alpha,
                                     marker='^', edgecolors='black', linewidths=2,
                                     zorder=5)
                        else:  # Ground station - square
                            ax.scatter(agent['x'], agent['y'], 
                                     c=color, s=size, alpha=alpha,
                                     marker='s', edgecolors='black', linewidths=2,
                                     zorder=5)
                        
                        # Agent ID label
                        ax.text(agent['x'], agent['y'] - 30, f"A{agent['id']}", 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                               zorder=6)
                
                # Add information panel
                info_text = f"Episode: {episode_num} | Step: {step} | Coverage: {total_coverage:.1%}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', va='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
                
                # Add legend
                legend_elements = []
                for agent_type in ['satellite', 'uav', 'ground_station']:
                    if agent_type == 'satellite':
                        marker = 'D'
                    elif agent_type == 'uav':
                        marker = '^'
                    else:
                        marker = 's'
                    
                    legend_elements.append(
                        plt.Line2D([0], [0], marker=marker, color='w', 
                                  markerfacecolor=self.colors[agent_type], 
                                  markersize=10, label=agent_type.replace('_', ' ').title(),
                                  markeredgecolor='black', markeredgewidth=1)
                    )
                
                legend_elements.append(
                    plt.Line2D([0], [0], marker='*', color='w',
                              markerfacecolor=self.colors['target_covered'],
                              markersize=12, label='Covered Target',
                              markeredgecolor='black', markeredgewidth=1)
                )
                legend_elements.append(
                    plt.Line2D([0], [0], marker='*', color='w',
                              markerfacecolor=self.colors['target_uncovered'],
                              markersize=12, label='Uncovered Target',
                              markeredgecolor='black', markeredgewidth=1)
                )
                
                ax.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(0.98, 0.98), fontsize=10)
            
            ax.set_title('Agent Movement and Coverage Animation', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('X Position (m)', fontsize=14)
            ax.set_ylabel('Y Position (m)', fontsize=14)
        
        # Create animation
        frames = min(len(episode_data), 300)  # Limit frames for reasonable file size
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=150, repeat=True)
        
        # Save animation
        save_path = os.path.join(self.output_dir, save_name)
        anim.save(save_path, writer='pillow', fps=8, dpi=100)
        plt.close()
        
        print(f"Agent movement animation saved: {save_path}")
        return save_path
    
    def create_coverage_heatmap_evolution(self,
                                        episode_data: List[Dict],
                                        save_name: str = 'coverage_heatmap_evolution.gif'):
        """
        Create animated heatmap showing coverage evolution over time
        
        Args:
            episode_data: List of episode states
            save_name: Filename for the animation
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Initialize coverage grid
        grid_size = 50
        x_grid = np.linspace(0, self.area_size, grid_size)
        y_grid = np.linspace(0, self.area_size, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        def animate(frame):
            for ax in [ax1, ax2]:
                ax.clear()
            
            if frame < len(episode_data):
                current_data = episode_data[frame]
                agents = current_data.get('agents', [])
                targets = current_data.get('targets', [])
                episode_num = current_data.get('episode', frame)
                coverage_rate = current_data.get('coverage_rate', 0)
                
                # Create coverage intensity grid
                coverage_grid = np.zeros((grid_size, grid_size))
                
                for i, x in enumerate(x_grid):
                    for j, y in enumerate(y_grid):
                        total_coverage = 0
                        for agent in agents:
                            if agent.get('active', True):
                                distance = np.sqrt((x - agent['x'])**2 + (y - agent['y'])**2)
                                coverage_radius = agent.get('coverage_radius', self.coverage_radii[agent['type']])
                                
                                if distance <= coverage_radius:
                                    # Coverage intensity decreases with distance
                                    intensity = 1.0 - (distance / coverage_radius) * 0.7
                                    total_coverage += intensity
                        
                        coverage_grid[j, i] = min(total_coverage, 1.0)  # Cap at 1.0
                
                # Plot 1: Coverage heatmap
                im1 = ax1.imshow(coverage_grid, extent=[0, self.area_size, 0, self.area_size],
                               origin='lower', cmap='YlOrRd', alpha=0.8, vmin=0, vmax=1)
                
                # Overlay agents
                for agent in agents:
                    if agent.get('active', True):
                        color = self.colors[agent['type']]
                        size = self.agent_sizes[agent['type']] // 3
                        
                        if agent['type'] == 'satellite':
                            marker = 'D'
                        elif agent['type'] == 'uav':
                            marker = '^'
                        else:
                            marker = 's'
                        
                        ax1.scatter(agent['x'], agent['y'], c=color, s=size,
                                  marker=marker, edgecolors='black', linewidths=1.5,
                                  zorder=5)
                
                # Overlay targets
                for target in targets:
                    color = 'yellow' if target.get('covered', False) else 'white'
                    size = 60 if target.get('covered', False) else 40
                    ax1.scatter(target['x'], target['y'], c=color, s=size,
                              marker='*', edgecolors='black', linewidths=1.5,
                              zorder=4)
                
                ax1.set_title(f'Coverage Heatmap - Episode {episode_num}', fontweight='bold')
                ax1.set_xlabel('X Position (m)')
                ax1.set_ylabel('Y Position (m)')
                
                # Add colorbar for heatmap
                cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
                cbar1.set_label('Coverage Intensity')
                
                # Plot 2: Agent positions with coverage circles
                ax2.set_xlim(0, self.area_size)
                ax2.set_ylim(0, self.area_size)
                ax2.set_aspect('equal')
                ax2.grid(True, alpha=0.3)
                
                # Draw coverage circles
                for agent in agents:
                    if agent.get('active', True):
                        coverage_radius = agent.get('coverage_radius', self.coverage_radii[agent['type']])
                        circle = Circle((agent['x'], agent['y']), coverage_radius,
                                      fill=False, edgecolor=self.colors[agent['type']],
                                      linewidth=2, alpha=0.7, zorder=2)
                        ax2.add_patch(circle)
                
                # Draw agents
                for agent in agents:
                    if agent.get('active', True):
                        color = self.colors[agent['type']]
                        size = self.agent_sizes[agent['type']]
                        
                        if agent['type'] == 'satellite':
                            marker = 'D'
                        elif agent['type'] == 'uav':
                            marker = '^'
                        else:
                            marker = 's'
                        
                        ax2.scatter(agent['x'], agent['y'], c=color, s=size,
                                  marker=marker, edgecolors='black', linewidths=2,
                                  zorder=5)
                        
                        # Agent ID
                        ax2.text(agent['x'], agent['y'] - 40, f"A{agent['id']}", 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # Draw targets
                for target in targets:
                    color = self.colors['target_covered'] if target.get('covered', False) else self.colors['target_uncovered']
                    size = 80 if target.get('covered', False) else 60
                    ax2.scatter(target['x'], target['y'], c=color, s=size,
                              marker='*', edgecolors='black', linewidths=1.5,
                              zorder=4)
                
                ax2.set_title(f'Coverage Areas - Rate: {coverage_rate:.1%}', fontweight='bold')
                ax2.set_xlabel('X Position (m)')
                ax2.set_ylabel('Y Position (m)')
            
            plt.tight_layout()
        
        # Create animation
        frames = min(len(episode_data), 200)
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, repeat=True)
        
        # Save animation
        save_path = os.path.join(self.output_dir, save_name)
        anim.save(save_path, writer='pillow', fps=6, dpi=80)
        plt.close()
        
        print(f"Coverage heatmap evolution saved: {save_path}")
        return save_path
    
    def create_trajectory_summary(self,
                                episode_data: List[Dict],
                                save_name: str = 'trajectory_summary.png'):
        """
        Create static summary of all agent trajectories
        
        Args:
            episode_data: List of episode states
            save_name: Filename for the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect all trajectory data
        agent_trajectories = {}
        all_targets = []
        
        for frame_data in episode_data:
            agents = frame_data.get('agents', [])
            targets = frame_data.get('targets', [])
            
            # Collect trajectories
            for agent in agents:
                agent_id = agent['id']
                if agent_id not in agent_trajectories:
                    agent_trajectories[agent_id] = {
                        'x': [], 'y': [], 'type': agent['type'], 'energy': []
                    }
                
                agent_trajectories[agent_id]['x'].append(agent['x'])
                agent_trajectories[agent_id]['y'].append(agent['y'])
                agent_trajectories[agent_id]['energy'].append(agent.get('energy_level', 1.0))
            
            # Collect targets (use last frame for final state)
            if targets:
                all_targets = targets
        
        # Plot 1: Complete trajectories with start/end points
        ax1.set_xlim(0, self.area_size)
        ax1.set_ylim(0, self.area_size)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        for agent_id, trajectory in agent_trajectories.items():
            if len(trajectory['x']) > 1:
                agent_type = trajectory['type']
                color = self.colors[agent_type]
                
                # Plot trajectory line
                ax1.plot(trajectory['x'], trajectory['y'], 
                        color=color, linewidth=2, alpha=0.7, label=f"{agent_type.title()} {agent_id}")
                
                # Start point (circle)
                ax1.scatter(trajectory['x'][0], trajectory['y'][0], 
                          c='green', s=100, marker='o', edgecolors='black', 
                          linewidths=2, zorder=5)
                
                # End point (square)
                ax1.scatter(trajectory['x'][-1], trajectory['y'][-1], 
                          c=color, s=150, marker='s', edgecolors='black', 
                          linewidths=2, zorder=5)
        
        # Add targets
        for target in all_targets:
            color = self.colors['target_covered'] if target.get('covered', False) else self.colors['target_uncovered']
            ax1.scatter(target['x'], target['y'], c=color, s=80,
                       marker='*', edgecolors='black', linewidths=1.5, zorder=4)
        
        ax1.set_title('Complete Agent Trajectories', fontweight='bold')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Coverage efficiency over time
        if len(episode_data) > 1:
            episodes = range(len(episode_data))
            coverage_rates = [ep.get('coverage_rate', 0) for ep in episode_data]
            
            ax2.plot(episodes, coverage_rates, linewidth=3, color=self.colors['satellite'])
            ax2.fill_between(episodes, coverage_rates, alpha=0.3, color=self.colors['satellite'])
            ax2.set_title('Coverage Rate Evolution', fontweight='bold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Coverage Rate')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # Plot 3: Agent energy levels over time
        for agent_id, trajectory in agent_trajectories.items():
            if trajectory['energy']:
                time_steps = range(len(trajectory['energy']))
                agent_type = trajectory['type']
                color = self.colors[agent_type]
                ax3.plot(time_steps, trajectory['energy'], 
                        linewidth=2, color=color, alpha=0.8, 
                        label=f"{agent_type.title()} {agent_id}")
        
        ax3.set_title('Agent Energy Levels', fontweight='bold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Energy Level')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Movement patterns heatmap
        # Create density map of agent positions
        all_x = []
        all_y = []
        for trajectory in agent_trajectories.values():
            all_x.extend(trajectory['x'])
            all_y.extend(trajectory['y'])
        
        if all_x and all_y:
            heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=30, 
                                                   range=[[0, self.area_size], [0, self.area_size]])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
            im = ax4.imshow(heatmap.T, extent=extent, origin='lower', 
                          cmap='Blues', alpha=0.8)
            ax4.set_title('Agent Position Density', fontweight='bold')
            ax4.set_xlabel('X Position (m)')
            ax4.set_ylabel('Y Position (m)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
            cbar.set_label('Visit Frequency')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Trajectory summary saved: {save_path}")
        return save_path