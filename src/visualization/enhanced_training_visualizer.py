"""
Enhanced Training Visualizer with Detailed Information Display
Integrates real-time training metrics with comprehensive visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon, Wedge
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Any
import os
import time
from collections import deque
from datetime import datetime


class EnhancedTrainingVisualizer:
    """Enhanced visualizer with detailed information and training integration"""
    
    def __init__(self, env_config: Dict, output_dir: str):
        """
        Initialize enhanced training visualizer
        
        Args:
            env_config: Environment configuration
            output_dir: Directory to save visualizations
        """
        self.env_config = env_config
        self.output_dir = output_dir
        self.video_dir = os.path.join(output_dir, 'training_videos')
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Environment parameters
        self.area_size = env_config.get('area_size', 1000)
        self.num_satellites = env_config.get('num_satellites', 2)
        self.num_uavs = env_config.get('num_uavs', 3)
        self.num_ground_stations = env_config.get('num_ground_stations', 2)
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.num_pois = env_config.get('num_pois', 12)
        
        # Agent configurations with detailed properties
        self.agent_configs = {
            'satellite': {
                'color': '#FF6B6B',
                'marker': '^',
                'size': 200,
                'coverage_radius': env_config.get('satellite_coverage_radius', 250),
                'max_speed': env_config.get('satellite_max_speed', 3),
                'altitude': 200,
                'name': 'Satellite',
                'icon': '🛰️'
            },
            'uav': {
                'color': '#4ECDC4',
                'marker': 'o',
                'size': 150,
                'coverage_radius': env_config.get('uav_coverage_radius', 120),
                'max_speed': env_config.get('uav_max_speed', 6),
                'altitude': 100,
                'energy_capacity': env_config.get('uav_energy_capacity', 1200),
                'name': 'UAV',
                'icon': '🚁'
            },
            'ground_station': {
                'color': '#95E77E',
                'marker': 's',
                'size': 120,
                'coverage_radius': env_config.get('ground_station_coverage_radius', 80),
                'max_speed': env_config.get('ground_station_max_speed', 2),
                'altitude': 0,
                'name': 'Ground Station',
                'icon': '📡'
            }
        }
        
        # POI configuration with detailed properties
        self.poi_configs = {
            1: {'color': '#FFE66D', 'name': 'Low Priority', 'reward': 10},
            2: {'color': '#FFA500', 'name': 'Medium-Low', 'reward': 20},
            3: {'color': '#FF7F50', 'name': 'Medium', 'reward': 30},
            4: {'color': '#FF6347', 'name': 'Medium-High', 'reward': 40},
            5: {'color': '#FF0000', 'name': 'High Priority', 'reward': 50}
        }
        
        # Training metrics storage
        self.training_metrics = {
            'episode': [],
            'total_reward': [],
            'coverage_rate': [],
            'energy_efficiency': [],
            'collision_count': [],
            'communication_efficiency': [],
            'task_completion_time': []
        }
        
        # Real-time data buffers
        self.position_buffer = deque(maxlen=50)
        self.reward_buffer = deque(maxlen=100)
        self.coverage_buffer = deque(maxlen=100)
        
        # Visualization settings
        self.fps = 30
        self.trail_length = 20
        self.show_communication = True
        self.show_coverage_circles = True
        self.show_3d_view = True
        self.show_metrics = True
        
    def create_training_visualization(self, episode_data: Dict, training_stats: Dict,
                                    algorithm_name: str, episode_num: int) -> str:
        """
        Create comprehensive training visualization with all details
        
        Args:
            episode_data: Episode position and coverage data
            training_stats: Training statistics and metrics
            algorithm_name: Name of the algorithm
            episode_num: Current episode number
            
        Returns:
            Path to generated video file
        """
        print(f"🎬 Creating enhanced training visualization for Episode {episode_num}...")
        
        # Set up the figure with multiple panels
        fig = self._setup_comprehensive_figure()
        
        # Create animation
        anim = self._create_training_animation(
            fig, episode_data, training_stats, algorithm_name, episode_num
        )
        
        # Save video
        output_name = f"{algorithm_name}_episode_{episode_num}_enhanced.gif"
        video_path = os.path.join(self.video_dir, output_name)
        
        writer = self._get_video_writer()
        print(f"💾 Saving enhanced visualization to: {video_path}")
        anim.save(video_path, writer=writer)
        plt.close(fig)
        
        print(f"✅ Enhanced visualization saved: {video_path}")
        return video_path
        
    def _setup_comprehensive_figure(self) -> plt.Figure:
        """Set up comprehensive figure with all panels"""
        fig = plt.figure(figsize=(24, 14), facecolor='#f0f0f0')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 4, figure=fig, 
                              height_ratios=[2, 1.5, 1],
                              width_ratios=[1.5, 1.5, 1, 1],
                              hspace=0.25, wspace=0.25)
        
        # Main 2D view (larger, left side)
        self.ax_main = fig.add_subplot(gs[:2, :2])
        self.ax_main.set_xlim(-50, self.area_size + 50)
        self.ax_main.set_ylim(-50, self.area_size + 50)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        self.ax_main.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        self.ax_main.set_title('Multi-Agent Coverage Mission', fontsize=16, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        self.ax_main.set_facecolor('#ffffff')
        
        # 3D view (top right)
        self.ax_3d = fig.add_subplot(gs[0, 2:], projection='3d')
        self.ax_3d.set_xlim(0, self.area_size)
        self.ax_3d.set_ylim(0, self.area_size)
        self.ax_3d.set_zlim(0, 300)
        self.ax_3d.set_xlabel('X (m)', fontsize=10)
        self.ax_3d.set_ylabel('Y (m)', fontsize=10)
        self.ax_3d.set_zlabel('Altitude (m)', fontsize=10)
        self.ax_3d.set_title('3D Agent Positions', fontsize=12, fontweight='bold')
        self.ax_3d.view_init(elev=25, azim=45)
        
        # Agent details panel (middle right)
        self.ax_agents = fig.add_subplot(gs[1, 2])
        self.ax_agents.axis('off')
        self.ax_agents.set_title('Agent Status', fontsize=12, fontweight='bold')
        
        # POI status panel (middle right, second column)
        self.ax_pois = fig.add_subplot(gs[1, 3])
        self.ax_pois.axis('off')
        self.ax_pois.set_title('POI Coverage Status', fontsize=12, fontweight='bold')
        
        # Training metrics plot (bottom left)
        self.ax_reward = fig.add_subplot(gs[2, 0])
        self.ax_reward.set_title('Episode Reward', fontsize=11, fontweight='bold')
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Cumulative Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Coverage rate plot (bottom middle-left)
        self.ax_coverage = fig.add_subplot(gs[2, 1])
        self.ax_coverage.set_title('Coverage Progress', fontsize=11, fontweight='bold')
        self.ax_coverage.set_xlabel('Step')
        self.ax_coverage.set_ylabel('Coverage %')
        self.ax_coverage.grid(True, alpha=0.3)
        
        # Performance metrics (bottom right)
        self.ax_metrics = fig.add_subplot(gs[2, 2:])
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        return fig
        
    def _create_training_animation(self, fig, episode_data, training_stats, 
                                  algorithm_name, episode_num):
        """Create comprehensive training animation"""
        
        position_history = episode_data.get('position_history', [])
        coverage_history = episode_data.get('coverage_history', [])
        reward_history = episode_data.get('reward_history', [])
        
        if not position_history:
            return None
            
        # Initialize all visual elements
        elements = self._initialize_visual_elements()
        
        # POI positions
        poi_positions = self._generate_poi_positions()
        
        # Agent trails storage
        agent_trails = {i: deque(maxlen=self.trail_length) for i in range(self.num_agents)}
        
        # Cumulative metrics
        cumulative_reward = []
        coverage_progress = []
        
        def animate(frame):
            """Animation function for each frame"""
            if frame >= len(position_history):
                return []
                
            current_positions = position_history[frame]
            current_coverage = coverage_history[frame] if frame < len(coverage_history) else []
            
            # Clear previous 3D elements
            self.ax_3d.clear()
            self.ax_3d.set_xlim(0, self.area_size)
            self.ax_3d.set_ylim(0, self.area_size)
            self.ax_3d.set_zlim(0, 300)
            self.ax_3d.set_xlabel('X (m)', fontsize=10)
            self.ax_3d.set_ylabel('Y (m)', fontsize=10)
            self.ax_3d.set_zlabel('Altitude (m)', fontsize=10)
            
            # Update main 2D view
            self._update_2d_view(elements, current_positions, current_coverage, 
                               poi_positions, agent_trails, frame)
            
            # Update 3D view
            self._update_3d_view(current_positions, poi_positions, current_coverage)
            
            # Update agent status panel
            self._update_agent_status(current_positions, frame)
            
            # Update POI status panel
            self._update_poi_status(poi_positions, current_coverage)
            
            # Update training metrics
            if frame < len(reward_history):
                cumulative_reward.append(sum(reward_history[:frame+1]))
                coverage_progress.append(sum(current_coverage) / len(current_coverage) * 100 
                                       if current_coverage else 0)
                
                self._update_metrics_plots(cumulative_reward, coverage_progress)
                
            # Update performance metrics panel
            self._update_performance_metrics(frame, cumulative_reward, coverage_progress,
                                           algorithm_name, episode_num)
            
            # Update title with current info
            self.ax_main.set_title(
                f'{algorithm_name.replace("_", " ").title()} - Episode {episode_num} - Step {frame}',
                fontsize=16, fontweight='bold'
            )
            
            return []
            
        # Create animation
        anim = FuncAnimation(
            fig, animate, frames=len(position_history),
            interval=1000/self.fps, blit=False, repeat=True
        )
        
        return anim
        
    def _initialize_visual_elements(self) -> Dict:
        """Initialize all visual elements"""
        elements = {
            'agent_markers': [],
            'agent_trails': [],
            'coverage_circles': [],
            'poi_patches': [],
            'communication_lines': []
        }
        
        # Create agent markers and coverage circles
        for i in range(self.num_agents):
            agent_type = self._get_agent_type(i)
            config = self.agent_configs[agent_type]
            
            # Agent marker
            marker, = self.ax_main.plot([], [], config['marker'],
                                       color=config['color'],
                                       markersize=np.sqrt(config['size'])/2,
                                       markeredgewidth=2,
                                       markeredgecolor='white',
                                       zorder=20,
                                       label=f"{config['icon']} {config['name']} {i+1}")
            elements['agent_markers'].append(marker)
            
            # Coverage circle
            circle = Circle((0, 0), config['coverage_radius'],
                          fill=False, edgecolor=config['color'],
                          alpha=0.3, linewidth=2, linestyle='--')
            self.ax_main.add_patch(circle)
            elements['coverage_circles'].append(circle)
            
        # Add legend for agents
        self.ax_main.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                          fontsize=10, framealpha=0.9)
        
        return elements
        
    def _update_2d_view(self, elements, positions, coverage, poi_positions, 
                       agent_trails, frame):
        """Update the main 2D view"""
        
        # Draw POIs first
        for idx, (x, y, priority) in enumerate(poi_positions):
            if idx < len(elements['poi_patches']):
                elements['poi_patches'][idx].remove()
                
            config = self.poi_configs[priority]
            is_covered = idx < len(coverage) and coverage[idx]
            
            # Create POI visual
            if is_covered:
                # Covered POI (checkmark overlay)
                rect = FancyBboxPatch(
                    (x-25, y-25), 50, 50,
                    boxstyle="round,pad=5",
                    facecolor=config['color'],
                    edgecolor='green',
                    alpha=0.4,
                    linewidth=3
                )
                self.ax_main.add_patch(rect)
                # Add checkmark
                self.ax_main.text(x, y, '✓', fontsize=20, color='green',
                                ha='center', va='center', fontweight='bold')
            else:
                # Uncovered POI
                rect = FancyBboxPatch(
                    (x-25, y-25), 50, 50,
                    boxstyle="round,pad=5",
                    facecolor=config['color'],
                    edgecolor='black',
                    alpha=0.8,
                    linewidth=2
                )
                self.ax_main.add_patch(rect)
                # Add POI label
                self.ax_main.text(x, y, f'P{idx+1}\n{priority}★', 
                                fontsize=10, color='black',
                                ha='center', va='center', fontweight='bold')
                
            if len(elements['poi_patches']) <= idx:
                elements['poi_patches'].append(rect)
            else:
                elements['poi_patches'][idx] = rect
                
        # Update agents
        for i in range(self.num_agents):
            agent_key = f'agent_{i}'
            if agent_key in positions:
                x, y, z = positions[agent_key]
                
                # Update marker position
                elements['agent_markers'][i].set_data([x], [y])
                
                # Update trail
                agent_trails[i].append((x, y))
                if len(agent_trails[i]) > 1:
                    # Draw trail with gradient
                    trail_points = list(agent_trails[i])
                    for j in range(len(trail_points) - 1):
                        alpha = (j + 1) / len(trail_points) * 0.5
                        self.ax_main.plot(
                            [trail_points[j][0], trail_points[j+1][0]],
                            [trail_points[j][1], trail_points[j+1][1]],
                            color=self.agent_configs[self._get_agent_type(i)]['color'],
                            alpha=alpha, linewidth=2
                        )
                        
                # Update coverage circle
                agent_type = self._get_agent_type(i)
                radius = self.agent_configs[agent_type]['coverage_radius']
                elements['coverage_circles'][i].center = (x, y)
                elements['coverage_circles'][i].radius = radius
                
        # Draw communication links
        if self.show_communication:
            self._draw_communication_links(positions)
            
    def _update_3d_view(self, positions, poi_positions, coverage):
        """Update the 3D view panel"""
        
        # Draw ground grid
        xx, yy = np.meshgrid(np.linspace(0, self.area_size, 10),
                            np.linspace(0, self.area_size, 10))
        self.ax_3d.plot_wireframe(xx, yy, np.zeros_like(xx), 
                                 color='gray', alpha=0.2, linewidth=0.5)
        
        # Draw POIs as vertical bars
        for idx, (x, y, priority) in enumerate(poi_positions):
            config = self.poi_configs[priority]
            is_covered = idx < len(coverage) and coverage[idx]
            
            # Draw POI pillar
            z_height = priority * 10
            self.ax_3d.bar3d(x-15, y-15, 0, 30, 30, z_height,
                           color=config['color'], 
                           alpha=0.3 if is_covered else 0.8)
            
        # Draw agents
        for i in range(self.num_agents):
            agent_key = f'agent_{i}'
            if agent_key in positions:
                x, y, z = positions[agent_key]
                agent_type = self._get_agent_type(i)
                config = self.agent_configs[agent_type]
                
                # Draw agent
                self.ax_3d.scatter(x, y, z, c=config['color'],
                                 s=config['size'], marker=config['marker'],
                                 edgecolors='white', linewidth=2)
                
                # Draw coverage area on ground
                theta = np.linspace(0, 2*np.pi, 30)
                radius = config['coverage_radius']
                x_circle = x + radius * np.cos(theta)
                y_circle = y + radius * np.sin(theta)
                z_circle = np.zeros_like(x_circle)
                self.ax_3d.plot(x_circle, y_circle, z_circle,
                              color=config['color'], alpha=0.3, linewidth=1)
                
                # Draw vertical line to ground
                self.ax_3d.plot([x, x], [y, y], [0, z],
                              color=config['color'], alpha=0.5, linewidth=1)
                
    def _update_agent_status(self, positions, frame):
        """Update agent status panel"""
        self.ax_agents.clear()
        self.ax_agents.axis('off')
        self.ax_agents.set_title('Agent Status', fontsize=12, fontweight='bold')
        
        y_pos = 0.9
        for i in range(self.num_agents):
            agent_key = f'agent_{i}'
            agent_type = self._get_agent_type(i)
            config = self.agent_configs[agent_type]
            
            if agent_key in positions:
                x, y, z = positions[agent_key]
                
                # Agent info text
                info_text = (f"{config['icon']} {config['name']} {i+1}\n"
                           f"Pos: ({x:.0f}, {y:.0f}, {z:.0f})\n"
                           f"Coverage: {config['coverage_radius']}m\n"
                           f"Speed: {config['max_speed']}m/s")
                
                if agent_type == 'uav' and 'energy_capacity' in config:
                    # Add energy info for UAVs
                    energy_percent = max(0, 100 - (frame * 2))  # Simulate energy consumption
                    info_text += f"\nEnergy: {energy_percent}%"
                    
                self.ax_agents.text(0.05, y_pos, info_text,
                                  fontsize=9, color=config['color'],
                                  transform=self.ax_agents.transAxes,
                                  fontweight='bold',
                                  bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor='white', alpha=0.8))
                y_pos -= 0.25
                
    def _update_poi_status(self, poi_positions, coverage):
        """Update POI coverage status panel"""
        self.ax_pois.clear()
        self.ax_pois.axis('off')
        self.ax_pois.set_title('POI Coverage Status', fontsize=12, fontweight='bold')
        
        total_pois = len(poi_positions)
        covered_pois = sum(coverage) if coverage else 0
        coverage_rate = (covered_pois / total_pois * 100) if total_pois > 0 else 0
        
        # Overall status
        self.ax_pois.text(0.5, 0.9, f"Coverage: {covered_pois}/{total_pois} ({coverage_rate:.1f}%)",
                        fontsize=11, fontweight='bold', ha='center',
                        transform=self.ax_pois.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor='lightgreen' if coverage_rate > 50 else 'lightyellow',
                                alpha=0.8))
        
        # Individual POI status
        y_pos = 0.75
        for idx, (x, y, priority) in enumerate(poi_positions[:8]):  # Show first 8 POIs
            config = self.poi_configs[priority]
            is_covered = idx < len(coverage) and coverage[idx]
            
            status_symbol = '✅' if is_covered else '⭕'
            status_text = f"{status_symbol} POI-{idx+1} (P{priority})"
            
            self.ax_pois.text(0.1, y_pos, status_text,
                            fontsize=9, color=config['color'],
                            transform=self.ax_pois.transAxes,
                            fontweight='bold' if is_covered else 'normal')
            y_pos -= 0.08
            
    def _update_metrics_plots(self, cumulative_reward, coverage_progress):
        """Update the metrics plots"""
        
        # Update reward plot
        self.ax_reward.clear()
        self.ax_reward.plot(cumulative_reward, color='#FF6B6B', linewidth=2)
        self.ax_reward.fill_between(range(len(cumulative_reward)), 
                                   cumulative_reward, alpha=0.3, color='#FF6B6B')
        self.ax_reward.set_title('Episode Reward', fontsize=11, fontweight='bold')
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Cumulative Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Update coverage plot
        self.ax_coverage.clear()
        self.ax_coverage.plot(coverage_progress, color='#4ECDC4', linewidth=2)
        self.ax_coverage.fill_between(range(len(coverage_progress)),
                                     coverage_progress, alpha=0.3, color='#4ECDC4')
        self.ax_coverage.set_title('Coverage Progress', fontsize=11, fontweight='bold')
        self.ax_coverage.set_xlabel('Step')
        self.ax_coverage.set_ylabel('Coverage %')
        self.ax_coverage.set_ylim(0, 100)
        self.ax_coverage.grid(True, alpha=0.3)
        
    def _update_performance_metrics(self, frame, cumulative_reward, coverage_progress,
                                   algorithm_name, episode_num):
        """Update performance metrics panel"""
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        # Calculate metrics
        current_reward = cumulative_reward[-1] if cumulative_reward else 0
        current_coverage = coverage_progress[-1] if coverage_progress else 0
        avg_reward_per_step = current_reward / (frame + 1) if frame > 0 else 0
        
        # Display metrics
        metrics_text = (
            f"🎯 Algorithm: {algorithm_name.replace('_', ' ').title()}\n"
            f"📊 Episode: {episode_num}\n"
            f"⏱️ Current Step: {frame}\n"
            f"💰 Total Reward: {current_reward:.1f}\n"
            f"📈 Avg Reward/Step: {avg_reward_per_step:.2f}\n"
            f"🎯 Coverage Rate: {current_coverage:.1f}%\n"
            f"⚡ Efficiency Score: {(current_coverage * current_reward / 100):.1f}"
        )
        
        self.ax_metrics.text(0.1, 0.5, metrics_text,
                           fontsize=11, transform=self.ax_metrics.transAxes,
                           verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.5",
                                   facecolor='white', alpha=0.9))
        
    def _draw_communication_links(self, positions):
        """Draw communication links between agents"""
        comm_range = self.env_config.get('communication_range', 200)
        
        for i in range(self.num_agents):
            agent1_key = f'agent_{i}'
            if agent1_key not in positions:
                continue
                
            x1, y1, _ = positions[agent1_key]
            
            for j in range(i+1, self.num_agents):
                agent2_key = f'agent_{j}'
                if agent2_key not in positions:
                    continue
                    
                x2, y2, _ = positions[agent2_key]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if dist < comm_range:
                    # Draw communication link
                    alpha = max(0.1, 1 - dist/comm_range) * 0.5
                    self.ax_main.plot([x1, x2], [y1, y2], 'b--',
                                    alpha=alpha, linewidth=1)
                    
    def _get_agent_type(self, agent_idx: int) -> str:
        """Get agent type based on index"""
        if agent_idx < self.num_satellites:
            return 'satellite'
        elif agent_idx < self.num_satellites + self.num_uavs:
            return 'uav'
        else:
            return 'ground_station'
            
    def _generate_poi_positions(self) -> List[Tuple[float, float, int]]:
        """Generate POI positions with priorities"""
        poi_positions = []
        
        for i in range(self.num_pois):
            # Create strategic POI placement
            if i < 3:  # High priority center POIs
                angle = i * 2 * np.pi / 3
                x = self.area_size/2 + 150 * np.cos(angle)
                y = self.area_size/2 + 150 * np.sin(angle)
                priority = 5
            elif i < 6:  # Medium priority middle ring
                angle = (i-3) * 2 * np.pi / 3
                x = self.area_size/2 + 300 * np.cos(angle)
                y = self.area_size/2 + 300 * np.sin(angle)
                priority = 3
            else:  # Lower priority outer ring
                angle = (i-6) * 2 * np.pi / (self.num_pois - 6)
                x = self.area_size/2 + 400 * np.cos(angle)
                y = self.area_size/2 + 400 * np.sin(angle)
                priority = 1 + (i % 2) * 2
                
            # Ensure within bounds
            x = max(50, min(x, self.area_size - 50))
            y = max(50, min(y, self.area_size - 50))
            
            poi_positions.append((x, y, priority))
            
        return poi_positions
        
    def _get_video_writer(self):
        """Get appropriate video writer"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                return FFMpegWriter(fps=self.fps, bitrate=3000)
        except:
            pass
            
        # Fallback to Pillow
        return PillowWriter(fps=min(self.fps, 10))
        
    def integrate_with_training(self, trainer):
        """
        Integrate visualizer with training process
        
        Args:
            trainer: Training object with step callback
        """
        def on_episode_end(episode, episode_data, metrics):
            """Callback for episode end"""
            # Generate visualization for significant episodes
            if episode % 10 == 0 or episode == 1:  # Every 10 episodes and first episode
                self.create_training_visualization(
                    episode_data,
                    metrics,
                    trainer.algorithm_name,
                    episode
                )
                
        # Register callback
        trainer.register_callback('episode_end', on_episode_end)
        
    def create_training_summary(self, all_episodes_data: List[Dict], 
                              algorithm_name: str) -> str:
        """
        Create a summary visualization of entire training
        
        Args:
            all_episodes_data: List of all episode data
            algorithm_name: Name of the algorithm
            
        Returns:
            Path to summary video
        """
        # Implementation for creating training summary
        pass