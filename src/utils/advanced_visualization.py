#!/usr/bin/env python3
"""
Advanced 3D Visualization System for SkyNetRL
Professional-grade visualization with smooth animations, trajectories, and comprehensive metrics display
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Set high-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class AgentState:
    """Complete agent state with history"""
    agent_id: int
    agent_type: str
    position: Tuple[float, float, float]  # x, y, z
    velocity: Tuple[float, float, float]
    energy: float
    max_energy: float
    coverage_radius: float
    is_active: bool
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    communication_links: List[int] = field(default_factory=list)


@dataclass 
class POIState:
    """POI state with coverage history"""
    poi_id: int
    position: Tuple[float, float]
    priority: float
    is_covered: bool
    coverage_time: float
    covering_agents: List[int] = field(default_factory=list)


@dataclass
class EnvironmentSnapshot:
    """Complete environment state snapshot"""
    episode: int
    step: int
    timestamp: float
    agents: List[AgentState]
    pois: List[POIState]
    obstacles: List[Tuple[float, float, float]]  # x, y, radius
    coverage_rate: float
    total_reward: float
    energy_consumption: float
    communication_efficiency: float
    cooperation_index: float
    
    # Area and bounds
    area_size: float
    height_bounds: Tuple[float, float] = (0, 200)


class AdvancedVisualizationSystem:
    """
    Professional 3D visualization system with smooth animations and comprehensive metrics
    """
    
    def __init__(self, output_manager=None, fps: int = 30, dpi: int = 150):
        self.output_manager = output_manager
        self.fps = fps
        self.dpi = dpi
        self.snapshots = []
        self.temp_dir = tempfile.mkdtemp()
        
        # Visual configuration
        self.colors = {
            'satellite': '#FF4081',   # Bright Pink
            'uav': '#00BCD4',        # Cyan  
            'ground_station': '#4CAF50', # Green
            'poi_uncovered': '#FF9800',  # Orange
            'poi_covered': '#8BC34A',    # Light Green
            'poi_priority': '#F44336',   # Red for high priority
            'obstacle': '#757575',       # Gray
            'trajectory': '#2196F3',     # Blue
            'communication': '#FFEB3B',  # Yellow
            'coverage': '#E1BEE7'        # Light Purple
        }
        
        # 3D Heights for different agent types
        self.agent_heights = {
            'satellite': 150,
            'uav': 100,
            'ground_station': 5
        }
        
        print("🎬 Advanced 3D Visualization System initialized")
        print(f"   🎯 FPS: {fps}, DPI: {dpi}")
        print(f"   📁 Temp dir: {self.temp_dir}")
    
    def record_snapshot(self, env_state: Dict[str, Any], episode: int, step: int):
        """Record a detailed environment snapshot"""
        
        # Generate realistic agent data
        agents = []
        num_agents = env_state.get('num_agents', 4)
        area_size = env_state.get('area_size', 400)
        
        for i in range(num_agents):
            # Determine agent type
            if i < 1:
                agent_type = 'satellite'
            elif i < 3:
                agent_type = 'uav'  
            else:
                agent_type = 'ground_station'
            
            # Generate realistic movement with continuity
            if step > 0 and len(self.snapshots) > 0:
                # Continue from last position with smooth movement
                last_snapshot = self.snapshots[-1]
                if i < len(last_snapshot.agents):
                    last_pos = last_snapshot.agents[i].position
                    # Add smooth movement
                    dx = np.random.normal(0, 3)
                    dy = np.random.normal(0, 3)
                    dz = np.random.normal(0, 1)
                    
                    new_x = max(10, min(area_size - 10, last_pos[0] + dx))
                    new_y = max(10, min(area_size - 10, last_pos[1] + dy))
                    base_height = self.agent_heights[agent_type]
                    new_z = max(base_height - 20, min(base_height + 20, last_pos[2] + dz))
                    position = (new_x, new_y, new_z)
                    
                    # Copy trajectory from last snapshot
                    trajectory = last_snapshot.agents[i].trajectory.copy()
                    trajectory.append(last_pos)
                else:
                    position = self._generate_initial_position(agent_type, area_size)
                    trajectory = deque(maxlen=50)
            else:
                position = self._generate_initial_position(agent_type, area_size)
                trajectory = deque(maxlen=50)
            
            # Generate velocity
            if len(trajectory) > 1:
                last_pos = list(trajectory)[-1] if trajectory else position
                velocity = (
                    position[0] - last_pos[0],
                    position[1] - last_pos[1], 
                    position[2] - last_pos[2]
                )
            else:
                velocity = (0, 0, 0)
            
            # Agent properties
            max_energy = {'satellite': 1000, 'uav': 600, 'ground_station': float('inf')}[agent_type]
            coverage_radius = {'satellite': 200, 'uav': 120, 'ground_station': 80}[agent_type]
            
            agents.append(AgentState(
                agent_id=i,
                agent_type=agent_type,
                position=position,
                velocity=velocity,
                energy=max_energy * (0.5 + 0.5 * np.random.random()),
                max_energy=max_energy,
                coverage_radius=coverage_radius,
                is_active=True,
                trajectory=trajectory,
                communication_links=self._generate_communication_links(i, num_agents)
            ))
        
        # Generate POIs
        pois = []
        num_pois = env_state.get('num_pois', 6)
        for i in range(num_pois):
            position = (
                np.random.uniform(30, area_size - 30),
                np.random.uniform(30, area_size - 30)
            )
            priority = 1.0 + np.random.random()  # 1.0 to 2.0
            
            # Check coverage by agents
            is_covered = False
            covering_agents = []
            for agent in agents:
                dist = np.sqrt((agent.position[0] - position[0])**2 + 
                             (agent.position[1] - position[1])**2)
                if dist <= agent.coverage_radius:
                    is_covered = True
                    covering_agents.append(agent.agent_id)
            
            pois.append(POIState(
                poi_id=i,
                position=position,
                priority=priority,
                is_covered=is_covered,
                coverage_time=np.random.uniform(0, step),
                covering_agents=covering_agents
            ))
        
        # Generate obstacles
        obstacles = []
        num_obstacles = env_state.get('num_obstacles', 1)
        for i in range(num_obstacles):
            x = 100 + np.random.random() * (area_size - 200)
            y = 100 + np.random.random() * (area_size - 200)
            radius = 30 + np.random.random() * 30
            obstacles.append((x, y, radius))
        
        # Calculate metrics
        coverage_rate = sum(poi.is_covered for poi in pois) / len(pois) if pois else 0
        total_reward = env_state.get('total_reward', coverage_rate * 100 + np.random.randn() * 5)
        energy_consumption = sum(agent.max_energy - agent.energy for agent in agents if agent.max_energy != float('inf'))
        communication_efficiency = len([link for agent in agents for link in agent.communication_links]) / (num_agents * (num_agents - 1))
        cooperation_index = coverage_rate * communication_efficiency
        
        snapshot = EnvironmentSnapshot(
            episode=episode,
            step=step,
            timestamp=step * 0.1,  # 0.1s per step
            agents=agents,
            pois=pois,
            obstacles=obstacles,
            coverage_rate=coverage_rate,
            total_reward=total_reward,
            energy_consumption=energy_consumption,
            communication_efficiency=communication_efficiency,
            cooperation_index=cooperation_index,
            area_size=area_size
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots to manage memory
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-500:]
    
    def _generate_initial_position(self, agent_type: str, area_size: float) -> Tuple[float, float, float]:
        """Generate realistic initial position for agent type"""
        x = 50 + np.random.random() * (area_size - 100)
        y = 50 + np.random.random() * (area_size - 100) 
        z = self.agent_heights[agent_type] + np.random.randn() * 10
        return (x, y, z)
    
    def _generate_communication_links(self, agent_id: int, num_agents: int) -> List[int]:
        """Generate realistic communication links"""
        max_links = min(3, num_agents - 1)
        if max_links <= 0:
            return []
        num_links = np.random.randint(0, max_links + 1)
        possible_links = [i for i in range(num_agents) if i != agent_id]
        if num_links == 0 or len(possible_links) == 0:
            return []
        return np.random.choice(possible_links, size=min(num_links, len(possible_links)), replace=False).tolist()
    
    def create_3d_episode_video(self, episode: int, algorithm: str, 
                               duration: float = 10.0) -> Optional[str]:
        """Create high-quality 3D episode video"""
        
        episode_snapshots = [s for s in self.snapshots if s.episode == episode]
        
        if len(episode_snapshots) < 5:
            print(f"⚠️ Not enough snapshots for episode {episode} ({len(episode_snapshots)})")
            return None
        
        print(f"🎬 Creating 3D video for episode {episode} with {len(episode_snapshots)} frames")
        
        # Set up 3D figure with professional styling
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        
        # Create 3D subplot
        ax_3d = fig.add_subplot(221, projection='3d', facecolor='black')
        
        # Create 2D overview subplot
        ax_2d = fig.add_subplot(222, facecolor='black')
        
        # Create metrics subplot  
        ax_metrics = fig.add_subplot(223, facecolor='black')
        
        # Create info subplot
        ax_info = fig.add_subplot(224, facecolor='black')
        
        # Style all subplots
        for ax in [ax_2d, ax_metrics, ax_info]:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        ax_3d.grid(True, alpha=0.3, color='white')
        
        # Animation function
        def animate(frame_idx):
            if frame_idx >= len(episode_snapshots):
                return []
                
            snapshot = episode_snapshots[frame_idx]
            
            # Clear all subplots
            ax_3d.clear()
            ax_2d.clear() 
            ax_metrics.clear()
            ax_info.clear()
            
            # Setup 3D plot
            self._setup_3d_plot(ax_3d, snapshot)
            self._draw_3d_environment(ax_3d, snapshot)
            
            # Setup 2D overview
            self._setup_2d_plot(ax_2d, snapshot)
            self._draw_2d_overview(ax_2d, snapshot)
            
            # Draw metrics
            self._draw_metrics_plot(ax_metrics, episode_snapshots[:frame_idx + 1])
            
            # Draw info panel
            self._draw_info_panel(ax_info, snapshot, algorithm)
            
            # Main title
            fig.suptitle(f'SkyNetRL: {algorithm.upper()} - Episode {episode}, Step {snapshot.step}', 
                        fontsize=20, fontweight='bold', color='white')
            
            plt.tight_layout()
            
            return []
        
        # Create animation
        frames = len(episode_snapshots)
        interval = int(1000 / self.fps)  # milliseconds per frame
        
        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval,
            blit=False, repeat=True
        )
        
        # Save as high-quality GIF
        output_path = os.path.join(self.temp_dir, f"episode_{episode:03d}_3d.gif")
        
        # Use pillow writer for better GIF quality
        try:
            writer = animation.PillowWriter(fps=self.fps, bitrate=1800)
            anim.save(output_path, writer=writer, dpi=self.dpi)
            print(f"🎥 Created 3D video: {output_path}")
            plt.close(fig)
            return output_path
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            plt.close(fig)
            return None
    
    def _setup_3d_plot(self, ax, snapshot):
        """Setup 3D plot with proper bounds and styling"""
        area = snapshot.area_size
        ax.set_xlim(0, area)
        ax.set_ylim(0, area) 
        ax.set_zlim(0, 200)
        
        ax.set_xlabel('X Position (m)', color='white', fontsize=10)
        ax.set_ylabel('Y Position (m)', color='white', fontsize=10)
        ax.set_zlabel('Height (m)', color='white', fontsize=10)
        
        ax.set_title('3D Environment View', color='white', fontsize=12, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Style
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3, color='white')
        ax.tick_params(colors='white', labelsize=8)
    
    def _draw_3d_environment(self, ax, snapshot):
        """Draw complete 3D environment with all elements"""
        
        # Draw ground plane
        xx, yy = np.meshgrid(np.linspace(0, snapshot.area_size, 5),
                            np.linspace(0, snapshot.area_size, 5))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        # Draw obstacles as 3D cylinders
        for obs in snapshot.obstacles:
            x, y, radius = obs
            # Create cylinder
            theta = np.linspace(0, 2*np.pi, 20)
            z = np.linspace(0, 50, 10)
            theta_mesh, z_mesh = np.meshgrid(theta, z)
            x_mesh = x + radius * np.cos(theta_mesh)
            y_mesh = y + radius * np.sin(theta_mesh)
            ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.6, color=self.colors['obstacle'])
        
        # Draw POIs with height indicators
        for poi in snapshot.pois:
            x, y = poi.position
            color = self.colors['poi_covered'] if poi.is_covered else self.colors['poi_uncovered']
            
            # POI base
            ax.scatter([x], [y], [0], c=[color], s=100, marker='^', alpha=0.8)
            
            # Priority height indicator
            height = poi.priority * 30
            ax.plot([x, x], [y, y], [0, height], color=color, linewidth=3, alpha=0.7)
            
            # Coverage indicator
            if poi.is_covered:
                # Draw coverage ring
                theta = np.linspace(0, 2*np.pi, 20)
                ring_x = x + 20 * np.cos(theta)
                ring_y = y + 20 * np.sin(theta)
                ring_z = np.full_like(ring_x, 2)
                ax.plot(ring_x, ring_y, ring_z, color=self.colors['poi_covered'], linewidth=2)
        
        # Draw agents with trajectories and coverage
        for agent in snapshot.agents:
            x, y, z = agent.position
            color = self.colors[agent.agent_type]
            
            # Agent marker with size based on type
            sizes = {'satellite': 200, 'uav': 150, 'ground_station': 100}
            markers = {'satellite': 's', 'uav': 'o', 'ground_station': 'D'}
            
            ax.scatter([x], [y], [z], c=[color], s=sizes[agent.agent_type], 
                      marker=markers[agent.agent_type], alpha=0.9, edgecolors='white', linewidths=2)
            
            # Draw trajectory
            if len(agent.trajectory) > 1:
                traj = list(agent.trajectory) + [agent.position]
                traj_x = [p[0] for p in traj]
                traj_y = [p[1] for p in traj]
                traj_z = [p[2] for p in traj]
                ax.plot(traj_x, traj_y, traj_z, color=color, alpha=0.6, linewidth=2)
            
            # Draw coverage area
            if agent.agent_type in ['uav', 'satellite']:
                theta = np.linspace(0, 2*np.pi, 20)
                coverage_x = x + agent.coverage_radius * np.cos(theta)
                coverage_y = y + agent.coverage_radius * np.sin(theta)
                coverage_z = np.full_like(coverage_x, z - 5)
                ax.plot(coverage_x, coverage_y, coverage_z, color=color, alpha=0.4, linewidth=1)
            
            # Draw communication links
            for link_id in agent.communication_links:
                if link_id < len(snapshot.agents):
                    other = snapshot.agents[link_id]
                    ax.plot([x, other.position[0]], [y, other.position[1]], 
                           [z, other.position[2]], color=self.colors['communication'], 
                           alpha=0.5, linewidth=1, linestyle='--')
        
        # Add legend
        legend_elements = []
        for agent_type, color in [('satellite', self.colors['satellite']), 
                                 ('uav', self.colors['uav']),
                                 ('ground_station', self.colors['ground_station'])]:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=agent_type.replace('_', ' ').title()))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _setup_2d_plot(self, ax, snapshot):
        """Setup 2D overview plot"""
        ax.set_xlim(0, snapshot.area_size)
        ax.set_ylim(0, snapshot.area_size)
        ax.set_xlabel('X Position (m)', color='white', fontsize=10)
        ax.set_ylabel('Y Position (m)', color='white', fontsize=10)
        ax.set_title('2D Overview & Coverage', color='white', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='white')
    
    def _draw_2d_overview(self, ax, snapshot):
        """Draw 2D overview with coverage visualization"""
        
        # Draw obstacles
        for obs in snapshot.obstacles:
            x, y, radius = obs
            circle = Circle((x, y), radius, color=self.colors['obstacle'], alpha=0.6)
            ax.add_patch(circle)
        
        # Draw POIs with coverage status
        for poi in snapshot.pois:
            x, y = poi.position
            color = self.colors['poi_covered'] if poi.is_covered else self.colors['poi_uncovered']
            size = poi.priority * 100 + 50
            
            ax.scatter(x, y, c=color, s=size, marker='^', alpha=0.8, edgecolors='white')
            
            # Show POI ID
            ax.text(x + 10, y + 10, f'POI{poi.poi_id}', color='white', fontsize=8)
        
        # Draw agent coverage areas first (background)
        for agent in snapshot.agents:
            x, y, _ = agent.position
            if agent.agent_type in ['uav', 'satellite']:
                circle = Circle((x, y), agent.coverage_radius, 
                              color=self.colors[agent.agent_type], alpha=0.1)
                ax.add_patch(circle)
        
        # Draw agents
        for agent in snapshot.agents:
            x, y, _ = agent.position
            color = self.colors[agent.agent_type]
            sizes = {'satellite': 200, 'uav': 150, 'ground_station': 100}
            markers = {'satellite': 's', 'uav': 'o', 'ground_station': 'D'}
            
            ax.scatter(x, y, c=color, s=sizes[agent.agent_type], 
                      marker=markers[agent.agent_type], alpha=0.9, 
                      edgecolors='white', linewidths=2)
            
            # Draw trajectory
            if len(agent.trajectory) > 1:
                traj = list(agent.trajectory) + [agent.position]
                traj_x = [p[0] for p in traj]
                traj_y = [p[1] for p in traj]
                ax.plot(traj_x, traj_y, color=color, alpha=0.6, linewidth=2)
            
            # Show agent ID
            ax.text(x + 15, y + 15, f'A{agent.agent_id}', color='white', fontsize=8, fontweight='bold')
            
            # Draw communication links
            for link_id in agent.communication_links:
                if link_id < len(snapshot.agents):
                    other = snapshot.agents[link_id]
                    ax.plot([x, other.position[0]], [y, other.position[1]], 
                           color=self.colors['communication'], alpha=0.6, linewidth=1, linestyle='--')
    
    def _draw_metrics_plot(self, ax, snapshots):
        """Draw real-time metrics visualization"""
        if len(snapshots) < 2:
            return
            
        steps = [s.step for s in snapshots]
        coverage = [s.coverage_rate * 100 for s in snapshots]
        rewards = [s.total_reward for s in snapshots]
        energy = [s.energy_consumption for s in snapshots]
        cooperation = [s.cooperation_index * 100 for s in snapshots]
        
        # Plot multiple metrics
        ax.plot(steps, coverage, color='#4CAF50', linewidth=2, label='Coverage %', alpha=0.8)
        ax.plot(steps, cooperation, color='#FF9800', linewidth=2, label='Cooperation %', alpha=0.8)
        
        # Create second y-axis for reward
        ax2 = ax.twinx()
        ax2.plot(steps, rewards, color='#2196F3', linewidth=2, label='Reward', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor='#2196F3', colors='white')
        
        ax.set_xlabel('Step', color='white', fontsize=10)
        ax.set_ylabel('Coverage & Cooperation (%)', color='white', fontsize=10)
        ax2.set_ylabel('Reward', color='#2196F3', fontsize=10)
        ax.set_title('Real-time Performance Metrics', color='white', fontsize=12, fontweight='bold')
        
        # Style
        ax.grid(True, alpha=0.3, color='white')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        
        # Show current values
        if snapshots:
            latest = snapshots[-1]
            ax.text(0.02, 0.98, f'Coverage: {latest.coverage_rate*100:.1f}%\n'
                              f'Cooperation: {latest.cooperation_index*100:.1f}%\n'
                              f'Reward: {latest.total_reward:.1f}',
                   transform=ax.transAxes, color='white', fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def _draw_info_panel(self, ax, snapshot, algorithm):
        """Draw information panel with key stats"""
        ax.axis('off')
        
        # Create info text
        info_text = f"""
ALGORITHM: {algorithm.upper()}
Episode: {snapshot.episode}
Step: {snapshot.step}
Time: {snapshot.timestamp:.1f}s

COVERAGE ANALYSIS:
• POIs Covered: {sum(poi.is_covered for poi in snapshot.pois)}/{len(snapshot.pois)}
• Coverage Rate: {snapshot.coverage_rate*100:.1f}%
• Avg Priority: {np.mean([poi.priority for poi in snapshot.pois]):.2f}

AGENT STATUS:
• Satellites: {sum(1 for a in snapshot.agents if a.agent_type == 'satellite')}
• UAVs: {sum(1 for a in snapshot.agents if a.agent_type == 'uav')}  
• Ground Stations: {sum(1 for a in snapshot.agents if a.agent_type == 'ground_station')}

PERFORMANCE:
• Total Reward: {snapshot.total_reward:.1f}
• Energy Used: {snapshot.energy_consumption:.1f}
• Comm. Efficiency: {snapshot.communication_efficiency*100:.1f}%
• Cooperation Index: {snapshot.cooperation_index*100:.1f}%

ENVIRONMENT:
• Area Size: {snapshot.area_size}m × {snapshot.area_size}m
• Obstacles: {len(snapshot.obstacles)}
• Active Links: {sum(len(a.communication_links) for a in snapshot.agents)}
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, color='white',
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white'))
    
    def create_training_overview_3d(self, algorithm: str, max_episodes: int = 5) -> Optional[str]:
        """Create 3D training overview video"""
        
        if not self.snapshots:
            return None
        
        episodes = sorted(list(set(s.episode for s in self.snapshots)))
        selected_episodes = episodes[:max_episodes] if len(episodes) > max_episodes else episodes
        
        print(f"🎬 Creating 3D training overview for episodes: {selected_episodes}")
        
        # Select representative snapshots from each episode
        overview_snapshots = []
        for ep in selected_episodes:
            ep_snapshots = [s for s in self.snapshots if s.episode == ep]
            if ep_snapshots:
                # Take snapshots from beginning, middle, and end of episode
                indices = [0, len(ep_snapshots)//2, -1] if len(ep_snapshots) > 2 else [0, -1]
                for idx in indices:
                    if idx < len(ep_snapshots):
                        overview_snapshots.append(ep_snapshots[idx])
        
        if not overview_snapshots:
            return None
        
        # Create overview video using the same 3D system
        temp_snapshots = self.snapshots
        self.snapshots = overview_snapshots
        
        try:
            # Use episode 0 as placeholder for overview
            for i, snapshot in enumerate(overview_snapshots):
                snapshot.episode = 0
                snapshot.step = i
            
            video_path = self.create_3d_episode_video(0, f"{algorithm}_overview", duration=8.0)
            
            if video_path:
                new_path = video_path.replace("episode_000_3d.gif", "training_overview_3d.gif")
                os.rename(video_path, new_path)
                print(f"🎬 Created 3D training overview: {new_path}")
                return new_path
        
        finally:
            self.snapshots = temp_snapshots
        
        return None
    
    def save_videos_to_output_manager(self, algorithm: str):
        """Save all generated videos to output manager"""
        if not self.output_manager:
            return
        
        episodes = sorted(list(set(s.episode for s in self.snapshots)))
        
        # Generate videos for key episodes
        key_episodes = []
        if len(episodes) >= 5:
            # Select first, quartiles, and last episodes
            key_episodes = [episodes[0], episodes[len(episodes)//4], 
                           episodes[len(episodes)//2], episodes[3*len(episodes)//4], 
                           episodes[-1]]
        else:
            key_episodes = episodes
        
        print(f"🎬 Generating 3D videos for episodes: {key_episodes}")
        
        for episode in key_episodes:
            video_path = self.create_3d_episode_video(episode, algorithm)
            if video_path and os.path.exists(video_path):
                self.output_manager.save_episode_video(episode, video_path, "gif")
        
        # Create training overview video  
        overview_path = self.create_training_overview_3d(algorithm)
        if overview_path and os.path.exists(overview_path):
            overview_target = os.path.join(
                str(self.output_manager.experiment_dir / "videos"), 
                "training_overview_3d.gif"
            )
            import shutil
            shutil.copy2(overview_path, overview_target)
            print(f"🎬 Saved 3D training overview video")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()


def create_advanced_visualization_system(output_manager=None, fps: int = 30, dpi: int = 150) -> AdvancedVisualizationSystem:
    """Factory function to create advanced visualization system"""
    return AdvancedVisualizationSystem(output_manager, fps, dpi)