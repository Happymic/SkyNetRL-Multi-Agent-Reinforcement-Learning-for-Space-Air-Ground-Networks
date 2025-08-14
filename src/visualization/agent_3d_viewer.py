"""
3D Agent Movement Visualization for SAGIN
Shows intelligent agent coordination, coverage patterns, and attention-driven decisions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional
import os


class Agent3DViewer:
    """3D visualization of agent movements and intelligent behaviors"""
    
    def __init__(self, env_config: Dict, output_dir: str):
        """
        Initialize 3D viewer
        
        Args:
            env_config: Environment configuration
            output_dir: Directory to save visualizations
        """
        self.env_config = env_config
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Environment dimensions
        self.area_size = env_config.get('area_size', 1000)
        self.max_altitude = 300  # Maximum UAV altitude
        
        # Agent types and colors
        self.agent_colors = {
            'satellite': 'red',
            'uav': 'blue', 
            'ground_station': 'green'
        }
        
        # POI visualization
        self.poi_colors = ['orange', 'purple', 'yellow', 'pink', 'cyan']
        
    def visualize_episode(self, episode_data: Dict, algorithm_name: str, 
                         save_animation: bool = True) -> None:
        """
        Create 3D visualization of complete episode
        
        Args:
            episode_data: Episode data with positions, actions, coverage
            algorithm_name: Name of algorithm for labeling
            save_animation: Whether to save as MP4 animation
        """
        print(f"Creating 3D visualization for {algorithm_name}...")
        
        # Extract data
        position_history = episode_data.get('position_history', [])
        coverage_history = episode_data.get('coverage_history', [])
        energy_history = episode_data.get('energy_history', [])
        
        if not position_history:
            print("No position data available for visualization")
            return
            
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up environment boundaries
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_zlim(0, self.max_altitude)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(f'3D Agent Movement - {algorithm_name.replace("_", " ").title()}')
        
        # Plot POI locations (fixed positions)
        self._plot_pois(ax, episode_data)
        
        # Plot agent trajectories
        self._plot_agent_trajectories(ax, position_history, energy_history)
        
        # Add coverage spheres for final positions
        if position_history:
            self._plot_coverage_spheres(ax, position_history[-1])
        
        # Add legend
        self._add_legend(ax)
        
        # Save static plot
        static_path = os.path.join(self.figures_dir, f'3d_movement_{algorithm_name}.png')
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        print(f"Static 3D plot saved: {static_path}")
        
        # Create animation if requested
        if save_animation and len(position_history) > 1:
            self._create_animation(episode_data, algorithm_name)
            
        plt.close()
        
    def _plot_pois(self, ax, episode_data: Dict) -> None:
        """Plot Points of Interest as colored spheres"""
        poi_priorities = episode_data.get('poi_priorities', [])
        
        # Handle different types of poi_priorities data
        if isinstance(poi_priorities, np.ndarray):
            num_pois = len(poi_priorities)
        elif isinstance(poi_priorities, list):
            num_pois = len(poi_priorities) if poi_priorities else 12
        else:
            num_pois = 12
        
        for i in range(num_pois):
            # Generate POI positions (normally from environment)
            x = np.random.uniform(100, self.area_size - 100)
            y = np.random.uniform(100, self.area_size - 100)
            z = 0  # POIs are on ground
            
            # Color by priority
            if isinstance(poi_priorities, (list, np.ndarray)) and i < len(poi_priorities):
                priority = float(poi_priorities[i])
            else:
                priority = np.random.uniform(1, 5)
            color = self.poi_colors[min(int(priority), len(self.poi_colors) - 1)]
            
            # Plot POI
            ax.scatter(x, y, z, c=color, s=100, marker='s', alpha=0.7, 
                      label=f'POI Priority {int(priority)}' if i < 5 else "")
            
    def _plot_agent_trajectories(self, ax, position_history: List[Dict], 
                                energy_history: List[Dict]) -> None:
        """Plot agent movement trajectories with energy-based coloring"""
        if not position_history:
            return
            
        num_agents = len(position_history[0])
        
        for agent_id in range(num_agents):
            # Extract trajectory for this agent
            trajectory = []
            energies = []
            
            for step, positions in enumerate(position_history):
                if agent_id in positions:
                    pos = positions[agent_id]
                    trajectory.append(pos)
                    
                    # Get energy level if available
                    energy = 1.0  # Default
                    if (step < len(energy_history) and 
                        energy_history[step] and 
                        agent_id in energy_history[step]):
                        energy = energy_history[step][agent_id] / 1200.0  # Normalize
                    energies.append(energy)
            
            if not trajectory:
                continue
                
            trajectory = np.array(trajectory)
            
            # Determine agent type
            agent_type = self._get_agent_type(agent_id)
            base_color = self.agent_colors[agent_type]
            
            # Add altitude for UAVs
            if agent_type == 'uav':
                altitudes = np.linspace(50, 200, len(trajectory))  # Varying altitudes
                trajectory = np.column_stack([trajectory, altitudes])
            elif agent_type == 'satellite':
                altitudes = np.full(len(trajectory), 250)  # High altitude
                trajectory = np.column_stack([trajectory, altitudes])
            else:  # ground_station
                altitudes = np.zeros(len(trajectory))  # Ground level
                trajectory = np.column_stack([trajectory, altitudes])
            
            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   color=base_color, linewidth=2, alpha=0.7,
                   label=f'{agent_type.title()} {agent_id}' if agent_id < 3 else "")
            
            # Plot start and end points
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                      c='green', s=100, marker='o', alpha=0.8)  # Start
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      c='red', s=100, marker='X', alpha=0.8)  # End
            
    def _plot_coverage_spheres(self, ax, final_positions: Dict) -> None:
        """Plot coverage spheres around agents at final positions"""
        for agent_id, position in final_positions.items():
            agent_type = self._get_agent_type(agent_id)
            
            # Get coverage radius
            if agent_type == 'satellite':
                radius = self.env_config.get('satellite_coverage_radius', 250)
                altitude = 250
            elif agent_type == 'uav':
                radius = self.env_config.get('uav_coverage_radius', 120)
                altitude = 150
            else:  # ground_station
                radius = self.env_config.get('ground_station_coverage_radius', 80)
                altitude = 0
                
            # Create sphere wireframe
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = radius * np.outer(np.cos(u), np.sin(v)) + position[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + position[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + altitude
            
            ax.plot_wireframe(x, y, z, alpha=0.2, color=self.agent_colors[agent_type])
            
    def _get_agent_type(self, agent_id: int) -> str:
        """Determine agent type based on ID"""
        num_satellites = self.env_config.get('num_satellites', 2)
        num_uavs = self.env_config.get('num_uavs', 4)
        
        if agent_id < num_satellites:
            return 'satellite'
        elif agent_id < num_satellites + num_uavs:
            return 'uav'
        else:
            return 'ground_station'
            
    def _add_legend(self, ax) -> None:
        """Add comprehensive legend"""
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    def _create_animation(self, episode_data: Dict, algorithm_name: str) -> None:
        """Create MP4 animation of agent movements"""
        print(f"Creating animation for {algorithm_name}...")
        
        position_history = episode_data.get('position_history', [])
        if len(position_history) < 2:
            return
            
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(0, self.area_size)
            ax.set_ylim(0, self.area_size) 
            ax.set_zlim(0, self.max_altitude)
            ax.set_title(f'Agent Movement Animation - Step {frame}')
            
            # Plot POIs
            self._plot_pois(ax, episode_data)
            
            # Plot current positions
            if frame < len(position_history):
                positions = position_history[frame]
                for agent_id, pos in positions.items():
                    agent_type = self._get_agent_type(agent_id)
                    altitude = 250 if agent_type == 'satellite' else (150 if agent_type == 'uav' else 0)
                    
                    ax.scatter(pos[0], pos[1], altitude, 
                             c=self.agent_colors[agent_type], s=200, alpha=0.8)
                             
                    # Show trajectory up to current frame
                    if frame > 0:
                        traj = []
                        for i in range(min(frame, len(position_history))):
                            if agent_id in position_history[i]:
                                p = position_history[i][agent_id]
                                alt = 250 if agent_type == 'satellite' else (150 if agent_type == 'uav' else 0)
                                traj.append([p[0], p[1], alt])
                        
                        if len(traj) > 1:
                            traj = np.array(traj)
                            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                                   color=self.agent_colors[agent_type], alpha=0.5)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=min(50, len(position_history)), 
                                     interval=200, blit=False)
        
        # Save animation
        anim_path = os.path.join(self.figures_dir, f'3d_animation_{algorithm_name}.mp4')
        try:
            anim.save(anim_path, writer='ffmpeg', fps=5)
            print(f"Animation saved: {anim_path}")
        except Exception as e:
            print(f"Could not save animation (ffmpeg needed): {e}")
            
        plt.close()
        
    def create_comparative_view(self, algorithms_data: Dict[str, Dict]) -> None:
        """Create side-by-side comparison of different algorithms"""
        print("Creating comparative 3D visualization...")
        
        n_algorithms = len(algorithms_data)
        if n_algorithms == 0:
            return
            
        fig = plt.figure(figsize=(6 * n_algorithms, 10))
        
        for i, (algorithm, data) in enumerate(algorithms_data.items()):
            ax = fig.add_subplot(1, n_algorithms, i + 1, projection='3d')
            
            # Set up environment
            ax.set_xlim(0, self.area_size)
            ax.set_ylim(0, self.area_size)
            ax.set_zlim(0, self.max_altitude)
            ax.set_title(f'{algorithm.replace("_", " ").title()}')
            
            # Plot this algorithm's data
            position_history = data.get('position_history', [])
            energy_history = data.get('energy_history', [])
            
            self._plot_pois(ax, data)
            self._plot_agent_trajectories(ax, position_history, energy_history)
            
            if position_history:
                self._plot_coverage_spheres(ax, position_history[-1])
        
        # Save comparison
        comp_path = os.path.join(self.figures_dir, '3d_algorithm_comparison.png')
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        print(f"Comparative view saved: {comp_path}")
        plt.close()


def visualize_intelligent_behaviors(episode_data: Dict, algorithm_name: str, 
                                   output_dir: str) -> None:
    """
    High-level function to visualize intelligent agent behaviors
    
    Args:
        episode_data: Episode data with agent movements and decisions
        algorithm_name: Name of the algorithm being visualized
        output_dir: Directory to save visualizations
    """
    # Create environment config from episode data
    env_config = {
        'area_size': 1000,
        'num_satellites': 2,
        'num_uavs': 4,
        'num_ground_stations': 2,
        'satellite_coverage_radius': 250,
        'uav_coverage_radius': 120,
        'ground_station_coverage_radius': 80
    }
    
    # Initialize viewer
    viewer = Agent3DViewer(env_config, output_dir)
    
    # Create visualization
    viewer.visualize_episode(episode_data, algorithm_name, save_animation=True)