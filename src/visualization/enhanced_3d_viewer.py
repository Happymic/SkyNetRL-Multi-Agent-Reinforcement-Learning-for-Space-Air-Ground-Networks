"""
Enhanced 3D Agent Visualization with Clear Frame-by-Frame Analysis
Shows intelligent agent behavior with detailed, easy-to-understand visuals
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import os


class Enhanced3DViewer:
    """Enhanced 3D visualization with clear, detailed agent movement analysis"""
    
    def __init__(self, env_config: Dict, output_dir: str):
        """
        Initialize enhanced 3D viewer
        
        Args:
            env_config: Environment configuration
            output_dir: Directory to save visualizations
        """
        self.env_config = env_config
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Environment setup
        self.area_size = env_config.get('area_size', 1000)
        self.max_altitude = 300
        
        # Create consistent POI positions (not random)
        self.poi_positions = self._create_consistent_pois()
        
        # Enhanced color scheme
        self.agent_colors = {
            'satellite': '#FF4444',    # Bright red
            'uav': '#4488FF',          # Bright blue
            'ground_station': '#44AA44' # Bright green
        }
        
        # POI priority colors (clear and distinct)
        self.priority_colors = {
            1: '#FFA500',  # Orange - low priority
            2: '#FFD700',  # Gold
            3: '#FF69B4',  # Hot pink
            4: '#8A2BE2',  # Blue violet
            5: '#FF0000'   # Red - highest priority
        }
        
        # Set matplotlib parameters for better quality
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        
    def _create_consistent_pois(self) -> List[Tuple[float, float, int]]:
        """Create consistent POI positions that make logical sense"""
        num_pois = self.env_config.get('num_pois', 12)
        pois = []
        
        # Create POIs in a logical pattern (not random)
        # High priority POIs in center, lower priority on edges
        for i in range(num_pois):
            if i < 3:  # High priority center POIs
                angle = i * 2 * np.pi / 3
                x = self.area_size/2 + 100 * np.cos(angle)
                y = self.area_size/2 + 100 * np.sin(angle)
                priority = 5
            elif i < 6:  # Medium priority mid-ring
                angle = (i-3) * 2 * np.pi / 3
                x = self.area_size/2 + 250 * np.cos(angle)
                y = self.area_size/2 + 250 * np.sin(angle)
                priority = 3
            else:  # Lower priority outer ring
                angle = (i-6) * 2 * np.pi / 6
                x = self.area_size/2 + 400 * np.cos(angle)
                y = self.area_size/2 + 400 * np.sin(angle)
                priority = 1 + (i % 2)
            
            # Ensure within bounds
            x = max(50, min(x, self.area_size - 50))
            y = max(50, min(y, self.area_size - 50))
            
            pois.append((x, y, priority))
            
        return pois
    
    def create_detailed_movement_analysis(self, episode_data: Dict, algorithm_name: str):
        """Create detailed frame-by-frame movement analysis"""
        print(f"📊 Creating detailed movement analysis for {algorithm_name}...")
        
        position_history = episode_data.get('position_history', [])
        coverage_history = episode_data.get('coverage_history', [])
        
        if not position_history:
            print("⚠️  No position data available")
            return
            
        # Create overview plot
        self._create_clean_overview_plot(position_history, algorithm_name)
        
        # Create frame-by-frame progression
        self._create_frame_sequence(position_history, coverage_history, algorithm_name)
        
        # Create coverage analysis
        self._create_coverage_analysis(position_history, coverage_history, algorithm_name)
        
    def _create_clean_overview_plot(self, position_history: List[Dict], algorithm_name: str):
        """Create a clean, readable overview plot"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create 2D top-down view (clearer than 3D)
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_xlim(0, self.area_size)
        ax1.set_ylim(0, self.area_size)
        ax1.set_aspect('equal')
        ax1.set_title(f'{algorithm_name.replace("_", " ").title()} - Top View', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True, alpha=0.3)
        
        # Plot POIs first
        for i, (x, y, priority) in enumerate(self.poi_positions):
            color = self.priority_colors.get(priority, '#FFA500')
            ax1.scatter(x, y, c=color, s=200, marker='s', alpha=0.8, 
                       edgecolors='black', linewidth=2, zorder=10)
            ax1.annotate(f'P{i+1}\n({priority})', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Plot agent trajectories
        if position_history:
            self._plot_clean_trajectories(ax1, position_history)
            
        # Add POI legend
        legend_elements = [plt.scatter([], [], c=color, s=100, marker='s', 
                          label=f'Priority {p}') 
                          for p, color in self.priority_colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Create 3D view (cleaner version)
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        ax2.set_xlim(0, self.area_size)
        ax2.set_ylim(0, self.area_size)
        ax2.set_zlim(0, self.max_altitude)
        ax2.set_title('3D Movement Patterns', fontsize=14, fontweight='bold')
        ax2.view_init(elev=20, azim=45)  # Better viewing angle
        
        self._plot_3d_clean(ax2, position_history)
        
        # Agent statistics
        ax3 = plt.subplot(2, 2, 3)
        self._plot_agent_statistics(ax3, position_history, algorithm_name)
        
        # Coverage progression
        ax4 = plt.subplot(2, 2, 4)
        self._plot_coverage_progression(ax4, position_history)
        
        plt.tight_layout()
        overview_path = os.path.join(self.figures_dir, f'detailed_overview_{algorithm_name}.png')
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        print(f"✅ Overview plot saved: {overview_path}")
        plt.close()
        
    def _plot_clean_trajectories(self, ax, position_history: List[Dict]):
        """Plot clean, readable agent trajectories"""
        if not position_history:
            return
            
        num_agents = len(position_history[0])
        
        for agent_id in range(num_agents):
            trajectory = []
            for positions in position_history:
                if agent_id in positions:
                    trajectory.append(positions[agent_id])
                    
            if not trajectory:
                continue
                
            trajectory = np.array(trajectory)
            agent_type = self._get_agent_type(agent_id)
            color = self.agent_colors[agent_type]
            
            # Plot trajectory with varying alpha (fade from start to end)
            for i in range(len(trajectory) - 1):
                alpha = 0.3 + 0.7 * i / len(trajectory)
                ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], 
                       color=color, alpha=alpha, linewidth=2)
            
            # Mark start and end clearly
            ax.scatter(trajectory[0, 0], trajectory[0, 1], c='lime', s=150, 
                      marker='o', edgecolors='black', linewidth=2, zorder=20,
                      label=f'{agent_type} start' if agent_id == self._get_first_of_type(agent_type) else "")
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150,
                      marker='X', edgecolors='black', linewidth=2, zorder=20,
                      label=f'{agent_type} end' if agent_id == self._get_first_of_type(agent_type) else "")
                      
            # Add agent ID labels
            ax.annotate(f'{agent_type[0].upper()}{agent_id}', 
                       (trajectory[-1, 0], trajectory[-1, 1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    def _plot_3d_clean(self, ax, position_history: List[Dict]):
        """Plot clean 3D trajectories"""
        if not position_history:
            return
            
        # Plot POIs on ground
        for i, (x, y, priority) in enumerate(self.poi_positions):
            color = self.priority_colors.get(priority, '#FFA500')
            ax.scatter(x, y, 0, c=color, s=100, marker='s', alpha=0.8)
            
        # Plot agent trajectories at different altitudes
        num_agents = len(position_history[0])
        
        for agent_id in range(num_agents):
            trajectory = []
            for positions in position_history:
                if agent_id in positions:
                    pos = positions[agent_id]
                    agent_type = self._get_agent_type(agent_id)
                    
                    # Set realistic altitudes
                    if agent_type == 'satellite':
                        altitude = 280
                    elif agent_type == 'uav':
                        altitude = 150
                    else:  # ground_station
                        altitude = 5
                        
                    trajectory.append([pos[0], pos[1], altitude])
                    
            if trajectory:
                trajectory = np.array(trajectory)
                color = self.agent_colors[self._get_agent_type(agent_id)]
                
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                       color=color, linewidth=2, alpha=0.8)
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                          c=color, s=100, alpha=1.0)
    
    def _create_frame_sequence(self, position_history: List[Dict], 
                              coverage_history: List, algorithm_name: str):
        """Create frame-by-frame sequence showing step-by-step progression"""
        print(f"🎬 Creating frame sequence for {algorithm_name}...")
        
        if not position_history:
            return
            
        # Create key frames (not every step to avoid clutter)
        key_frames = [0, len(position_history)//4, len(position_history)//2, 
                     3*len(position_history)//4, len(position_history)-1]
        
        fig, axes = plt.subplots(1, len(key_frames), figsize=(20, 4))
        fig.suptitle(f'{algorithm_name.replace("_", " ").title()} - Movement Progression', 
                    fontsize=16, fontweight='bold')
        
        for i, frame_idx in enumerate(key_frames):
            ax = axes[i] if len(key_frames) > 1 else axes
            ax.set_xlim(0, self.area_size)
            ax.set_ylim(0, self.area_size)
            ax.set_aspect('equal')
            ax.set_title(f'Step {frame_idx + 1}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Plot POIs
            for j, (x, y, priority) in enumerate(self.poi_positions):
                color = self.priority_colors.get(priority, '#FFA500')
                covered = self._is_poi_covered(j, frame_idx, coverage_history)
                alpha = 1.0 if not covered else 0.3
                ax.scatter(x, y, c=color, s=150, marker='s', alpha=alpha,
                          edgecolors='black', linewidth=1)
                
                # Mark covered POIs
                if covered:
                    ax.scatter(x, y, c='green', s=80, marker='o', alpha=1.0)
            
            # Plot agent positions and short trails
            if frame_idx < len(position_history):
                positions = position_history[frame_idx]
                
                for agent_id, pos in positions.items():
                    agent_type = self._get_agent_type(agent_id)
                    color = self.agent_colors[agent_type]
                    
                    # Plot current position
                    ax.scatter(pos[0], pos[1], c=color, s=200, alpha=0.9,
                              edgecolors='black', linewidth=2)
                    
                    # Plot trail (last few steps)
                    trail_length = min(5, frame_idx + 1)
                    if trail_length > 1:
                        trail_positions = []
                        for t in range(max(0, frame_idx - trail_length + 1), frame_idx + 1):
                            if agent_id in position_history[t]:
                                trail_positions.append(position_history[t][agent_id])
                        
                        if len(trail_positions) > 1:
                            trail = np.array(trail_positions)
                            ax.plot(trail[:, 0], trail[:, 1], color=color, 
                                   alpha=0.5, linewidth=2, linestyle='--')
                    
                    # Add agent label
                    ax.annotate(f'{agent_type[0].upper()}{agent_id}', 
                               (pos[0], pos[1]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Add coverage info
            if coverage_history and frame_idx < len(coverage_history):
                covered_count = sum(1 for j in range(len(self.poi_positions)) 
                                   if self._is_poi_covered(j, frame_idx, coverage_history))
                coverage_rate = covered_count / len(self.poi_positions)
                ax.text(0.02, 0.98, f'Coverage: {coverage_rate:.1%}\n({covered_count}/{len(self.poi_positions)})',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        sequence_path = os.path.join(self.figures_dir, f'frame_sequence_{algorithm_name}.png')
        plt.savefig(sequence_path, dpi=300, bbox_inches='tight')
        print(f"✅ Frame sequence saved: {sequence_path}")
        plt.close()
    
    def _create_coverage_analysis(self, position_history: List[Dict], 
                                 coverage_history: List, algorithm_name: str):
        """Create detailed coverage analysis visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{algorithm_name.replace("_", " ").title()} - Coverage Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Coverage heatmap
        ax1.set_title('Agent Coverage Areas', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, self.area_size)
        ax1.set_ylim(0, self.area_size)
        ax1.set_aspect('equal')
        
        # Create coverage heatmap
        x = np.linspace(0, self.area_size, 50)
        y = np.linspace(0, self.area_size, 50)
        X, Y = np.meshgrid(x, y)
        coverage_map = np.zeros_like(X)
        
        if position_history:
            final_positions = position_history[-1]
            for agent_id, pos in final_positions.items():
                agent_type = self._get_agent_type(agent_id)
                radius = self._get_coverage_radius(agent_type)
                
                # Add coverage to heatmap
                dist_map = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
                coverage_map += np.where(dist_map <= radius, 1, 0)
        
        im = ax1.imshow(coverage_map, extent=[0, self.area_size, 0, self.area_size],
                       origin='lower', cmap='Reds', alpha=0.6)
        plt.colorbar(im, ax=ax1, label='Coverage Intensity')
        
        # Plot POIs and final agent positions
        for i, (x, y, priority) in enumerate(self.poi_positions):
            color = self.priority_colors.get(priority, '#FFA500')
            ax1.scatter(x, y, c=color, s=200, marker='s', edgecolors='black', linewidth=2)
            ax1.annotate(f'P{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        if position_history:
            final_positions = position_history[-1]
            for agent_id, pos in final_positions.items():
                agent_type = self._get_agent_type(agent_id)
                color = self.agent_colors[agent_type]
                ax1.scatter(pos[0], pos[1], c=color, s=300, alpha=0.8,
                           edgecolors='black', linewidth=2)
                ax1.annotate(f'{agent_type[0].upper()}{agent_id}', 
                            (pos[0], pos[1]), xytext=(10, 10), 
                            textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Coverage over time
        ax2.set_title('Coverage Progress Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Coverage Rate')
        ax2.grid(True, alpha=0.3)
        
        if coverage_history:
            steps = range(len(coverage_history))
            coverage_rates = []
            for step in steps:
                covered = sum(1 for j in range(len(self.poi_positions)) 
                             if self._is_poi_covered(j, step, coverage_history))
                coverage_rates.append(covered / len(self.poi_positions))
            
            ax2.plot(steps, coverage_rates, 'b-', linewidth=3, marker='o', markersize=6,
                    label='Coverage Rate')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='100% Coverage')
            ax2.set_ylim(0, 1.1)
            ax2.legend()
            
            # Add final coverage text
            final_coverage = coverage_rates[-1] if coverage_rates else 0
            ax2.text(0.7, 0.3, f'Final Coverage:\n{final_coverage:.1%}', 
                    transform=ax2.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        coverage_path = os.path.join(self.figures_dir, f'coverage_analysis_{algorithm_name}.png')
        plt.savefig(coverage_path, dpi=300, bbox_inches='tight')
        print(f"✅ Coverage analysis saved: {coverage_path}")
        plt.close()
    
    def _plot_agent_statistics(self, ax, position_history: List[Dict], algorithm_name: str):
        """Plot agent movement statistics"""
        ax.set_title('Agent Movement Statistics', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        if not position_history:
            ax.text(0.5, 0.5, 'No position data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Calculate statistics
        stats_text = f"Algorithm: {algorithm_name.replace('_', ' ').title()}\n\n"
        stats_text += f"Total Steps: {len(position_history)}\n"
        stats_text += f"Total Agents: {len(position_history[0]) if position_history else 0}\n\n"
        
        # Agent type breakdown
        if position_history:
            agent_types = {}
            for agent_id in position_history[0].keys():
                agent_type = self._get_agent_type(agent_id)
                agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            
            for agent_type, count in agent_types.items():
                stats_text += f"{agent_type.title()}s: {count}\n"
        
        stats_text += f"\nEnvironment: {self.area_size}x{self.area_size}m\n"
        stats_text += f"POIs: {len(self.poi_positions)}\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    def _plot_coverage_progression(self, ax, position_history: List[Dict]):
        """Plot simple coverage progression info"""
        ax.set_title('POI Priority Distribution', fontsize=12, fontweight='bold')
        
        priorities = [priority for _, _, priority in self.poi_positions]
        priority_counts = {i: priorities.count(i) for i in range(1, 6)}
        
        bars = ax.bar(priority_counts.keys(), priority_counts.values(),
                     color=[self.priority_colors.get(i, '#888888') for i in priority_counts.keys()],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Priority Level')
        ax.set_ylabel('Number of POIs')
        ax.set_xticks(list(priority_counts.keys()))
        
        # Add value labels on bars
        for bar, value in zip(bars, priority_counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(value), ha='center', va='bottom', fontweight='bold')
    
    def _get_agent_type(self, agent_id: int) -> str:
        """Get agent type from ID"""
        num_satellites = self.env_config.get('num_satellites', 2)
        num_uavs = self.env_config.get('num_uavs', 4)
        
        if agent_id < num_satellites:
            return 'satellite'
        elif agent_id < num_satellites + num_uavs:
            return 'uav'
        else:
            return 'ground_station'
    
    def _get_first_of_type(self, agent_type: str) -> int:
        """Get the first agent ID of a given type"""
        if agent_type == 'satellite':
            return 0
        elif agent_type == 'uav':
            return self.env_config.get('num_satellites', 2)
        else:
            return self.env_config.get('num_satellites', 2) + self.env_config.get('num_uavs', 4)
    
    def _get_coverage_radius(self, agent_type: str) -> float:
        """Get coverage radius for agent type"""
        if agent_type == 'satellite':
            return self.env_config.get('satellite_coverage_radius', 250)
        elif agent_type == 'uav':
            return self.env_config.get('uav_coverage_radius', 120)
        else:
            return self.env_config.get('ground_station_coverage_radius', 80)
    
    def _is_poi_covered(self, poi_index: int, step: int, coverage_history: List) -> bool:
        """Check if POI is covered at given step"""
        if not coverage_history or step >= len(coverage_history):
            return False
        
        # Simplified coverage check (in real implementation this would be more sophisticated)
        if isinstance(coverage_history[step], dict):
            return coverage_history[step].get(poi_index, 0) > 0.5
        elif isinstance(coverage_history[step], list) and poi_index < len(coverage_history[step]):
            return coverage_history[step][poi_index] > 0.5
        else:
            return False