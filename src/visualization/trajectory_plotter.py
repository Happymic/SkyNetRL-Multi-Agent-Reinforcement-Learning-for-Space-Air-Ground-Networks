"""
Agent Trajectory and Network Performance Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
from pathlib import Path
import pandas as pd

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrajectoryVisualizer:
    """Visualizes agent trajectories and network performance"""
    
    def __init__(self, area_size: float = 1000, save_dir: str = "results/visualizations"):
        self.area_size = area_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.agent_colors = {
            'satellite': '#FF6B6B',    # Red
            'uav': '#4ECDC4',          # Teal  
            'ground_station': '#45B7D1', # Blue
            'agent': '#9B59B6',        # Purple (generic agent)
            'default': '#95A5A6'       # Gray (fallback)
        }
        
        self.agent_markers = {
            'satellite': '^',
            'uav': 'o', 
            'ground_station': 's',
            'agent': 'D',
            'default': '.'
        }
        
        # Performance tracking
        self.performance_history = []
        self.trajectory_data = []
    
    def record_step(self, agents: Dict, pois: List, episode_metrics: Dict, step: int):
        """Record step data for visualization"""
        step_data = {
            'step': step,
            'agents': {},
            'pois': [],
            'metrics': episode_metrics.copy()
        }
        
        # Record agent positions and states
        for agent_id, agent_info in agents.items():
            step_data['agents'][agent_id] = {
                'position': agent_info['position'].copy(),
                'type': agent_info['type'],
                'energy': agent_info.get('energy', None),
                'active': agent_info.get('active', True)
            }
        
        # Record POI states
        for poi in pois:
            step_data['pois'].append({
                'position': [poi.x, poi.y],
                'covered': poi.covered,
                'priority': poi.priority
            })
        
        self.trajectory_data.append(step_data)
    
    def plot_trajectories(self, algorithm_name: str = "algorithm", episode: int = 0) -> str:
        """Plot agent trajectories over time"""
        if not self.trajectory_data:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up the plot
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Agent Trajectories - {algorithm_name} (Episode {episode})', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        
        # Plot POIs
        if self.trajectory_data:
            pois = self.trajectory_data[0]['pois']
            for poi in pois:
                x, y = poi['position']
                priority = poi['priority']
                size = priority * 20 + 50
                alpha = 0.8 if poi['covered'] else 0.4
                color = 'green' if poi['covered'] else 'orange'
                ax.scatter(x, y, s=size, c=color, alpha=alpha, marker='*', 
                          edgecolors='black', linewidth=1, label='POI' if poi == pois[0] else "")
        
        # Plot agent trajectories
        agent_trajectories = {}
        for step_data in self.trajectory_data:
            for agent_id, agent_info in step_data['agents'].items():
                if agent_id not in agent_trajectories:
                    agent_trajectories[agent_id] = {
                        'x': [], 'y': [], 'type': agent_info['type']
                    }
                agent_trajectories[agent_id]['x'].append(agent_info['position'][0])
                agent_trajectories[agent_id]['y'].append(agent_info['position'][1])
        
        # Plot each agent's trajectory
        for agent_id, traj in agent_trajectories.items():
            agent_type = traj['type']
            color = self.agent_colors.get(agent_type, self.agent_colors['default'])
            marker = self.agent_markers.get(agent_type, self.agent_markers['default'])
            
            # Plot trajectory line
            ax.plot(traj['x'], traj['y'], color=color, alpha=0.6, linewidth=2, linestyle='-')
            
            # Plot start position
            ax.scatter(traj['x'][0], traj['y'][0], c=color, s=100, marker=marker, 
                      edgecolors='white', linewidth=2, alpha=0.8)
            
            # Plot end position
            ax.scatter(traj['x'][-1], traj['y'][-1], c=color, s=150, marker=marker,
                      edgecolors='black', linewidth=2, alpha=1.0,
                      label=f'{agent_type.title()} {agent_id}' if agent_id == list(agent_trajectories.keys())[0] else "")
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Save plot
        filename = f"trajectories_{algorithm_name}_ep{episode}.png"
        filepath = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_performance_metrics(self, algorithm_name: str = "algorithm") -> str:
        """Plot performance metrics over time"""
        if not self.trajectory_data:
            return None
            
        # Extract metrics over time
        steps = []
        coverage_rates = []
        energy_levels = []
        active_agents = []
        
        for step_data in self.trajectory_data:
            steps.append(step_data['step'])
            
            # Coverage rate
            pois = step_data['pois']
            covered = sum(1 for poi in pois if poi['covered'])
            coverage_rates.append(covered / len(pois) if pois else 0)
            
            # Average energy level
            agents = step_data['agents']
            energy_sum = 0
            energy_count = 0
            active_count = 0
            
            for agent_info in agents.values():
                if agent_info['active']:
                    active_count += 1
                    if agent_info['energy'] is not None:
                        energy_sum += agent_info['energy']
                        energy_count += 1
            
            energy_levels.append(energy_sum / energy_count if energy_count > 0 else 0)
            active_agents.append(active_count)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Metrics - {algorithm_name}', fontsize=16, fontweight='bold')
        
        # Coverage rate over time
        ax1.plot(steps, coverage_rates, color='#2E8B57', linewidth=2, marker='o', markersize=4)
        ax1.set_title('POI Coverage Rate', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Coverage Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Energy levels over time
        ax2.plot(steps, energy_levels, color='#FF6B6B', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Average Energy Level', fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Energy Level')
        ax2.grid(True, alpha=0.3)
        
        # Active agents over time
        ax3.plot(steps, active_agents, color='#4ECDC4', linewidth=2, marker='^', markersize=4)
        ax3.set_title('Active Agents', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Number of Active Agents')
        ax3.grid(True, alpha=0.3)
        
        # Coverage efficiency (POIs covered per step)
        cumulative_coverage = np.cumsum(coverage_rates)
        efficiency = cumulative_coverage / (np.array(steps) + 1)
        ax4.plot(steps, efficiency, color='#9B59B6', linewidth=2, marker='d', markersize=4)
        ax4.set_title('Cumulative Coverage Efficiency', fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Avg Coverage per Step')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"performance_{algorithm_name}.png"
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_animated_trajectory(self, algorithm_name: str = "algorithm", episode: int = 0) -> str:
        """Create animated visualization of agent movements"""
        if not self.trajectory_data:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Initialize empty plots
        agent_plots = {}
        trail_plots = {}
        
        def init():
            ax.clear()
            ax.set_xlim(0, self.area_size)
            ax.set_ylim(0, self.area_size)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Agent Movement Animation - {algorithm_name} (Episode {episode})', 
                        fontsize=16, fontweight='bold')
            return []
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(0, self.area_size)
            ax.set_ylim(0, self.area_size)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Agent Movement Animation - {algorithm_name} (Episode {episode}) - Step {frame}', 
                        fontsize=16, fontweight='bold')
            
            if frame >= len(self.trajectory_data):
                return []
            
            step_data = self.trajectory_data[frame]
            
            # Plot POIs
            for poi in step_data['pois']:
                x, y = poi['position']
                priority = poi['priority']
                size = priority * 20 + 50
                alpha = 0.8 if poi['covered'] else 0.4
                color = 'green' if poi['covered'] else 'orange'
                ax.scatter(x, y, s=size, c=color, alpha=alpha, marker='*', 
                          edgecolors='black', linewidth=1)
            
            # Plot agents and their trails
            for agent_id, agent_info in step_data['agents'].items():
                if not agent_info['active']:
                    continue
                    
                agent_type = agent_info['type']
                color = self.agent_colors[agent_type]
                marker = self.agent_markers[agent_type]
                
                # Plot trail (last 10 positions)
                trail_x = []
                trail_y = []
                start_frame = max(0, frame - 10)
                for i in range(start_frame, frame + 1):
                    if i < len(self.trajectory_data) and agent_id in self.trajectory_data[i]['agents']:
                        pos = self.trajectory_data[i]['agents'][agent_id]['position']
                        trail_x.append(pos[0])
                        trail_y.append(pos[1])
                
                if len(trail_x) > 1:
                    ax.plot(trail_x, trail_y, color=color, alpha=0.5, linewidth=2)
                
                # Plot current position
                pos = agent_info['position']
                ax.scatter(pos[0], pos[1], c=color, s=150, marker=marker,
                          edgecolors='black', linewidth=2, alpha=1.0)
                
                # Add agent label
                ax.annotate(f'{agent_type[0].upper()}{agent_id}', 
                           (pos[0], pos[1]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, fontweight='bold')
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(self.trajectory_data), 
                                     interval=200, blit=False, repeat=True)
        
        # Save animation
        filename = f"animation_{algorithm_name}_ep{episode}.gif"
        filepath = self.save_dir / filename
        anim.save(str(filepath), writer='pillow', fps=5)
        plt.close()
        
        return str(filepath)
    
    def plot_network_topology(self, algorithm_name: str = "algorithm", step: int = -1) -> str:
        """Plot network connectivity topology"""
        if not self.trajectory_data:
            return None
            
        step_data = self.trajectory_data[step]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Network Topology - {algorithm_name} (Step {step_data["step"]})', 
                    fontsize=16, fontweight='bold')
        
        # Communication range (simplified)
        comm_range = 200  # Default communication range
        
        # Plot agents
        agent_positions = {}
        for agent_id, agent_info in step_data['agents'].items():
            if not agent_info['active']:
                continue
                
            pos = agent_info['position']
            agent_type = agent_info['type']
            color = self.agent_colors.get(agent_type, self.agent_colors['default'])
            marker = self.agent_markers.get(agent_type, self.agent_markers['default'])
            
            agent_positions[agent_id] = pos
            
            # Plot agent
            ax.scatter(pos[0], pos[1], c=color, s=200, marker=marker,
                      edgecolors='black', linewidth=2, alpha=0.8)
            
            # Plot communication range
            circle = Circle(pos, comm_range, fill=False, color=color, alpha=0.3, linestyle='--')
            ax.add_patch(circle)
            
            # Add label
            ax.annotate(f'{agent_type[0].upper()}{agent_id}', 
                       (pos[0], pos[1]), xytext=(10, 10), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Draw communication links
        agents_list = list(agent_positions.items())
        for i, (agent1_id, pos1) in enumerate(agents_list):
            for j, (agent2_id, pos2) in enumerate(agents_list):
                if i < j:  # Avoid duplicate links
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    if distance <= comm_range:
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               color='gray', alpha=0.6, linewidth=1, linestyle='-')
        
        # Plot POIs
        for poi in step_data['pois']:
            x, y = poi['position']
            priority = poi['priority']
            size = priority * 20 + 50
            alpha = 0.8 if poi['covered'] else 0.4
            color = 'green' if poi['covered'] else 'orange'
            ax.scatter(x, y, s=size, c=color, alpha=alpha, marker='*', 
                      edgecolors='black', linewidth=1)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"network_topology_{algorithm_name}_step{step_data['step']}.png"
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def reset(self):
        """Reset visualization data"""
        self.trajectory_data = []
        self.performance_history = []
    
    def generate_summary_report(self, algorithm_name: str = "algorithm") -> str:
        """Generate summary visualization report"""
        if not self.trajectory_data:
            return None
            
        # Generate all visualizations
        traj_file = self.plot_trajectories(algorithm_name)
        perf_file = self.plot_performance_metrics(algorithm_name)
        topo_file = self.plot_network_topology(algorithm_name)
        anim_file = self.create_animated_trajectory(algorithm_name)
        
        # Create summary report
        report = f"""# Visualization Report - {algorithm_name}

## Generated Files:
- Trajectory Plot: {traj_file}
- Performance Metrics: {perf_file}  
- Network Topology: {topo_file}
- Animation: {anim_file}

## Summary Statistics:
- Total Steps: {len(self.trajectory_data)}
- Final Coverage: {self.trajectory_data[-1]['pois'] if self.trajectory_data else 'N/A'}
- Active Agents: {sum(1 for a in self.trajectory_data[-1]['agents'].values() if a['active']) if self.trajectory_data else 'N/A'}
"""
        
        report_file = self.save_dir / f"visualization_report_{algorithm_name}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return str(report_file)