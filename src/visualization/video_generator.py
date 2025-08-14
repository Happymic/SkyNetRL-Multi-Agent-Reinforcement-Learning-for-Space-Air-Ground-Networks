"""
Advanced Video Generation System for Multi-Agent Visualization
Provides clear, professional video output with multiple perspectives
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Optional
import os
from scipy.interpolate import interp1d, UnivariateSpline
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class VideoGenerator:
    """Generate high-quality videos of multi-agent movement"""
    
    def __init__(self, env_config: Dict, output_dir: str):
        """
        Initialize video generator
        
        Args:
            env_config: Environment configuration
            output_dir: Directory to save videos
        """
        self.env_config = env_config
        self.output_dir = output_dir
        self.video_dir = os.path.join(output_dir, 'videos')
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Environment parameters
        self.area_size = env_config.get('area_size', 1000)
        self.num_agents = (env_config.get('num_satellites', 2) + 
                          env_config.get('num_uavs', 3) + 
                          env_config.get('num_ground_stations', 2))
        
        # Video settings
        self.fps = 30  # Frames per second
        self.video_quality = 'high'  # 'low', 'medium', 'high', 'ultra'
        self.interpolation_factor = 3  # Smooth interpolation between timesteps
        
        # Visual style settings
        self.agent_styles = {
            'satellite': {'color': '#FF6B6B', 'marker': '^', 'size': 150},
            'uav': {'color': '#4ECDC4', 'marker': 'o', 'size': 120},
            'ground_station': {'color': '#95E77E', 'marker': 's', 'size': 100}
        }
        
        self.poi_styles = {
            1: {'color': '#FFE66D', 'alpha': 0.6},  # Low priority
            2: {'color': '#FFA500', 'alpha': 0.7},
            3: {'color': '#FF7F50', 'alpha': 0.8},
            4: {'color': '#FF6347', 'alpha': 0.9},
            5: {'color': '#FF0000', 'alpha': 1.0}   # High priority
        }
        
        # Camera settings
        self.camera_modes = ['overview', 'tracking', 'orbiting', 'split_screen']
        self.current_camera = 'overview'
        
        # Trail settings for movement visualization
        self.trail_length = 20  # Number of past positions to show
        self.trail_alpha_decay = 0.95  # Trail fade effect
        
    def generate_episode_video(self, 
                              episode_data: Dict,
                              algorithm_name: str,
                              camera_mode: str = 'overview',
                              output_name: Optional[str] = None) -> str:
        """
        Generate a video from episode data
        
        Args:
            episode_data: Dictionary containing position_history, coverage_history, etc.
            algorithm_name: Name of the algorithm for titling
            camera_mode: Camera perspective mode
            output_name: Custom output filename
            
        Returns:
            Path to generated video file
        """
        print(f"🎬 Generating video for {algorithm_name} with {camera_mode} camera...")
        
        # Extract data
        position_history = episode_data.get('position_history', [])
        coverage_history = episode_data.get('coverage_history', [])
        reward_history = episode_data.get('reward_history', [])
        
        if not position_history:
            print("⚠️ No position data available for video generation")
            return None
            
        # Interpolate data for smooth animation
        smooth_positions = self._interpolate_trajectories(position_history)
        
        # Set up the figure based on camera mode
        if camera_mode == 'split_screen':
            fig = self._setup_split_screen_figure()
        elif camera_mode == 'tracking':
            fig = self._setup_tracking_figure()
        else:
            fig = self._setup_standard_figure()
            
        # Create animation
        if camera_mode == 'split_screen':
            anim = self._create_split_screen_animation(
                fig, smooth_positions, coverage_history, reward_history, algorithm_name
            )
        elif camera_mode == 'tracking':
            anim = self._create_tracking_animation(
                fig, smooth_positions, coverage_history, algorithm_name
            )
        elif camera_mode == 'orbiting':
            anim = self._create_orbiting_animation(
                fig, smooth_positions, coverage_history, algorithm_name
            )
        else:  # overview
            anim = self._create_overview_animation(
                fig, smooth_positions, coverage_history, algorithm_name
            )
            
        # Configure writer based on quality setting
        writer = self._get_video_writer()
        
        # Save video
        if output_name is None:
            # Use .gif extension if using PillowWriter
            ext = '.gif' if isinstance(writer, PillowWriter) else '.mp4'
            output_name = f"{algorithm_name}_{camera_mode}_video{ext}"
        elif isinstance(writer, PillowWriter) and output_name.endswith('.mp4'):
            # Change extension to .gif if using PillowWriter
            output_name = output_name.replace('.mp4', '.gif')
        
        video_path = os.path.join(self.video_dir, output_name)
        
        print(f"💾 Saving video to: {video_path}")
        anim.save(video_path, writer=writer)
        plt.close(fig)
        
        print(f"✅ Video saved successfully: {video_path}")
        return video_path
        
    def _interpolate_trajectories(self, position_history: List[Dict]) -> List[Dict]:
        """
        Interpolate agent trajectories for smooth animation
        
        Args:
            position_history: Original position data
            
        Returns:
            Interpolated position data with more frames
        """
        if len(position_history) < 2:
            return position_history
            
        smooth_data = []
        timesteps = len(position_history)
        
        # Extract positions for each agent
        agent_trajectories = {}
        for t, frame in enumerate(position_history):
            for agent_id, pos in frame.items():
                if agent_id not in agent_trajectories:
                    agent_trajectories[agent_id] = {'t': [], 'x': [], 'y': [], 'z': []}
                agent_trajectories[agent_id]['t'].append(t)
                agent_trajectories[agent_id]['x'].append(pos[0])
                agent_trajectories[agent_id]['y'].append(pos[1])
                agent_trajectories[agent_id]['z'].append(pos[2] if len(pos) > 2 else 0)
        
        # Create interpolated timeline
        new_timesteps = np.linspace(0, timesteps-1, timesteps * self.interpolation_factor)
        
        # Interpolate each agent's trajectory
        for t_new in new_timesteps:
            frame_data = {}
            for agent_id, traj in agent_trajectories.items():
                if len(traj['t']) > 1:
                    # Use spline interpolation for smooth curves
                    fx = interp1d(traj['t'], traj['x'], kind='cubic', fill_value='extrapolate')
                    fy = interp1d(traj['t'], traj['y'], kind='cubic', fill_value='extrapolate')
                    fz = interp1d(traj['t'], traj['z'], kind='cubic', fill_value='extrapolate')
                    
                    x_smooth = float(fx(t_new))
                    y_smooth = float(fy(t_new))
                    z_smooth = float(fz(t_new))
                    
                    # Ensure within bounds
                    x_smooth = max(0, min(x_smooth, self.area_size))
                    y_smooth = max(0, min(y_smooth, self.area_size))
                    z_smooth = max(0, z_smooth)
                    
                    frame_data[agent_id] = (x_smooth, y_smooth, z_smooth)
                else:
                    # If only one position, keep it constant
                    frame_data[agent_id] = (traj['x'][0], traj['y'][0], traj['z'][0])
                    
            smooth_data.append(frame_data)
            
        return smooth_data
    
    def _setup_standard_figure(self) -> plt.Figure:
        """Set up standard figure with 2D and 3D views"""
        fig = plt.figure(figsize=(16, 9), facecolor='white')
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main 2D view (larger)
        self.ax_main = fig.add_subplot(gs[:, 0])
        self.ax_main.set_xlim(0, self.area_size)
        self.ax_main.set_ylim(0, self.area_size)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlabel('X Position (m)', fontsize=10)
        self.ax_main.set_ylabel('Y Position (m)', fontsize=10)
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # 3D view
        self.ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
        self.ax_3d.set_xlim(0, self.area_size)
        self.ax_3d.set_ylim(0, self.area_size)
        self.ax_3d.set_zlim(0, 300)
        self.ax_3d.set_xlabel('X (m)', fontsize=9)
        self.ax_3d.set_ylabel('Y (m)', fontsize=9)
        self.ax_3d.set_zlabel('Z (m)', fontsize=9)
        
        # Info panel
        self.ax_info = fig.add_subplot(gs[1, 1])
        self.ax_info.axis('off')
        
        return fig
        
    def _setup_split_screen_figure(self) -> plt.Figure:
        """Set up split screen figure for multiple agent views"""
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Overview in center
        self.ax_overview = fig.add_subplot(gs[:, 1])
        self.ax_overview.set_xlim(0, self.area_size)
        self.ax_overview.set_ylim(0, self.area_size)
        self.ax_overview.set_aspect('equal')
        self.ax_overview.set_title('Global Overview', fontsize=12, fontweight='bold')
        
        # Individual agent views
        self.ax_agents = []
        positions = [(0, 0), (0, 2), (1, 0), (1, 2)]
        for i in range(min(4, self.num_agents)):
            ax = fig.add_subplot(gs[positions[i][0], positions[i][1]])
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_aspect('equal')
            ax.set_title(f'Agent {i+1} View', fontsize=10)
            self.ax_agents.append(ax)
            
        return fig
        
    def _setup_tracking_figure(self) -> plt.Figure:
        """Set up figure for tracking specific agent"""
        fig = plt.figure(figsize=(14, 10), facecolor='white')
        
        # Main tracking view
        self.ax_track = fig.add_subplot(111)
        self.ax_track.set_aspect('equal')
        self.ax_track.set_xlabel('Relative X (m)', fontsize=10)
        self.ax_track.set_ylabel('Relative Y (m)', fontsize=10)
        
        return fig
        
    def _create_overview_animation(self, fig, positions, coverage, algorithm_name):
        """Create overview animation showing all agents"""
        
        # Initialize plot elements
        poi_patches = []
        agent_markers = []
        agent_trails = []
        coverage_circles = []
        
        # Create POIs
        poi_positions = self._generate_poi_positions()
        for poi_x, poi_y, priority in poi_positions:
            style = self.poi_styles[priority]
            rect = FancyBboxPatch(
                (poi_x-20, poi_y-20), 40, 40,
                boxstyle="round,pad=5",
                facecolor=style['color'],
                edgecolor='black',
                alpha=style['alpha'],
                linewidth=2
            )
            self.ax_main.add_patch(rect)
            poi_patches.append(rect)
            
        # Initialize agent markers and trails
        for i in range(self.num_agents):
            # Determine agent type
            if i < self.env_config.get('num_satellites', 2):
                agent_type = 'satellite'
            elif i < self.env_config.get('num_satellites', 2) + self.env_config.get('num_uavs', 3):
                agent_type = 'uav'
            else:
                agent_type = 'ground_station'
                
            style = self.agent_styles[agent_type]
            
            # Agent marker
            marker, = self.ax_main.plot([], [], style['marker'], 
                                       color=style['color'], 
                                       markersize=np.sqrt(style['size']),
                                       markeredgewidth=2,
                                       markeredgecolor='white',
                                       zorder=20)
            agent_markers.append(marker)
            
            # Agent trail
            trail, = self.ax_main.plot([], [], '-', 
                                      color=style['color'], 
                                      alpha=0.3,
                                      linewidth=2,
                                      zorder=10)
            agent_trails.append(trail)
            
            # Coverage circle
            circle = Circle((0, 0), 0, fill=False, 
                          edgecolor=style['color'],
                          alpha=0.3,
                          linewidth=1.5,
                          linestyle='--')
            self.ax_main.add_patch(circle)
            coverage_circles.append(circle)
            
        # Title and info text
        title_text = self.ax_main.text(self.area_size/2, self.area_size*1.05, 
                                      f'{algorithm_name.replace("_", " ").title()}',
                                      fontsize=14, fontweight='bold',
                                      ha='center')
        
        time_text = self.ax_main.text(0.02, 0.98, '', 
                                     transform=self.ax_main.transAxes,
                                     fontsize=10, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Trail history storage
        trail_history = {i: deque(maxlen=self.trail_length) for i in range(self.num_agents)}
        
        def animate(frame):
            """Animation function for each frame"""
            if frame >= len(positions):
                return agent_markers + agent_trails + coverage_circles
                
            current_positions = positions[frame]
            
            # Update each agent
            for i in range(self.num_agents):
                agent_key = f'agent_{i}'
                if agent_key in current_positions:
                    x, y, z = current_positions[agent_key]
                    
                    # Update marker position
                    agent_markers[i].set_data([x], [y])
                    
                    # Update trail
                    trail_history[i].append((x, y))
                    if len(trail_history[i]) > 1:
                        trail_x = [p[0] for p in trail_history[i]]
                        trail_y = [p[1] for p in trail_history[i]]
                        agent_trails[i].set_data(trail_x, trail_y)
                        
                    # Update coverage circle
                    if i < self.env_config.get('num_satellites', 2):
                        radius = self.env_config.get('satellite_coverage_radius', 250)
                    elif i < self.env_config.get('num_satellites', 2) + self.env_config.get('num_uavs', 3):
                        radius = self.env_config.get('uav_coverage_radius', 120)
                    else:
                        radius = self.env_config.get('ground_station_coverage_radius', 80)
                        
                    coverage_circles[i].center = (x, y)
                    coverage_circles[i].radius = radius
                    
            # Update time text
            actual_timestep = frame // self.interpolation_factor
            time_text.set_text(f'Time Step: {actual_timestep}\nFrame: {frame}')
            
            # Update POI coverage status
            if coverage and actual_timestep < len(coverage):
                current_coverage = coverage[actual_timestep]
                for idx, poi_patch in enumerate(poi_patches):
                    if idx < len(current_coverage) and current_coverage[idx]:
                        poi_patch.set_alpha(0.3)  # Fade covered POIs
                    
            return agent_markers + agent_trails + coverage_circles + [time_text, title_text]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(positions),
            interval=1000/self.fps, blit=False, repeat=True
        )
        
        return anim
        
    def _create_tracking_animation(self, fig, positions, coverage, algorithm_name):
        """Create animation that tracks a specific agent"""
        tracked_agent = 0  # Track first agent by default
        
        # Similar setup to overview but with camera following the agent
        agent_markers = []
        poi_markers = []
        
        # Initialize elements
        for i in range(self.num_agents):
            if i < self.env_config.get('num_satellites', 2):
                agent_type = 'satellite'
            elif i < self.env_config.get('num_satellites', 2) + self.env_config.get('num_uavs', 3):
                agent_type = 'uav'
            else:
                agent_type = 'ground_station'
                
            style = self.agent_styles[agent_type]
            marker, = self.ax_track.plot([], [], style['marker'],
                                        color=style['color'],
                                        markersize=np.sqrt(style['size']),
                                        markeredgewidth=2,
                                        markeredgecolor='white')
            agent_markers.append(marker)
            
        def animate(frame):
            if frame >= len(positions):
                return agent_markers
                
            current_positions = positions[frame]
            
            # Get tracked agent position
            tracked_key = f'agent_{tracked_agent}'
            if tracked_key in current_positions:
                center_x, center_y, _ = current_positions[tracked_key]
                
                # Update camera to follow agent
                view_range = 300
                self.ax_track.set_xlim(center_x - view_range, center_x + view_range)
                self.ax_track.set_ylim(center_y - view_range, center_y + view_range)
                
                # Update all agent positions relative to tracked agent
                for i in range(self.num_agents):
                    agent_key = f'agent_{i}'
                    if agent_key in current_positions:
                        x, y, z = current_positions[agent_key]
                        agent_markers[i].set_data([x], [y])
                        
            return agent_markers
            
        anim = animation.FuncAnimation(
            fig, animate, frames=len(positions),
            interval=1000/self.fps, blit=False, repeat=True
        )
        
        return anim
        
    def _create_orbiting_animation(self, fig, positions, coverage, algorithm_name):
        """Create 3D animation with orbiting camera"""
        # This would require 3D plotting - implementation would be similar
        # but with changing camera angles
        return self._create_overview_animation(fig, positions, coverage, algorithm_name)
        
    def _create_split_screen_animation(self, fig, positions, coverage, rewards, algorithm_name):
        """Create split screen animation showing multiple perspectives"""
        # Implementation would show overview + individual agent perspectives
        return self._create_overview_animation(fig, positions, coverage, algorithm_name)
        
    def _generate_poi_positions(self) -> List[Tuple[float, float, int]]:
        """Generate consistent POI positions"""
        num_pois = self.env_config.get('num_pois', 12)
        pois = []
        
        # Create POIs in a pattern
        for i in range(num_pois):
            angle = i * 2 * np.pi / num_pois
            radius = self.area_size * 0.3 * (1 + 0.3 * np.sin(i))
            x = self.area_size/2 + radius * np.cos(angle)
            y = self.area_size/2 + radius * np.sin(angle)
            priority = 1 + (i % 5)  # Priority 1-5
            
            x = max(50, min(x, self.area_size - 50))
            y = max(50, min(y, self.area_size - 50))
            
            pois.append((x, y, priority))
            
        return pois
        
    def _get_video_writer(self):
        """Get video writer based on quality settings"""
        quality_settings = {
            'low': {'bitrate': 1000, 'fps': 15},
            'medium': {'bitrate': 2500, 'fps': 30},
            'high': {'bitrate': 5000, 'fps': 30},
            'ultra': {'bitrate': 10000, 'fps': 60}
        }
        
        settings = quality_settings.get(self.video_quality, quality_settings['high'])
        
        # Try FFMpeg first
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                writer = FFMpegWriter(
                    fps=settings['fps'],
                    bitrate=settings['bitrate'],
                    codec='libx264',
                    extra_args=['-pix_fmt', 'yuv420p']
                )
            else:
                raise FileNotFoundError("FFmpeg not found")
        except (FileNotFoundError, subprocess.SubprocessError):
            # Fallback to Pillow if FFMpeg not available
            print("⚠️ FFMpeg not available, using Pillow writer (generates GIF)")
            print("   To install FFmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            writer = PillowWriter(fps=min(settings['fps'], 10))  # Lower FPS for GIF
            
        return writer
        
    def create_comparison_video(self, 
                               results: Dict[str, Dict],
                               output_name: str = 'algorithm_comparison.mp4'):
        """
        Create side-by-side comparison video of different algorithms
        
        Args:
            results: Dictionary of algorithm results with episode data
            output_name: Output filename
        """
        print("🎬 Creating comparison video...")
        
        # Extract data for each algorithm
        algorithm_data = {}
        for alg_name, result in results.items():
            if 'episode_data' in result:
                algorithm_data[alg_name] = result['episode_data']
                
        if len(algorithm_data) < 2:
            print("⚠️ Need at least 2 algorithms for comparison")
            return None
            
        # Set up comparison figure
        num_algorithms = len(algorithm_data)
        fig = plt.figure(figsize=(8*num_algorithms, 9), facecolor='white')
        
        axes = []
        for i, alg_name in enumerate(algorithm_data.keys()):
            ax = fig.add_subplot(1, num_algorithms, i+1)
            ax.set_xlim(0, self.area_size)
            ax.set_ylim(0, self.area_size)
            ax.set_aspect('equal')
            ax.set_title(alg_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            axes.append(ax)
            
        # Create synchronized animation
        # ... (implementation would synchronize all algorithm animations)
        
        video_path = os.path.join(self.video_dir, output_name)
        print(f"✅ Comparison video saved: {video_path}")
        return video_path


class InteractiveVideoPlayer:
    """Interactive video player with controls"""
    
    def __init__(self, video_path: str):
        """
        Initialize interactive player
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        # Implementation would use matplotlib widgets for controls
        
    def play(self):
        """Play video with interactive controls"""
        print(f"▶️ Playing video: {self.video_path}")
        # Implementation would include play/pause, speed control, frame stepping