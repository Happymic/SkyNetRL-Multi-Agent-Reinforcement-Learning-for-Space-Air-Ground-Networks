"""
Video Visualization Integration Module
Connects video generation with training and evaluation systems
"""

import numpy as np
from typing import Dict, List, Optional, Any
import os
import json
from datetime import datetime

from .video_generator import VideoGenerator
from .realtime_3d_viewer import Realtime3DViewer, VideoExporter


class VideoVisualizationManager:
    """Manages all video visualization components"""
    
    def __init__(self, env_config: Dict, output_dir: str):
        """
        Initialize visualization manager
        
        Args:
            env_config: Environment configuration
            output_dir: Base output directory
        """
        self.env_config = env_config
        self.output_dir = output_dir
        
        # Create video output directory
        self.video_output_dir = os.path.join(output_dir, 'videos', 
                                             datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # Initialize components
        self.video_generator = VideoGenerator(env_config, self.video_output_dir)
        self.realtime_viewer = None
        self.video_exporter = None
        
        # Recording state
        self.is_recording = False
        self.episode_buffer = []
        self.current_episode_data = {
            'position_history': [],
            'coverage_history': [],
            'reward_history': [],
            'action_history': [],
            'metadata': {}
        }
        
        # Settings
        self.auto_generate_video = True
        self.realtime_enabled = False
        self.video_modes = ['overview', 'tracking', 'orbiting', 'split_screen']
        self.current_mode_idx = 0
        
    def enable_realtime_viewer(self):
        """Enable real-time 3D visualization"""
        if not self.realtime_viewer:
            self.realtime_viewer = Realtime3DViewer(self.env_config)
            self.video_exporter = VideoExporter(self.realtime_viewer)
            self.realtime_viewer.start()
            self.realtime_enabled = True
            print("✅ Real-time 3D viewer enabled")
            
    def disable_realtime_viewer(self):
        """Disable real-time 3D visualization"""
        if self.realtime_viewer:
            self.realtime_viewer.stop()
            self.realtime_viewer = None
            self.realtime_enabled = False
            print("⏹ Real-time 3D viewer disabled")
            
    def start_episode_recording(self, algorithm_name: str, episode: int):
        """Start recording a new episode"""
        self.is_recording = True
        self.current_episode_data = {
            'position_history': [],
            'coverage_history': [],
            'reward_history': [],
            'action_history': [],
            'metadata': {
                'algorithm': algorithm_name,
                'episode': episode,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if self.realtime_enabled and self.video_exporter:
            self.video_exporter.start_recording()
            
    def record_step(self, 
                   agent_positions: Dict,
                   coverage_status: List,
                   rewards: float,
                   actions: Optional[np.ndarray] = None):
        """Record a single step of the episode"""
        if not self.is_recording:
            return
            
        # Store data
        self.current_episode_data['position_history'].append(agent_positions.copy())
        self.current_episode_data['coverage_history'].append(coverage_status.copy())
        self.current_episode_data['reward_history'].append(rewards)
        
        if actions is not None:
            self.current_episode_data['action_history'].append(actions.tolist())
            
        # Update real-time viewer if enabled
        if self.realtime_enabled and self.realtime_viewer:
            self.realtime_viewer.update_positions(agent_positions)
            self.realtime_viewer.update_coverage(coverage_status)
            
    def end_episode_recording(self, generate_video: bool = None):
        """End episode recording and optionally generate video"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        # Stop real-time recording if active
        if self.realtime_enabled and self.video_exporter:
            frames = self.video_exporter.stop_recording()
            # Could save frames here if needed
            
        # Add to episode buffer
        self.episode_buffer.append(self.current_episode_data.copy())
        
        # Generate video if requested
        if generate_video is None:
            generate_video = self.auto_generate_video
            
        video_path = None
        if generate_video and len(self.current_episode_data['position_history']) > 0:
            video_path = self.generate_episode_video(
                self.current_episode_data,
                camera_mode=self.video_modes[self.current_mode_idx]
            )
            
        return video_path
        
    def generate_episode_video(self, 
                              episode_data: Dict,
                              camera_mode: str = 'overview') -> str:
        """Generate video for a specific episode"""
        algorithm = episode_data['metadata'].get('algorithm', 'unknown')
        episode = episode_data['metadata'].get('episode', 0)
        
        output_name = f"{algorithm}_ep{episode}_{camera_mode}.mp4"
        
        video_path = self.video_generator.generate_episode_video(
            episode_data,
            algorithm,
            camera_mode,
            output_name
        )
        
        return video_path
        
    def generate_comparison_video(self, algorithms_data: Dict[str, Dict]):
        """Generate comparison video for multiple algorithms"""
        # Prepare data for comparison
        comparison_data = {}
        
        for alg_name, episodes in algorithms_data.items():
            if episodes:
                # Use best episode for comparison
                best_episode = max(episodes, 
                                 key=lambda x: sum(x.get('reward_history', [0])))
                comparison_data[alg_name] = {'episode_data': best_episode}
                
        # Generate comparison video
        if len(comparison_data) >= 2:
            video_path = self.video_generator.create_comparison_video(
                comparison_data,
                'algorithm_comparison.mp4'
            )
            return video_path
        else:
            print("⚠️ Need at least 2 algorithms for comparison video")
            return None
            
    def generate_highlight_reel(self, 
                               num_highlights: int = 5,
                               highlight_duration: int = 50):
        """Generate a highlight reel from best moments"""
        if not self.episode_buffer:
            print("⚠️ No episodes recorded yet")
            return None
            
        # Sort episodes by total reward
        sorted_episodes = sorted(self.episode_buffer, 
                               key=lambda x: sum(x.get('reward_history', [0])),
                               reverse=True)
        
        # Take top episodes
        highlights = sorted_episodes[:num_highlights]
        
        # Create highlight video data
        highlight_data = {
            'position_history': [],
            'coverage_history': [],
            'reward_history': [],
            'metadata': {
                'algorithm': 'highlight_reel',
                'num_highlights': num_highlights
            }
        }
        
        for episode in highlights:
            # Take best segment from each episode
            rewards = episode.get('reward_history', [])
            if len(rewards) > highlight_duration:
                # Find best segment
                best_start = 0
                best_sum = sum(rewards[:highlight_duration])
                
                for i in range(1, len(rewards) - highlight_duration):
                    current_sum = sum(rewards[i:i+highlight_duration])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_start = i
                        
                # Extract segment
                end = best_start + highlight_duration
                highlight_data['position_history'].extend(
                    episode['position_history'][best_start:end]
                )
                highlight_data['coverage_history'].extend(
                    episode['coverage_history'][best_start:end]
                )
                highlight_data['reward_history'].extend(
                    rewards[best_start:end]
                )
            else:
                # Use entire episode if shorter than highlight duration
                highlight_data['position_history'].extend(episode['position_history'])
                highlight_data['coverage_history'].extend(episode['coverage_history'])
                highlight_data['reward_history'].extend(rewards)
                
        # Generate highlight video
        video_path = self.video_generator.generate_episode_video(
            highlight_data,
            'Highlight Reel',
            'overview',
            'highlight_reel.mp4'
        )
        
        return video_path
        
    def cycle_camera_mode(self):
        """Cycle through different camera modes"""
        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.video_modes)
        mode = self.video_modes[self.current_mode_idx]
        print(f"📹 Camera mode: {mode}")
        return mode
        
    def save_episode_data(self, filename: str = 'episode_data.json'):
        """Save recorded episode data to file"""
        filepath = os.path.join(self.video_output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_buffer = []
        for episode in self.episode_buffer:
            serializable_episode = {
                'position_history': episode['position_history'],
                'coverage_history': episode['coverage_history'],
                'reward_history': episode['reward_history'],
                'action_history': episode.get('action_history', []),
                'metadata': episode['metadata']
            }
            serializable_buffer.append(serializable_episode)
            
        with open(filepath, 'w') as f:
            json.dump(serializable_buffer, f, indent=2)
            
        print(f"💾 Episode data saved: {filepath}")
        
    def load_episode_data(self, filepath: str) -> List[Dict]:
        """Load episode data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.episode_buffer = data
        print(f"📂 Loaded {len(data)} episodes from {filepath}")
        return data
        
    def get_statistics(self) -> Dict:
        """Get statistics about recorded episodes"""
        if not self.episode_buffer:
            return {}
            
        stats = {
            'num_episodes': len(self.episode_buffer),
            'total_steps': sum(len(ep['position_history']) for ep in self.episode_buffer),
            'avg_episode_length': np.mean([len(ep['position_history']) for ep in self.episode_buffer]),
            'avg_total_reward': np.mean([sum(ep['reward_history']) for ep in self.episode_buffer]),
            'best_episode_reward': max(sum(ep['reward_history']) for ep in self.episode_buffer),
            'algorithms': list(set(ep['metadata'].get('algorithm', 'unknown') for ep in self.episode_buffer))
        }
        
        return stats


class EpisodeRecorder:
    """Simplified episode recorder for easy integration"""
    
    def __init__(self, env, viz_manager: VideoVisualizationManager):
        """
        Initialize episode recorder
        
        Args:
            env: Environment instance
            viz_manager: Visualization manager instance
        """
        self.env = env
        self.viz_manager = viz_manager
        
    def record_episode(self, 
                      agent,
                      algorithm_name: str,
                      episode_num: int,
                      max_steps: int = 200,
                      generate_video: bool = True) -> Dict:
        """
        Record a single episode with video
        
        Args:
            agent: Agent to evaluate
            algorithm_name: Name of algorithm
            episode_num: Episode number
            max_steps: Maximum steps per episode
            generate_video: Whether to generate video
            
        Returns:
            Episode statistics and video path
        """
        # Start recording
        self.viz_manager.start_episode_recording(algorithm_name, episode_num)
        
        # Reset environment
        obs = self.env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from agent
            if hasattr(agent, 'act'):
                actions = agent.act(obs)
            else:
                # Random actions for testing
                actions = self.env.action_space.sample()
                
            # Step environment
            next_obs, rewards, done, info = self.env.step(actions)
            
            # Record step
            agent_positions = self._extract_positions(obs)
            coverage_status = info.get('coverage_status', [])
            
            self.viz_manager.record_step(
                agent_positions,
                coverage_status,
                rewards,
                actions
            )
            
            total_reward += rewards
            obs = next_obs
            
            if done:
                break
                
        # End recording and generate video
        video_path = self.viz_manager.end_episode_recording(generate_video)
        
        return {
            'total_reward': total_reward,
            'steps': step + 1,
            'video_path': video_path
        }
        
    def _extract_positions(self, obs) -> Dict:
        """Extract agent positions from observation"""
        positions = {}
        
        # This would need to be adapted based on your observation structure
        # Example implementation:
        if isinstance(obs, dict):
            for i, agent_obs in enumerate(obs.values()):
                if len(agent_obs) >= 3:
                    positions[f'agent_{i}'] = (
                        float(agent_obs[0]) * self.env.area_size,  # x
                        float(agent_obs[1]) * self.env.area_size,  # y
                        float(agent_obs[2]) * 300 if len(agent_obs) > 2 else 50  # z
                    )
        elif isinstance(obs, np.ndarray):
            # Assuming flattened observation
            num_agents = self.env.num_agents
            for i in range(num_agents):
                if i * 3 + 2 < len(obs):
                    positions[f'agent_{i}'] = (
                        float(obs[i*3]) * self.env.area_size,
                        float(obs[i*3 + 1]) * self.env.area_size,
                        float(obs[i*3 + 2]) * 300 if i*3 + 2 < len(obs) else 50
                    )
                    
        return positions