#!/usr/bin/env python3
"""
Demo script for video visualization of multi-agent movement
Shows different camera modes and visualization capabilities
"""

import sys
import os
from pathlib import Path
import numpy as np
import json

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.visualization.video_integration import VideoVisualizationManager, EpisodeRecorder
from src.visualization.video_generator import VideoGenerator
from src.environments.enhanced_sagin_env import EnhancedSAGINEnvironment


def create_demo_trajectory():
    """Create demo trajectory data for testing"""
    num_steps = 100
    num_agents = 5
    area_size = 1000
    
    position_history = []
    coverage_history = []
    reward_history = []
    
    for t in range(num_steps):
        frame_positions = {}
        
        for i in range(num_agents):
            # Create different movement patterns for each agent
            if i == 0:  # Circular motion
                angle = t * 2 * np.pi / num_steps
                x = area_size/2 + 200 * np.cos(angle)
                y = area_size/2 + 200 * np.sin(angle)
                z = 200
            elif i == 1:  # Figure-8 motion
                angle = t * 4 * np.pi / num_steps
                x = area_size/2 + 150 * np.cos(angle)
                y = area_size/2 + 150 * np.sin(angle/2)
                z = 150
            elif i == 2:  # Spiral motion
                angle = t * 2 * np.pi / num_steps
                radius = 50 + t * 3
                x = area_size/2 + radius * np.cos(angle)
                y = area_size/2 + radius * np.sin(angle)
                z = 100
            elif i == 3:  # Linear patrol
                x = 100 + (area_size - 200) * (t % 40) / 40 if (t // 40) % 2 == 0 else area_size - 100 - (area_size - 200) * (t % 40) / 40
                y = 200 if (t // 40) % 4 < 2 else area_size - 200
                z = 50
            else:  # Random walk
                x = area_size/2 + np.random.randn() * 20
                y = area_size/2 + np.random.randn() * 20
                z = 30
                
            frame_positions[f'agent_{i}'] = (x, y, z)
            
        position_history.append(frame_positions)
        
        # Simulate coverage (gradually covering POIs)
        coverage = [t > i * 5 for i in range(12)]
        coverage_history.append(coverage)
        
        # Simulate rewards
        reward = np.sin(t * 0.1) * 10 + 20
        reward_history.append(reward)
        
    return {
        'position_history': position_history,
        'coverage_history': coverage_history,
        'reward_history': reward_history,
        'metadata': {
            'algorithm': 'demo_trajectory',
            'episode': 1
        }
    }


def run_video_demo():
    """Run comprehensive video visualization demo"""
    
    print("=" * 80)
    print("🎬 SkyNetRL Video Visualization Demo")
    print("=" * 80)
    
    # Load configuration
    with open("configs/default_config.json", "r") as f:
        config = json.load(f)
    
    env_config = config['environment']
    
    # Create output directory
    output_dir = "results/video_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualization manager
    print("\n📋 Initializing Video Visualization Manager...")
    viz_manager = VideoVisualizationManager(env_config, output_dir)
    
    # Generate demo trajectory
    print("\n🎯 Generating demo trajectory data...")
    demo_data = create_demo_trajectory()
    
    # Test different camera modes
    camera_modes = ['overview', 'tracking', 'orbiting', 'split_screen']
    
    print("\n🎥 Generating videos with different camera modes:")
    print("-" * 40)
    
    for mode in camera_modes:
        print(f"\n📹 Camera Mode: {mode}")
        print(f"   Generating video...")
        
        video_path = viz_manager.video_generator.generate_episode_video(
            demo_data,
            f"Demo_{mode.title()}",
            mode,
            f"demo_{mode}.mp4"
        )
        
        if video_path:
            print(f"   ✅ Video saved: {video_path}")
        else:
            print(f"   ⚠️ Failed to generate video")
    
    # Test real-time 3D viewer (if available)
    try:
        print("\n🌐 Testing Real-time 3D Viewer...")
        print("-" * 40)
        
        # Check if OpenGL dependencies are available
        import pygame
        import OpenGL.GL
        
        print("   ✅ OpenGL dependencies available")
        print("   📝 Note: Real-time viewer runs in separate window")
        print("   📝 Controls:")
        print("      - Mouse drag: Rotate camera")
        print("      - Scroll: Zoom in/out")
        print("      - WASD: Move camera target")
        print("      - Space: Reset camera")
        
        # Enable real-time viewer
        viz_manager.enable_realtime_viewer()
        
        # Simulate some steps
        print("\n   🔄 Simulating agent movements...")
        for i in range(min(30, len(demo_data['position_history']))):
            viz_manager.realtime_viewer.update_positions(
                demo_data['position_history'][i]
            )
            viz_manager.realtime_viewer.update_coverage(
                demo_data['coverage_history'][i]
            )
            
        print("   ✅ Real-time simulation complete")
        
        # Disable viewer
        viz_manager.disable_realtime_viewer()
        
    except ImportError as e:
        print("\n⚠️ Real-time 3D viewer not available")
        print(f"   Missing dependencies: {e}")
        print("   Install with: pip install pygame PyOpenGL PyOpenGL_accelerate")
    
    # Generate comparison video (simulate multiple algorithms)
    print("\n🔀 Generating Algorithm Comparison Video...")
    print("-" * 40)
    
    algorithms_data = {}
    algorithm_names = ['AE-MADDPG', 'Baseline-MADDPG', 'Q-MIX']
    
    for alg_name in algorithm_names:
        # Create slightly different trajectories for each algorithm
        alg_data = create_demo_trajectory()
        
        # Modify trajectory slightly for variation
        for t, positions in enumerate(alg_data['position_history']):
            for agent_id in positions:
                x, y, z = positions[agent_id]
                # Add some variation
                noise = np.random.randn() * 10
                positions[agent_id] = (x + noise, y + noise, z)
        
        algorithms_data[alg_name] = [alg_data]
    
    comparison_path = viz_manager.generate_comparison_video(algorithms_data)
    if comparison_path:
        print(f"✅ Comparison video saved: {comparison_path}")
    
    # Generate highlight reel
    print("\n⭐ Generating Highlight Reel...")
    print("-" * 40)
    
    # Add multiple episodes to buffer for highlight reel
    for i in range(5):
        episode = create_demo_trajectory()
        viz_manager.episode_buffer.append(episode)
    
    highlight_path = viz_manager.generate_highlight_reel(num_highlights=3, highlight_duration=20)
    if highlight_path:
        print(f"✅ Highlight reel saved: {highlight_path}")
    
    # Save episode data
    print("\n💾 Saving episode data...")
    viz_manager.save_episode_data("demo_episodes.json")
    
    # Print statistics
    print("\n📊 Video Generation Statistics:")
    print("-" * 40)
    stats = viz_manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ Video Visualization Demo Complete!")
    print(f"📁 All videos saved in: {output_dir}/videos/")
    print("=" * 80)


def run_integration_test():
    """Test integration with actual environment"""
    
    print("\n🧪 Testing Environment Integration...")
    print("-" * 40)
    
    # Load config
    with open("configs/default_config.json", "r") as f:
        config = json.load(f)
    
    env_config = config['environment']
    
    # Create environment
    try:
        env = EnhancedSAGINEnvironment(env_config)
        print("✅ Environment created successfully")
        
        # Create visualization manager
        output_dir = "results/integration_test"
        viz_manager = VideoVisualizationManager(env_config, output_dir)
        
        # Create episode recorder
        recorder = EpisodeRecorder(env, viz_manager)
        
        # Record a test episode
        print("\n📹 Recording test episode...")
        
        # Create a dummy agent (random actions)
        class DummyAgent:
            def __init__(self, action_space):
                self.action_space = action_space
            
            def act(self, obs):
                return self.action_space.sample()
        
        dummy_agent = DummyAgent(env.action_space)
        
        result = recorder.record_episode(
            dummy_agent,
            "test_algorithm",
            episode_num=1,
            max_steps=50,
            generate_video=True
        )
        
        print(f"✅ Episode recorded:")
        print(f"   Total reward: {result['total_reward']:.2f}")
        print(f"   Steps: {result['steps']}")
        if result['video_path']:
            print(f"   Video: {result['video_path']}")
        
    except Exception as e:
        print(f"⚠️ Integration test failed: {e}")
        print("   This is expected if environment is not fully configured")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Visualization Demo")
    parser.add_argument("--full", action="store_true", help="Run full demo")
    parser.add_argument("--test", action="store_true", help="Run integration test")
    
    args = parser.parse_args()
    
    if args.test:
        run_integration_test()
    else:
        run_video_demo()
        
        if args.full:
            run_integration_test()
    
    print("\n💡 Tip: Use --full flag to run complete demo with integration test")
    print("💡 Tip: Use --test flag to only run integration test")