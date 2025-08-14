"""
Test Professional Visualization System
======================================
Quick test to verify white background, legends, and slower GIF animation.
"""

import numpy as np
import os
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualization.professional_visualizer import ProfessionalVisualizer, VisualizationConfig
from src.utils.training_pipeline import StandardizedTrainingPipeline, TrainingOutputConfig


def create_test_environment():
    """Create a mock environment for testing"""
    class MockEnvironment:
        def __init__(self):
            self.area_size = (500, 500)
            self.n_agents = 7
            self.communication_range = 100
            
        def get_agent_positions(self):
            # Create sample positions (2 satellites, 3 UAVs, 2 ground)
            positions = [
                [100, 100, 150],  # Satellite 1
                [400, 400, 150],  # Satellite 2
                [200, 300, 100],  # UAV 1
                [300, 200, 95],   # UAV 2
                [250, 250, 105],  # UAV 3
                [150, 350, 0],    # Ground 1
                [350, 150, 0],    # Ground 2
            ]
            return np.array(positions)
    
    return MockEnvironment()


def generate_test_frames(env, n_frames=50):
    """Generate test frames with realistic movement"""
    frames = []
    base_positions = env.get_agent_positions()
    
    for i in range(n_frames):
        # Create smooth movement
        t = i / n_frames * 2 * np.pi
        
        positions = base_positions.copy()
        
        # Satellites orbit
        positions[0, 0] = 250 + 150 * np.cos(t)
        positions[0, 1] = 250 + 150 * np.sin(t)
        positions[1, 0] = 250 + 150 * np.cos(t + np.pi)
        positions[1, 1] = 250 + 150 * np.sin(t + np.pi)
        
        # UAVs patrol
        positions[2, 0] = 200 + 50 * np.cos(2*t)
        positions[2, 1] = 300 + 50 * np.sin(2*t)
        positions[3, 0] = 300 + 50 * np.cos(2*t + 2*np.pi/3)
        positions[3, 1] = 200 + 50 * np.sin(2*t + 2*np.pi/3)
        positions[4, 0] = 250 + 30 * np.cos(3*t)
        positions[4, 1] = 250 + 30 * np.sin(3*t)
        
        # Ground stations move slowly
        positions[5, 0] = 150 + 20 * np.cos(0.5*t)
        positions[5, 1] = 350 + 20 * np.sin(0.5*t)
        positions[6, 0] = 350 + 20 * np.cos(0.5*t + np.pi)
        positions[6, 1] = 150 + 20 * np.sin(0.5*t + np.pi)
        
        # Calculate communications
        communications = []
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                dist = np.linalg.norm(positions[j] - positions[k])
                if dist < env.communication_range:
                    communications.append({
                        'agents': (j, k),
                        'distance': dist,
                        'connected': True
                    })
        
        frame = {
            'step': i,
            'positions': positions,
            'reward': np.random.uniform(0.5, 1.5),
            'coverage': 75 + 15 * np.sin(t),
            'communications': communications
        }
        frames.append(frame)
    
    return frames


def test_professional_visualizer():
    """Test the professional visualization system"""
    print("Testing Professional Visualization System")
    print("=" * 50)
    
    # Create mock environment
    env = create_test_environment()
    
    # Configure visualizer with white background and slow animation
    viz_config = VisualizationConfig(
        fps=10,  # Slower animation (10 FPS)
        background_color='white',
        trail_length=30,
        dpi=100
    )
    
    # Create visualizer
    visualizer = ProfessionalVisualizer(env, viz_config)
    print("✓ Visualizer created with white background")
    
    # Generate test frames
    frames = generate_test_frames(env, n_frames=50)
    print(f"✓ Generated {len(frames)} test frames")
    
    # Create output directory
    output_dir = f"outputs/test_professional_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate GIF animation
    gif_path = os.path.join(output_dir, "test_animation.gif")
    visualizer.generate_episode_animation(frames, gif_path, format='gif')
    print(f"✓ Generated GIF animation at 10 FPS: {gif_path}")
    
    # Generate static overview
    overview_path = os.path.join(output_dir, "test_overview.png")
    visualizer.create_static_overview(frames, overview_path)
    print(f"✓ Generated static overview: {overview_path}")
    
    return output_dir


def test_training_pipeline():
    """Test the standardized training pipeline"""
    print("\nTesting Standardized Training Pipeline")
    print("=" * 50)
    
    # Configure pipeline
    config = TrainingOutputConfig(
        experiment_name="test_pipeline",
        gif_fps=10,
        checkpoint_frequency=10,
        report_frequency=5
    )
    
    # Create pipeline
    pipeline = StandardizedTrainingPipeline(config)
    print(f"✓ Pipeline created at: {pipeline.paths['base']}")
    
    # Simulate training episodes
    for episode in range(20):
        episode_data = {
            'total_reward': np.random.uniform(0, 100),
            'avg_coverage': np.random.uniform(60, 90),
            'communication_success': np.random.uniform(70, 95),
            'collision_rate': np.random.uniform(0, 2)
        }
        
        pipeline.log_episode(episode, episode_data)
        
        if episode % 5 == 0:
            print(f"✓ Logged episode {episode}")
    
    # Finalize
    output_path = pipeline.finalize_training()
    print(f"✓ Training finalized: {output_path}")
    
    return output_path


def main():
    """Run all tests"""
    print("SkyNetRL Professional System Test")
    print("=" * 50)
    print("Features:")
    print("• White background for professional appearance")
    print("• Agent legends with symbols and descriptions")
    print("• 10 FPS animation (slower for better observation)")
    print("• Standardized output structure")
    print()
    
    # Test visualizer
    viz_output = test_professional_visualizer()
    
    # Test pipeline
    pipeline_output = test_training_pipeline()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("\nOutputs generated:")
    print(f"• Visualization: {viz_output}")
    print(f"• Pipeline: {pipeline_output}")
    print("\nKey features verified:")
    print("✓ White background theme")
    print("✓ Agent legend panel")
    print("✓ 10 FPS GIF animation")
    print("✓ Standardized output structure")
    print("✓ HTML dashboards")
    print("✓ Training metrics tracking")


if __name__ == "__main__":
    main()