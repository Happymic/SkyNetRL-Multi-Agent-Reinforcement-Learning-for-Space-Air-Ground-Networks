#!/usr/bin/env python3
"""
Enhanced Training Visualization Demo
Shows comprehensive training visualization with detailed information panels
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
import time

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.visualization.training_integration import TrainingVisualizer, TrainingCallback
from src.visualization.enhanced_training_visualizer import EnhancedTrainingVisualizer


def create_realistic_training_episode(episode_num: int, algorithm_name: str):
    """Create realistic training episode data with proper metrics"""
    
    num_steps = 80  # Shorter for demo
    num_agents = 5
    num_pois = 12
    area_size = 1000
    
    # Create more realistic agent trajectories
    episode_data = {
        'position_history': [],
        'coverage_history': [],
        'reward_history': [],
        'action_history': [],
        'agent_states': [],
        'communication_data': [],
        'metadata': {
            'episode': episode_num,
            'algorithm': algorithm_name,
            'num_agents': num_agents,
            'area_size': area_size
        }
    }
    
    # POI positions (fixed for consistency)
    poi_positions = []
    for i in range(num_pois):
        if i < 3:  # Center high-priority POIs
            angle = i * 2 * np.pi / 3
            x = area_size/2 + 150 * np.cos(angle)
            y = area_size/2 + 150 * np.sin(angle)
        elif i < 6:  # Middle ring
            angle = (i-3) * 2 * np.pi / 3
            x = area_size/2 + 300 * np.cos(angle)
            y = area_size/2 + 300 * np.sin(angle)
        else:  # Outer ring
            angle = (i-6) * 2 * np.pi / (num_pois-6)
            x = area_size/2 + 400 * np.cos(angle)
            y = area_size/2 + 400 * np.sin(angle)
        
        poi_positions.append((x, y))
    
    # Agent initial positions and targets
    agent_positions = []
    agent_targets = []
    
    for i in range(num_agents):
        # Starting positions around the perimeter
        start_angle = i * 2 * np.pi / num_agents
        start_radius = area_size * 0.45
        start_x = area_size/2 + start_radius * np.cos(start_angle)
        start_y = area_size/2 + start_radius * np.sin(start_angle)
        
        # Different starting altitudes
        if i < 2:  # Satellites
            start_z = 200
        elif i < 5:  # UAVs
            start_z = 100
        else:  # Ground stations
            start_z = 10
            
        agent_positions.append([start_x, start_y, start_z])
        
        # Assign target POIs
        target_poi = poi_positions[i % len(poi_positions)]
        agent_targets.append(target_poi)
    
    # Coverage tracking
    poi_covered = [False] * num_pois
    coverage_times = [0] * num_pois
    
    # Generate episode steps
    for step in range(num_steps):
        frame_positions = {}
        step_reward = 0
        
        # Move agents towards targets with some intelligence
        for i in range(num_agents):
            current_pos = agent_positions[i]
            target = agent_targets[i]
            
            # Move towards target
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 10:  # If not at target
                # Move towards target
                move_speed = 8 if i >= 2 else 4  # UAVs faster than satellites
                step_x = move_speed * dx / distance
                step_y = move_speed * dy / distance
                
                current_pos[0] += step_x
                current_pos[1] += step_y
            else:
                # At target, pick new target
                available_targets = [poi for poi in poi_positions if poi != target]
                if available_targets:
                    agent_targets[i] = available_targets[np.random.randint(len(available_targets))]
            
            # Add some noise for realism
            current_pos[0] += np.random.randn() * 2
            current_pos[1] += np.random.randn() * 2
            
            # Constrain to area
            current_pos[0] = max(50, min(area_size-50, current_pos[0]))
            current_pos[1] = max(50, min(area_size-50, current_pos[1]))
            
            frame_positions[f'agent_{i}'] = tuple(current_pos)
        
        # Check POI coverage
        coverage_radii = [250, 250, 120, 120, 120, 80, 80]  # Different for agent types
        current_coverage = []
        
        for poi_idx, (poi_x, poi_y) in enumerate(poi_positions):
            is_covered = False
            
            for agent_idx, agent_key in enumerate(frame_positions.keys()):
                if agent_idx < len(coverage_radii):
                    agent_pos = frame_positions[agent_key]
                    dist = np.sqrt((poi_x - agent_pos[0])**2 + (poi_y - agent_pos[1])**2)
                    
                    if dist < coverage_radii[agent_idx]:
                        is_covered = True
                        if not poi_covered[poi_idx]:
                            poi_covered[poi_idx] = True
                            coverage_times[poi_idx] = step
                            # Reward for new coverage
                            priority = min(5, max(1, (poi_idx % 5) + 1))
                            step_reward += priority * 10
                        break
            
            current_coverage.append(is_covered)
        
        # Calculate step reward
        coverage_rate = sum(current_coverage) / len(current_coverage)
        step_reward += coverage_rate * 5  # Base coverage reward
        
        # Energy efficiency bonus (better algorithms use less energy)
        if algorithm_name == 'ae_maddpg':
            step_reward += 2  # More efficient
        elif algorithm_name == 'qmix':
            step_reward += 1
        
        # Add some variation based on algorithm performance
        if algorithm_name == 'ae_maddpg':
            step_reward *= (1.2 + np.sin(step * 0.1) * 0.2)  # Better performance curve
        elif algorithm_name == 'baseline_maddpg':
            step_reward *= (1.0 + np.sin(step * 0.15) * 0.15)
        else:
            step_reward *= (0.8 + np.sin(step * 0.2) * 0.1)
        
        # Store step data
        episode_data['position_history'].append(frame_positions)
        episode_data['coverage_history'].append(current_coverage)
        episode_data['reward_history'].append(step_reward)
        episode_data['action_history'].append([np.random.randn(4) for _ in range(num_agents)])
        
        # Agent states
        agent_states = {}
        for i in range(num_agents):
            if i < 2:  # Satellites
                agent_type = 'satellite'
                energy = 100  # Satellites don't lose energy
            elif i < 5:  # UAVs
                agent_type = 'uav'
                energy = max(0, 100 - step * 1.2)  # UAVs lose energy
            else:  # Ground stations
                agent_type = 'ground_station'
                energy = 100  # Ground stations don't lose energy
                
            agent_states[f'agent_{i}'] = {
                'type': agent_type,
                'energy': energy,
                'active': energy > 5,
                'communication_active': True
            }
        
        episode_data['agent_states'].append(agent_states)
        
        # Communication data
        comm_data = {
            'active_links': list(range(min(num_agents-1, 8))),  # Some active links
            'network_efficiency': 0.85 + np.random.randn() * 0.1,
            'data_transmitted': step * 0.5
        }
        episode_data['communication_data'].append(comm_data)
    
    return episode_data


def run_enhanced_training_demo():
    """Run enhanced training visualization demo"""
    
    print("=" * 80)
    print("🚀 Enhanced Training Visualization Demo")
    print("=" * 80)
    
    # Load configuration
    with open("configs/default_config.json", "r") as f:
        config = json.load(f)
    
    env_config = config['environment']
    output_dir = "results/enhanced_training_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different algorithms
    algorithms = ['ae_maddpg', 'baseline_maddpg', 'qmix']
    
    print(f"\n📊 Testing Enhanced Visualization with {len(algorithms)} algorithms")
    print("-" * 60)
    
    for alg_idx, algorithm in enumerate(algorithms):
        print(f"\n🎯 Algorithm: {algorithm.upper()}")
        print(f"   Generating realistic training episode...")
        
        # Initialize training visualizer
        training_viz = TrainingVisualizer(env_config, output_dir, enable_video=True)
        
        # Simulate training episode
        episode_num = alg_idx + 1
        episode_data = create_realistic_training_episode(episode_num, algorithm)
        
        # Calculate episode statistics
        total_reward = sum(episode_data['reward_history'])
        final_coverage = episode_data['coverage_history'][-1]
        coverage_rate = sum(final_coverage) / len(final_coverage) * 100
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': len(episode_data['position_history']),
            'coverage_rate': coverage_rate,
            'energy_efficiency': total_reward / len(episode_data['position_history']),
            'collision_count': 0,
            'communication_efficiency': 0.85
        }
        
        print(f"   📈 Episode Stats:")
        print(f"      Total Reward: {total_reward:.1f}")
        print(f"      Coverage Rate: {coverage_rate:.1f}%")
        print(f"      Steps: {episode_stats['steps']}")
        
        # Generate enhanced visualization
        print(f"   🎬 Creating enhanced visualization...")
        video_path = training_viz.visualizer.create_training_visualization(
            episode_data,
            episode_stats,
            algorithm,
            episode_num
        )
        
        if video_path:
            print(f"   ✅ Enhanced video generated: {os.path.basename(video_path)}")
        else:
            print(f"   ⚠️ Video generation failed")
    
    print(f"\n📁 All enhanced visualizations saved in: {output_dir}")
    
    # Show what's included in enhanced visualization
    print("\n" + "="*80)
    print("🎨 ENHANCED VISUALIZATION FEATURES")
    print("="*80)
    print("📊 Main 2D View:")
    print("   • Agent positions with distinct icons (🛰️ satellites, 🚁 UAVs, 📡 ground stations)")
    print("   • POI status with priority colors and coverage indicators")
    print("   • Coverage circles showing sensor ranges")
    print("   • Communication links between agents")
    print("   • Agent trails showing movement history")
    
    print("\n🌐 3D View Panel:")
    print("   • Three-dimensional agent positions")
    print("   • POI pillars with height indicating priority")
    print("   • Coverage areas projected on ground")
    print("   • Altitude differences clearly visible")
    
    print("\n📋 Information Panels:")
    print("   • Agent Status: Position, energy, coverage radius, speed")
    print("   • POI Coverage: Individual status, priority levels, completion")
    print("   • Performance Metrics: Real-time algorithm performance")
    
    print("\n📈 Training Metrics:")
    print("   • Episode reward progression")
    print("   • Coverage rate over time")
    print("   • Energy efficiency indicators")
    print("   • Communication network status")
    
    print("\n🎯 Key Improvements:")
    print("   • Fixed 3D visualization (properly working now)")
    print("   • Detailed legends and symbol explanations")
    print("   • Real-time performance indicators")
    print("   • Integration with training process")
    print("   • Comprehensive information display")
    print("="*80)


def run_training_integration_demo():
    """Demonstrate integration with training process"""
    
    print("\n🔗 Training Integration Demo")
    print("-" * 40)
    
    # Load config
    with open("configs/default_config.json", "r") as f:
        config = json.load(f)
    
    env_config = config['environment']
    output_dir = "results/training_integration_demo"
    
    # Initialize training visualizer
    training_viz = TrainingVisualizer(env_config, output_dir)
    callback = TrainingCallback(training_viz)
    
    # Simulate training process
    algorithm_name = "enhanced_ae_maddpg"
    num_episodes = 3
    
    print(f"📚 Simulating {num_episodes} episodes of {algorithm_name}...")
    
    for episode in range(1, num_episodes + 1):
        print(f"\n🎯 Episode {episode}:")
        
        # Start episode
        callback.on_episode_start(episode, algorithm_name)
        
        # Generate episode data
        episode_data = create_realistic_training_episode(episode, algorithm_name)
        
        # Simulate step-by-step training
        for step in range(len(episode_data['position_history'])):
            # Extract step data
            positions = episode_data['position_history'][step]
            rewards = episode_data['reward_history'][step]
            coverage = episode_data['coverage_history'][step]
            
            # Create dummy observations
            observations = {f'agent_{i}': np.array([
                pos[0] / 1000, pos[1] / 1000, pos[2] / 300
            ]) for i, pos in enumerate(positions.values())}
            
            actions = np.array([np.random.randn(4) for _ in range(len(positions))])
            info = {'coverage_status': coverage}
            
            # Record step
            callback.on_step(observations, actions, rewards, info)
        
        # End episode
        stats, video_path = callback.on_episode_end()
        
        print(f"   📊 Final Stats: Reward={stats['total_reward']:.1f}, Coverage={stats['coverage_rate']:.1f}%")
        if video_path:
            print(f"   🎬 Video: {os.path.basename(video_path)}")
    
    # Complete training
    summary = callback.on_training_complete()
    
    print(f"\n✅ Training integration demo completed!")
    print(f"📁 Results saved in: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Training Visualization Demo")
    parser.add_argument("--enhanced", action="store_true", help="Run enhanced visualization demo")
    parser.add_argument("--integration", action="store_true", help="Run training integration demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    
    args = parser.parse_args()
    
    if args.all or (not args.enhanced and not args.integration):
        run_enhanced_training_demo()
        run_training_integration_demo()
    else:
        if args.enhanced:
            run_enhanced_training_demo()
        if args.integration:
            run_training_integration_demo()
    
    print("\n💡 Tips:")
    print("💡 Use --enhanced to see detailed visualization features")
    print("💡 Use --integration to see training process integration")
    print("💡 Use --all to run complete demonstration")