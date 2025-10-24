#!/usr/bin/env python3
"""
Agent Movement Visualization Demo
================================
Generates synthetic agent movement data and creates vivid visualizations.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import random

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from visualization.agent_movement_visualizer import AgentMovementVisualizer


def generate_realistic_agent_movement_data(num_timesteps=150, area_size=1000):
    """Generate realistic agent movement and coverage data"""
    
    # Agent configurations
    agents_config = [
        {'id': 0, 'type': 'satellite', 'coverage_radius': 200, 'speed': 30},
        {'id': 1, 'type': 'satellite', 'coverage_radius': 200, 'speed': 25},
        {'id': 2, 'type': 'uav', 'coverage_radius': 100, 'speed': 40},
        {'id': 3, 'type': 'uav', 'coverage_radius': 100, 'speed': 35},
        {'id': 4, 'type': 'uav', 'coverage_radius': 100, 'speed': 45},
        {'id': 5, 'type': 'ground_station', 'coverage_radius': 150, 'speed': 0}
    ]
    
    # Generate fixed targets
    num_targets = 25
    targets = []
    for i in range(num_targets):
        targets.append({
            'id': i,
            'x': random.uniform(50, area_size - 50),
            'y': random.uniform(50, area_size - 50),
            'priority': random.uniform(0.5, 1.0),
            'covered': False
        })
    
    # Initialize agent positions
    agent_positions = {}
    agent_waypoints = {}
    
    for agent in agents_config:
        agent_id = agent['id']
        
        if agent['type'] == 'ground_station':
            # Ground stations are stationary
            x = random.uniform(100, area_size - 100)
            y = random.uniform(100, area_size - 100)
            agent_positions[agent_id] = {'x': x, 'y': y}
            agent_waypoints[agent_id] = [(x, y)] * num_timesteps
        elif agent['type'] == 'satellite':
            # Satellites follow orbital-like patterns
            center_x = area_size / 2
            center_y = area_size / 2
            radius = area_size * 0.35
            
            waypoints = []
            for t in range(num_timesteps):
                angle = (t / num_timesteps) * 4 * np.pi + agent_id * np.pi / 3
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Keep within bounds
                x = np.clip(x, 50, area_size - 50)
                y = np.clip(y, 50, area_size - 50)
                waypoints.append((x, y))
            
            agent_waypoints[agent_id] = waypoints
            agent_positions[agent_id] = {'x': waypoints[0][0], 'y': waypoints[0][1]}
        else:  # UAVs
            # UAVs follow adaptive patrol patterns
            waypoints = []
            current_x = random.uniform(100, area_size - 100)
            current_y = random.uniform(100, area_size - 100)
            
            for t in range(num_timesteps):
                # Add some intelligent movement towards uncovered areas
                if t % 30 == 0:  # Change direction every 30 steps
                    # Pick a new target area
                    target_x = random.uniform(100, area_size - 100)
                    target_y = random.uniform(100, area_size - 100)
                else:
                    # Move towards current target
                    if len(waypoints) > 0:
                        prev_x, prev_y = waypoints[-1]
                        dx = target_x - prev_x
                        dy = target_y - prev_y
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        if dist > agent['speed']:
                            dx = dx / dist * agent['speed']
                            dy = dy / dist * agent['speed']
                        
                        current_x = prev_x + dx + random.uniform(-5, 5)
                        current_y = prev_y + dy + random.uniform(-5, 5)
                    else:
                        current_x = current_x + random.uniform(-20, 20)
                        current_y = current_y + random.uniform(-20, 20)
                
                # Keep within bounds
                current_x = np.clip(current_x, 50, area_size - 50)
                current_y = np.clip(current_y, 50, area_size - 50)
                waypoints.append((current_x, current_y))
            
            agent_waypoints[agent_id] = waypoints
            agent_positions[agent_id] = {'x': waypoints[0][0], 'y': waypoints[0][1]}
    
    # Generate episode data with coverage calculation
    episode_data = []
    
    for timestep in range(num_timesteps):
        # Update agent positions
        current_agents = []
        for agent in agents_config:
            agent_id = agent['id']
            x, y = agent_waypoints[agent_id][timestep]
            
            # Calculate energy level (decreases over time, different rates)
            if agent['type'] == 'ground_station':
                energy = 1.0  # Always full
            elif agent['type'] == 'satellite':
                energy = 0.8 + 0.2 * np.sin(timestep / 30)  # Periodic energy
            else:  # UAV
                energy = max(0.3, 1.0 - (timestep / num_timesteps) * 0.5 + random.uniform(-0.1, 0.1))
            
            current_agents.append({
                'id': agent_id,
                'type': agent['type'],
                'x': x,
                'y': y,
                'coverage_radius': agent['coverage_radius'],
                'energy_level': energy,
                'active': energy > 0.2
            })
        
        # Calculate coverage for each target
        current_targets = []
        covered_count = 0
        
        for target in targets:
            is_covered = False
            
            for agent_data in current_agents:
                if agent_data['active']:
                    distance = np.sqrt((target['x'] - agent_data['x'])**2 + 
                                     (target['y'] - agent_data['y'])**2)
                    if distance <= agent_data['coverage_radius']:
                        is_covered = True
                        break
            
            if is_covered:
                covered_count += 1
            
            current_targets.append({
                'id': target['id'],
                'x': target['x'],
                'y': target['y'],
                'priority': target['priority'],
                'covered': is_covered
            })
        
        # Calculate overall coverage rate
        coverage_rate = covered_count / len(targets) if targets else 0
        
        episode_data.append({
            'timestep': timestep,
            'episode': timestep // 10,  # Group timesteps into episodes
            'step': timestep % 10,
            'agents': current_agents,
            'targets': current_targets,
            'coverage_rate': coverage_rate,
            'total_covered': covered_count
        })
    
    return episode_data


def run_movement_visualization_demo():
    """Run comprehensive movement visualization demo"""
    
    # Create timestamp for unique output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"movement_demo_{timestamp}"
    output_dir = Path("outputs") / experiment_name
    
    print(f"🎬 Creating agent movement visualization demo")
    print(f"📁 Output directory: {output_dir}")
    
    # Generate realistic movement data
    print("🚁 Generating realistic agent movement data...")
    area_size = 1000
    movement_data = generate_realistic_agent_movement_data(num_timesteps=120, area_size=area_size)
    
    print(f"   Generated {len(movement_data)} timesteps of movement data")
    print(f"   Area size: {area_size}x{area_size}m")
    print(f"   Agents: 2 satellites, 3 UAVs, 1 ground station")
    print(f"   Targets: 25 coverage targets")
    
    # Create visualizations
    print("🎨 Creating movement visualizations...")
    
    visualizer = AgentMovementVisualizer(str(output_dir / "movement_visualizations"), area_size)
    
    viz_files = []
    
    # 1. Real-time movement animation
    print("   🎬 Creating real-time movement animation...")
    movement_anim = visualizer.create_realtime_movement_animation(
        movement_data,
        save_name='realtime_agent_movements.gif'
    )
    viz_files.append(movement_anim)
    
    # 2. Coverage heatmap evolution
    print("   🌡️  Creating coverage heatmap evolution...")
    heatmap_anim = visualizer.create_coverage_heatmap_evolution(
        movement_data,
        save_name='coverage_heatmap_evolution.gif'
    )
    viz_files.append(heatmap_anim)
    
    # 3. Trajectory summary
    print("   📈 Creating trajectory summary...")
    trajectory_summary = visualizer.create_trajectory_summary(
        movement_data,
        save_name='agent_trajectory_summary.png'
    )
    viz_files.append(trajectory_summary)
    
    # Save movement data
    with open(output_dir / "movement_data.json", 'w') as f:
        json.dump(movement_data, f, indent=2, default=str)
    
    # Calculate final statistics
    final_coverage = movement_data[-1]['coverage_rate']
    avg_coverage = np.mean([step['coverage_rate'] for step in movement_data])
    max_coverage = max([step['coverage_rate'] for step in movement_data])
    
    print("\n📊 Movement Analysis Summary:")
    print(f"   • Duration: {len(movement_data)} timesteps")
    print(f"   • Final Coverage: {final_coverage:.1%}")
    print(f"   • Average Coverage: {avg_coverage:.1%}")
    print(f"   • Peak Coverage: {max_coverage:.1%}")
    print(f"   • Agent Types: Satellites (orbital), UAVs (adaptive), Ground Station (fixed)")
    
    print(f"\n📁 Visualization Files Created:")
    for i, file_path in enumerate(viz_files, 1):
        filename = os.path.basename(file_path)
        print(f"   {i}. {filename}")
    
    print(f"\n🎉 Movement visualization demo completed!")
    print(f"🔍 All files saved to: {output_dir}")
    print(f"🎬 Main animation: {output_dir}/movement_visualizations/realtime_agent_movements.gif")
    print(f"🌡️  Heatmap evolution: {output_dir}/movement_visualizations/coverage_heatmap_evolution.gif")
    
    return str(output_dir)


if __name__ == "__main__":
    try:
        output_path = run_movement_visualization_demo()
        print(f"\n✨ Success! Check the movement visualizations to see agent movements and coverage areas.")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()