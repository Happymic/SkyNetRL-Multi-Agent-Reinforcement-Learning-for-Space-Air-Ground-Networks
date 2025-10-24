#!/usr/bin/env python3
"""
Simple Training with Clean Visualization
=======================================
Creates synthetic training data and comprehensive visualizations.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from visualization.clean_training_visualizer import CleanTrainingVisualizer


def generate_synthetic_training_data(num_episodes=100):
    """Generate realistic synthetic training data"""
    
    # Initialize tracking variables
    episode_rewards = []
    coverage_rates = []
    energy_efficiency = []
    actor_losses = []
    critic_losses = []
    exploration_rates = []
    
    # Training parameters
    base_reward = -50
    convergence_episode = 60
    noise_level = 15
    
    for episode in range(num_episodes):
        # Episode reward with learning curve
        progress = min(episode / convergence_episode, 1.0)
        
        # Sigmoid-like improvement
        improvement = 1 / (1 + np.exp(-8 * (progress - 0.5)))
        reward = base_reward + 150 * improvement + np.random.normal(0, noise_level * (1 - progress * 0.7))
        episode_rewards.append(reward)
        
        # Coverage rate improvement
        coverage = 0.2 + 0.7 * improvement + np.random.normal(0, 0.05 * (1 - progress * 0.5))
        coverage = np.clip(coverage, 0, 1)
        coverage_rates.append(coverage)
        
        # Energy efficiency
        efficiency = 0.3 + 0.5 * improvement + np.random.normal(0, 0.03 * (1 - progress * 0.6))
        efficiency = np.clip(efficiency, 0, 1)
        energy_efficiency.append(efficiency)
        
        # Actor loss (decreasing)
        actor_loss = 2.0 * np.exp(-episode / 30) + np.random.normal(0, 0.1)
        actor_loss = max(actor_loss, 0.01)
        actor_losses.append(actor_loss)
        
        # Critic loss (decreasing)
        critic_loss = 1.5 * np.exp(-episode / 25) + np.random.normal(0, 0.08)
        critic_loss = max(critic_loss, 0.005)
        critic_losses.append(critic_loss)
        
        # Exploration rate (decreasing)
        exploration = 1.0 * np.exp(-episode / 40) + 0.05
        exploration_rates.append(exploration)
    
    # Create training statistics
    training_stats = {
        'episode_rewards': episode_rewards,
        'coverage_rates': coverage_rates,
        'energy_efficiency': energy_efficiency,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'exploration_rates': exploration_rates
    }
    
    # Create evaluation statistics
    final_episodes = episode_rewards[-10:]
    evaluation_stats = {
        'final_metrics': {
            'avg_coverage_rate': np.mean(coverage_rates[-10:]),
            'avg_energy_efficiency': np.mean(energy_efficiency[-10:]),
            'avg_episode_reward': np.mean(final_episodes),
            'std_episode_reward': np.std(final_episodes)
        },
        'final_episode_rewards': final_episodes,
        'stability_score': 1.0 - np.std(coverage_rates[-20:]) / np.mean(coverage_rates[-20:]),
        'convergence_speed': convergence_episode / num_episodes,
        'safety_score': 0.95  # Synthetic safety score
    }
    
    # Create episode data for animation
    episode_data = []
    for i in range(num_episodes):
        episode_data.append({
            'episode': i,
            'reward': episode_rewards[i],
            'coverage_rate': coverage_rates[i],
            'energy_efficiency': energy_efficiency[i],
            'actor_loss': actor_losses[i]
        })
    
    # Performance data for 3D visualization
    performance_data = {
        'episodes': list(range(num_episodes)),
        'coverage_rates': coverage_rates,
        'energy_efficiency': energy_efficiency,
        'episode_rewards': episode_rewards
    }
    
    return {
        'training_stats': training_stats,
        'evaluation_stats': evaluation_stats,
        'episode_data': episode_data,
        'performance_data': performance_data,
        'convergence_episode': convergence_episode,
        'config': {
            'algorithm': 'AE-MADDPG',
            'num_episodes': num_episodes,
            'environment': {
                'num_agents': 6,
                'area_size': 1000,
                'num_satellites': 2,
                'num_uavs': 3,
                'num_ground_stations': 1
            }
        }
    }


def run_visualization_demo():
    """Run comprehensive visualization demo"""
    
    # Create timestamp for unique output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"demo_training_{timestamp}"
    output_dir = Path("outputs") / experiment_name
    
    print(f"🎨 Creating comprehensive training visualization demo")
    print(f"📁 Output directory: {output_dir}")
    
    # Generate synthetic training data
    print("📊 Generating realistic training data...")
    training_data = generate_synthetic_training_data(num_episodes=100)
    
    # Add training time
    training_data['training_time'] = "45.7 seconds"
    
    # Create visualizations
    print("🎨 Creating comprehensive visualizations...")
    
    visualizer = CleanTrainingVisualizer(str(output_dir / "visualizations"))
    
    # Generate all visualizations
    viz_files = []
    
    # 1. Main comprehensive dashboard
    print("   📈 Creating comprehensive training dashboard...")
    dashboard_file = visualizer.create_comprehensive_training_dashboard(
        training_data,
        save_name='comprehensive_training_dashboard.png'
    )
    viz_files.append(dashboard_file)
    
    # 2. 3D performance landscape
    print("   🌍 Creating 3D performance landscape...")
    performance_3d = visualizer.create_3d_performance_landscape(
        training_data['performance_data'],
        save_name='3d_performance_landscape.png'
    )
    viz_files.append(performance_3d)
    
    # 3. Training animation
    print("   🎬 Creating training progress animation...")
    animation_file = visualizer.create_training_animation(
        training_data['episode_data'],
        save_name='training_progress_animation.gif'
    )
    viz_files.append(animation_file)
    
    # 4. Generate comprehensive report
    print("   📋 Generating comprehensive report...")
    report_file = visualizer.generate_training_report(
        training_data,
        save_name='comprehensive_training_report.json'
    )
    viz_files.append(report_file)
    
    # Save all training data
    with open(output_dir / "training_data.json", 'w') as f:
        json.dump(training_data, f, indent=2, default=str)
    
    print("\n📊 Training Results Summary:")
    print(f"   • Algorithm: AE-MADDPG with Multi-Head Attention")
    print(f"   • Episodes: {training_data['config']['num_episodes']}")
    print(f"   • Final Coverage: {training_data['evaluation_stats']['final_metrics']['avg_coverage_rate']:.3f}")
    print(f"   • Energy Efficiency: {training_data['evaluation_stats']['final_metrics']['avg_energy_efficiency']:.3f}")
    print(f"   • Final Avg Reward: {training_data['evaluation_stats']['final_metrics']['avg_episode_reward']:.2f}")
    print(f"   • Convergence Episode: {training_data['convergence_episode']}")
    print(f"   • Stability Score: {training_data['evaluation_stats']['stability_score']:.3f}")
    
    print(f"\n📁 Files Created:")
    for i, file_path in enumerate(viz_files, 1):
        filename = os.path.basename(file_path)
        print(f"   {i}. {filename}")
    
    print(f"\n🎉 Comprehensive visualization demo completed!")
    print(f"🔍 All files saved to: {output_dir}")
    print(f"🖼️  Main dashboard: {output_dir}/visualizations/comprehensive_training_dashboard.png")
    
    return str(output_dir)


if __name__ == "__main__":
    try:
        output_path = run_visualization_demo()
        print(f"\n✨ Success! Check the output directory for all visualization files.")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()