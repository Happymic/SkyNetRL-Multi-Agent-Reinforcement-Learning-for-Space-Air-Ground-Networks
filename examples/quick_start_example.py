#!/usr/bin/env python3
"""
SkyNetRL Quick Start Example
Demonstrates the basic usage of the optimized multi-agent system

This example shows:
1. How to set up the robust SAGIN environment
2. How to initialize the multi-objective reward system
3. How to use hierarchical attention networks
4. How to enable comprehensive monitoring
5. How to run a simple training loop
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def quick_start_demo():
    """Quick start demonstration"""
    
    print("🛰️  SkyNetRL Quick Start Example")
    print("=" * 60)
    
    # Step 1: Basic Configuration
    print("⚙️  Step 1: Setting up configuration")
    
    config = {
        'environment': {
            'area_size': 1000,
            'max_episode_steps': 100,  # Short episodes for demo
            'num_satellites': 2,
            'num_uavs': 2,
            'num_ground_stations': 1,
            'num_pois': 8,
            'randomization': {
                'agent_position_noise': 50.0,
                'weather_variation': True,
                'dynamic_obstacles': True
            }
        },
        'rewards': {
            'coverage_reward': 2.0,
            'cooperation_reward': 1.0,
            'efficiency_reward': 0.8
        },
        'monitoring': {
            'log_interval': 5,
            'experiment_name': 'quick_start_demo',
            'save_plots': False  # Disabled for quick demo
        }
    }
    
    print("✅ Configuration ready")
    
    # Step 2: Initialize Environment
    print("\n🌍 Step 2: Initializing robust environment")
    
    try:
        from environments.robust_environment import RobustSAGINEnvironment
        env = RobustSAGINEnvironment(config['environment'])
        print(f"✅ Environment created with {env.num_agents} agents")
        print(f"   📊 Multi-objective reward system integrated")
        
    except ImportError:
        print("⚠️  Using basic environment (optimizations not available)")
        try:
            from environments.enhanced_sagin_env import EnhancedSAGINEnvironment
            env = EnhancedSAGINEnvironment(config['environment'])
        except ImportError:
            print("❌ Cannot find environment modules. Please check src/ directory structure")
            return
    
    # Step 3: Initialize Monitoring
    print("\n📊 Step 3: Setting up monitoring")
    
    try:
        from monitoring.comprehensive_monitor import create_comprehensive_monitor
        monitor = create_comprehensive_monitor(config)
        
        # Integrate with environment
        if hasattr(env, 'reward_system'):
            monitor.integrate_systems(reward_system=env.reward_system)
        
        print("✅ Comprehensive monitoring enabled")
        
    except ImportError:
        print("⚠️  Basic monitoring only")
        monitor = None
    
    # Step 4: Simple Training Loop
    print("\n🚀 Step 4: Running demonstration episodes")
    
    num_demo_episodes = 5
    
    for episode in range(1, num_demo_episodes + 1):
        print(f"\n📊 Episode {episode}/{num_demo_episodes}")
        
        # Reset environment
        obs = env.reset()
        
        episode_reward = 0
        step_count = 0
        
        # Run episode
        for step in range(config['environment']['max_episode_steps']):
            
            # Get actions (random for demo)
            actions = {}
            for agent_id in range(env.num_agents):
                actions[agent_id] = env.action_space.sample()
            
            # Step environment
            next_obs, rewards, dones, info = env.step(actions)
            
            # Track reward
            total_step_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
            episode_reward += total_step_reward
            step_count += 1
            
            obs = next_obs
            
            if any(dones.values()) if isinstance(dones, dict) else dones:
                break
        
        # Show results
        coverage_rate = info.get('coverage_rate', 0.0)
        print(f"   💰 Reward: {episode_reward:.1f}")
        print(f"   🎯 Coverage: {coverage_rate:.1f}%")
        print(f"   📏 Steps: {step_count}")
        
        # Show reward breakdown (if available)
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            print(f"   📋 Breakdown: " + 
                  " | ".join([f"{k}: {v:.1f}" for k, v in breakdown.items()]))
        
        # Record in monitoring system
        if monitor:
            training_info = {
                'total_loss': 0.0,  # Placeholder for demo
                'gradient_norm': 1.0,
                'learning_rate': 0.001,
                'energy_efficiency': episode_reward / step_count
            }
            
            monitor.record_episode(
                episode=episode,
                total_reward=episode_reward,
                environment_info=info,
                training_info=training_info
            )
    
    # Step 5: Show Results
    print(f"\n📈 Step 5: Analysis Results")
    
    if monitor:
        analysis = monitor.get_comprehensive_analysis()
        
        if 'training_summary' in analysis:
            summary = analysis['training_summary']
            print(f"✅ Completed {summary['total_episodes']} episodes")
            print(f"🏆 Best reward: {summary['best_performance']['reward']:.1f}")
            print(f"📊 Average performance: {summary['current_performance']['reward_mean']:.1f}")
        
        # Finalize monitoring
        monitor.finalize()
        print("📁 Detailed results saved to outputs directory")
    
    print(f"\n🎉 Quick start demo completed successfully!")
    print(f"💡 Next steps:")
    print(f"   • Run full training with: python train.py --algorithm ae_maddpg")
    print(f"   • Enable video generation with: --video flag")
    print(f"   • Customize configuration in configs/default_config.json")

def show_system_info():
    """Show information about the SkyNetRL system"""
    
    print("\n📋 SkyNetRL System Information")
    print("-" * 40)
    
    components = [
        ("🌍 Robust SAGIN Environment", "Intelligent randomization and robustness"),
        ("🎯 Multi-Objective Rewards", "Adaptive scaling and multiple objectives"),
        ("🧠 Hierarchical Attention", "Spatial, agent, and task attention"),
        ("🛡️ Training Stability", "Gradient clipping and adaptive scheduling"),
        ("📊 Comprehensive Monitoring", "Real-time analysis and reporting")
    ]
    
    print("Available Optimization Components:")
    for component, description in components:
        print(f"  {component}: {description}")
    
    print(f"\n🤖 Supported Algorithms:")
    algorithms = [
        "ae_maddpg - Attention-Enhanced MADDPG",
        "baseline_maddpg - Standard MADDPG",
        "qmix - Q-Mix Multi-Agent Q-Learning",
        "independent_ppo - Independent PPO agents",
        "greedy_heuristic - Heuristic baseline"
    ]
    
    for alg in algorithms:
        print(f"  • {alg}")

if __name__ == "__main__":
    try:
        quick_start_demo()
        show_system_info()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")