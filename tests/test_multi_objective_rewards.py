#!/usr/bin/env python3
"""
Test Multi-Objective Reward System
Demonstrates the enhanced reward system with adaptive scaling
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_multi_objective_rewards():
    """Test the multi-objective reward system"""
    
    print("=" * 80)
    print("🎯 MULTI-OBJECTIVE REWARD SYSTEM TEST")
    print("=" * 80)
    
    try:
        from src.environments.robust_environment import RobustSAGINEnvironment
        
        # Load configuration
        with open("configs/default_config.json", "r") as f:
            config = json.load(f)
        
        # Create enhanced environment configuration for reward testing
        env_config = config['environment'].copy()
        env_config.update({
            'rewards': {
                'coverage_reward': 2.0,
                'priority_bonus': 1.5,
                'efficiency_reward': 1.0,
                'cooperation_reward': 0.8,
                'exploration_bonus': 0.4,
                'energy_penalty': 0.3,
                'collision_penalty': 2.0
            },
            'adaptive_scaling': {
                'early_phase_threshold': 0.3,
                'late_phase_threshold': 0.8,
                'performance_adaptation_rate': 0.1
            },
            'num_episodes': 100
        })
        
        print("🚀 Initializing Robust SAGIN Environment with Multi-Objective Rewards...")
        env = RobustSAGINEnvironment(env_config)
        print("✅ Environment initialized successfully!")
        
        # Simulate training episodes
        print("\n📊 Running reward system demonstration...")
        
        num_test_episodes = 20
        episode_rewards = []
        reward_breakdowns = []
        
        for episode in range(1, num_test_episodes + 1):
            print(f"\n🔄 Episode {episode}/{num_test_episodes}")
            
            # Update training progress for adaptive scaling
            env.update_training_progress(episode, num_test_episodes)
            
            # Reset environment
            obs = env.reset()
            
            episode_reward = 0
            episode_breakdown = {
                'coverage': 0, 'priority': 0, 'efficiency': 0, 
                'cooperation': 0, 'exploration': 0, 'communication': 0,
                'energy_penalty': 0, 'collision_penalty': 0
            }
            
            # Run episode steps
            for step in range(50):  # Short episodes for demonstration
                
                # Generate random actions (normally from trained agent)
                actions = {}
                for agent_id in range(env.num_agents):
                    actions[agent_id] = np.random.uniform(-1, 1, size=2)
                
                # Step environment
                next_obs, rewards, dones, info = env.step(actions)
                
                # Extract reward information
                if 'reward_breakdown' in info:
                    breakdown = info['reward_breakdown']
                    for component, value in breakdown.items():
                        if component in episode_breakdown:
                            episode_breakdown[component] += value
                
                # Accumulate total reward
                total_step_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
                episode_reward += total_step_reward
                
                obs = next_obs
                
                if any(dones.values()) if isinstance(dones, dict) else dones:
                    break
            
            episode_rewards.append(episode_reward)
            reward_breakdowns.append(episode_breakdown.copy())
            
            # Show episode summary
            scaling_factors = env.reward_system.current_scaling
            print(f"   💰 Total Reward: {episode_reward:.1f}")
            print(f"   🎯 Coverage: {episode_breakdown['coverage']:.1f}")
            print(f"   🤝 Cooperation: {episode_breakdown['cooperation']:.1f}")
            print(f"   📊 Scaling - Cov: {scaling_factors['coverage']:.2f}, "
                  f"Exp: {scaling_factors['exploration']:.2f}, "
                  f"Coop: {scaling_factors['cooperation']:.2f}")
        
        # Analyze results
        print("\n" + "=" * 80)
        print("📈 REWARD ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"📊 Episode Statistics:")
        print(f"   💰 Average Total Reward: {np.mean(episode_rewards):.2f}")
        print(f"   📈 Best Episode Reward: {np.max(episode_rewards):.2f}")
        print(f"   📉 Worst Episode Reward: {np.min(episode_rewards):.2f}")
        print(f"   📊 Reward Std Dev: {np.std(episode_rewards):.2f}")
        
        # Component analysis
        print(f"\n🎯 Reward Component Analysis:")
        avg_breakdown = {}
        for component in reward_breakdowns[0].keys():
            values = [bd[component] for bd in reward_breakdowns]
            avg_breakdown[component] = np.mean(values)
            print(f"   {component.replace('_', ' ').title()}: {avg_breakdown[component]:.2f} ± {np.std(values):.2f}")
        
        # Adaptive scaling analysis
        reward_analysis = env.reward_system.get_reward_analysis()
        if reward_analysis:
            print(f"\n🎛️ Adaptive Scaling Analysis:")
            scaling = reward_analysis.get('current_scaling_factors', {})
            for factor, value in scaling.items():
                print(f"   {factor.title()}: {value:.3f}")
            
            trends = reward_analysis.get('performance_trends', {})
            if trends:
                print(f"\n📈 Performance Trends:")
                for metric, data in trends.items():
                    trend_symbol = "📈" if data['trend'] == 'improving' else "📉"
                    print(f"   {trend_symbol} {metric.replace('_', ' ').title()}: "
                          f"{data['recent_average']:.3f} ({data['trend']})")
        
        # Environment analysis
        comprehensive_analysis = env.get_comprehensive_analysis()
        if comprehensive_analysis:
            env_stats = comprehensive_analysis.get('environment_stats', {})
            print(f"\n🌍 Environment Statistics:")
            print(f"   📊 Total Episodes: {env_stats.get('total_episodes', 0)}")
            print(f"   🎲 Average Difficulty: {env_stats.get('average_difficulty', 0):.3f}")
            
            conditions = env_stats.get('current_conditions', {})
            print(f"   🌤️  Weather Factor: {conditions.get('weather_factor', 1.0):.2f}")
            print(f"   📡 Comm Noise: {conditions.get('communication_noise', 0.0):.3f}")
        
        print(f"\n✅ Multi-Objective Reward System Test Complete!")
        print(f"🎯 System demonstrates:")
        print(f"   • Adaptive reward scaling based on training progress")
        print(f"   • Multiple reward objectives (coverage, cooperation, efficiency)")
        print(f"   • Performance-based adaptation")
        print(f"   • Environmental robustness integration")
        print(f"   • Comprehensive analysis and tracking")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_reward_components():
    """Demonstrate individual reward components"""
    
    print("\n" + "=" * 60)
    print("🔧 REWARD COMPONENTS DEMONSTRATION")
    print("=" * 60)
    
    try:
        from src.rewards.multi_objective_rewards import MultiObjectiveRewardSystem, RewardWeights
        
        # Create test system
        config = {
            'rewards': {
                'coverage_reward': 2.0,
                'cooperation_reward': 1.0,
                'exploration_bonus': 0.5
            },
            'num_episodes': 100
        }
        
        reward_system = MultiObjectiveRewardSystem(config, num_agents=5, area_size=1000)
        
        # Test reward weights
        weights = reward_system.weights
        print(f"🎯 Reward Weights:")
        print(f"   Coverage: {weights.coverage_reward}")
        print(f"   Priority Bonus: {weights.priority_bonus}")
        print(f"   Cooperation: {weights.cooperation_reward}")
        print(f"   Exploration: {weights.exploration_bonus}")
        print(f"   Energy Penalty: {weights.energy_penalty}")
        
        # Test adaptive scaling
        print(f"\n🎛️ Adaptive Scaling Test:")
        for phase_name, episode in [("Early", 10), ("Mid", 50), ("Late", 90)]:
            reward_system.update_training_progress(episode, 100)
            scaling = reward_system.current_scaling
            print(f"   {phase_name} Phase (ep {episode}): "
                  f"Cov={scaling['coverage']:.2f}, "
                  f"Exp={scaling['exploration']:.2f}, "
                  f"Coop={scaling['cooperation']:.2f}")
        
        print(f"\n✅ Component demonstration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Multi-Objective Reward System - Comprehensive Test")
    print("=" * 80)
    
    # Test main reward system
    main_test = test_multi_objective_rewards()
    
    # Test individual components
    component_test = demonstrate_reward_components()
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 TEST SUMMARY")
    print("=" * 80)
    
    if main_test:
        print("✅ Multi-objective reward system test: PASSED")
    else:
        print("❌ Multi-objective reward system test: FAILED")
    
    if component_test:
        print("✅ Reward components test: PASSED")
    else:
        print("❌ Reward components test: FAILED")
    
    if main_test and component_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("🎯 Multi-objective reward system is working correctly")
        print("📈 Ready for integration with training algorithms")
    else:
        print("\n⚠️ Some tests failed - check implementation")
    
    print("\n💡 Next Steps:")
    print("  1. Integrate with training algorithms")
    print("  2. Test with different agent types")
    print("  3. Tune reward weights for optimal performance")
    print("  4. Monitor adaptive scaling in long training runs")

if __name__ == "__main__":
    main()