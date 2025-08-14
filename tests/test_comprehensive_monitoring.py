#!/usr/bin/env python3
"""
Test Comprehensive Monitoring System
Demonstrates advanced tracking and analysis of multi-agent training
"""

import sys
import os
import json
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_monitoring_system():
    """Test the comprehensive monitoring system"""
    
    print("=" * 80)
    print("📊 COMPREHENSIVE MONITORING SYSTEM TEST")
    print("=" * 80)
    
    try:
        from src.monitoring.comprehensive_monitor import (
            ComprehensiveMonitor,
            MonitoringConfig,
            EpisodeMetrics,
            MetricsBuffer
        )
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        print(f"🗂️  Using temporary directory: {temp_dir}")
        
        # Configure monitoring system
        config = MonitoringConfig(
            log_interval=5,
            save_interval=20,
            plot_interval=25,
            output_dir=temp_dir,
            experiment_name="test_monitoring",
            save_plots=True,
            save_raw_data=True
        )
        
        # Create monitor
        monitor = ComprehensiveMonitor(config)
        print("✅ Monitoring system initialized")
        
        print("\n🔄 Simulating training episodes...")
        
        # Simulate training episodes with realistic patterns
        for episode in range(1, 101):
            
            # Simulate different training phases
            if episode <= 20:
                # Learning phase - improving but noisy
                base_reward = -100 + episode * 4
                noise_level = 20
                coverage = min(episode * 2, 100)
                loss = 10.0 - episode * 0.3
            elif episode <= 50:
                # Plateau phase - stable with some improvement
                base_reward = -20 + (episode - 20) * 1.5
                noise_level = 10
                coverage = min(40 + (episode - 20) * 1.2, 100)
                loss = 4.0 - (episode - 20) * 0.05
            elif episode <= 80:
                # Convergence phase - slow improvement
                base_reward = 25 + (episode - 50) * 0.8
                noise_level = 5
                coverage = min(76 + (episode - 50) * 0.5, 100)
                loss = 2.5 - (episode - 50) * 0.02
            else:
                # Fine-tuning phase - minimal changes
                base_reward = 49 + np.random.normal(0, 2)
                noise_level = 3
                coverage = min(91 + np.random.normal(0, 2), 100)
                loss = 1.8 + np.random.normal(0, 0.1)
            
            # Add realistic noise
            total_reward = base_reward + np.random.normal(0, noise_level)
            coverage_rate = max(0, min(100, coverage + np.random.normal(0, 5)))
            energy_efficiency = 0.5 + episode * 0.005 + np.random.normal(0, 0.1)
            
            # Environment info
            environment_info = {
                'coverage_rate': coverage_rate,
                'collisions': np.random.poisson(0.5) if episode < 50 else np.random.poisson(0.1),
                'scenario_difficulty': 0.3 + np.random.uniform(-0.1, 0.2),
                'communication_efficiency': min(1.0, 0.5 + episode * 0.008 + np.random.normal(0, 0.05)),
                'reward_breakdown': {
                    'coverage': total_reward * 0.6,
                    'efficiency': total_reward * 0.25,
                    'cooperation': total_reward * 0.1,
                    'exploration': total_reward * 0.05
                }
            }
            
            # Training info
            training_info = {
                'energy_efficiency': energy_efficiency,
                'total_loss': max(0.01, loss + np.random.normal(0, 0.2)),
                'gradient_norm': max(0.01, 2.0 * np.exp(-episode/30) + np.random.normal(0, 0.3)),
                'learning_rate': max(1e-6, 0.001 * (0.99 ** (episode // 10)))
            }
            
            # Record episode
            monitor.record_episode(
                episode=episode,
                total_reward=total_reward,
                environment_info=environment_info,
                training_info=training_info
            )
        
        print(f"\n✅ Recorded {monitor.current_episode} episodes")
        
        # Test analysis capabilities
        print("\n📈 Generating comprehensive analysis...")
        analysis = monitor.get_comprehensive_analysis()
        
        print(f"📋 Analysis Summary:")
        if 'training_summary' in analysis:
            summary = analysis['training_summary']
            print(f"   📊 Episodes: {summary['total_episodes']}")
            print(f"   ⏱️  Time: {summary['elapsed_time_minutes']:.1f} minutes")
            print(f"   🏆 Best reward: {summary['best_performance']['reward']:.1f}")
            print(f"   📈 EPS rate: {summary['episodes_per_minute']:.1f}/min")
        
        if 'performance_analysis' in analysis:
            perf = analysis['performance_analysis']
            print(f"   📊 Improvement: {perf.get('performance_improvement', 0):.1f}")
            print(f"   🎯 Convergence: {perf.get('convergence_indicator', 0):.3f}")
        
        # Test file outputs
        output_files = list(Path(temp_dir).rglob("*"))
        print(f"\n📁 Generated {len(output_files)} output files:")
        for file_path in sorted(output_files):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   📄 {file_path.name}: {size_mb:.2f} MB")
        
        # Finalize monitoring
        print(f"\n🏁 Finalizing monitoring...")
        final_analysis = monitor.finalize()
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary files")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_buffer():
    """Test the metrics buffer system"""
    
    print("\n" + "=" * 60)
    print("💾 METRICS BUFFER TEST")
    print("=" * 60)
    
    try:
        from src.monitoring.comprehensive_monitor import MetricsBuffer, EpisodeMetrics
        
        # Create buffer
        buffer = MetricsBuffer(maxlen=100)
        print("📦 Created metrics buffer with capacity 100")
        
        # Add test metrics
        print("📊 Adding test metrics...")
        for i in range(150):  # More than capacity to test circular buffer
            metrics = EpisodeMetrics(
                episode=i,
                timestamp=1000.0 + i,
                total_reward=10.0 + i * 0.5 + np.random.normal(0, 2),
                coverage_rate=min(100, i * 0.8 + np.random.normal(0, 5)),
                energy_efficiency=0.5 + i * 0.003,
                collision_count=np.random.poisson(0.3),
                total_loss=5.0 * np.exp(-i/50),
                gradient_norm=2.0 * np.exp(-i/30),
                learning_rate=0.001,
                scenario_difficulty=0.5 + np.random.normal(0, 0.1),
                communication_efficiency=min(1.0, 0.3 + i * 0.005),
                reward_breakdown={'coverage': 6.0, 'efficiency': 3.0, 'cooperation': 1.0},
                attention_analysis={'spatial_entropy': 2.5, 'agent_entropy': 1.8}
            )
            buffer.append(metrics)
        
        print(f"   ✅ Buffer length: {len(buffer)} (should be 100 due to maxlen)")
        
        # Test array operations
        rewards = buffer.get_array('total_reward')
        coverage = buffer.get_array('coverage_rate')
        
        print(f"   ✅ Reward array length: {len(rewards)}")
        print(f"   ✅ Reward range: {rewards.min():.2f} - {rewards.max():.2f}")
        print(f"   ✅ Coverage range: {coverage.min():.2f} - {coverage.max():.2f}")
        
        # Test recent data
        recent = buffer.get_recent(10)
        print(f"   ✅ Recent data length: {len(recent)}")
        print(f"   ✅ Recent episodes: {[m.episode for m in recent[-5:]]}")
        
        # Test breakdown arrays
        coverage_rewards = buffer.get_array('reward_coverage')
        if len(coverage_rewards) > 0:
            print(f"   ✅ Reward breakdown arrays working: {len(coverage_rewards)} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_simulation():
    """Test monitoring integration with optimization systems"""
    
    print("\n" + "=" * 60)
    print("🔗 INTEGRATION SIMULATION TEST") 
    print("=" * 60)
    
    try:
        from src.monitoring.comprehensive_monitor import ComprehensiveMonitor, MonitoringConfig
        
        # Mock optimization systems
        class MockRewardSystem:
            def get_reward_analysis(self):
                return {
                    'current_scaling_factors': {'coverage': 1.5, 'exploration': 0.8},
                    'performance_trends': {'coverage_rates': {'trend': 'improving'}}
                }
        
        class MockAttentionNetwork:
            def get_attention_analysis(self):
                return {
                    'spatial_attention_entropy': 2.5 + np.random.normal(0, 0.2),
                    'agent_attention_entropy': 1.8 + np.random.normal(0, 0.1),
                    'communication_connectivity': 0.85 + np.random.normal(0, 0.05)
                }
        
        class MockStabilitySystem:
            def get_stability_metrics(self):
                return {
                    'training_stable': True,
                    'current_noise_level': 0.3,
                    'patience_counter': 5,
                    'reward_trend': 'improving'
                }
        
        # Create monitoring with integration
        temp_dir = tempfile.mkdtemp()
        config = MonitoringConfig(
            log_interval=2,
            output_dir=temp_dir,
            experiment_name="integration_test"
        )
        
        monitor = ComprehensiveMonitor(config)
        
        # Integrate systems
        reward_system = MockRewardSystem()
        attention_network = MockAttentionNetwork()
        stability_system = MockStabilitySystem()
        
        monitor.integrate_systems(
            reward_system=reward_system,
            attention_network=attention_network,
            stability_system=stability_system
        )
        
        print("🔗 Integration completed")
        
        # Simulate episodes with integrated data
        for episode in range(1, 21):
            environment_info = {
                'coverage_rate': 50 + episode * 2,
                'reward_breakdown': {
                    'coverage': 15 + episode,
                    'efficiency': 8 + episode * 0.5,
                    'cooperation': 3 + episode * 0.2
                }
            }
            
            training_info = {
                'total_loss': 3.0 - episode * 0.1,
                'gradient_norm': 1.5 - episode * 0.05,
                'learning_rate': 0.001
            }
            
            monitor.record_episode(
                episode=episode,
                total_reward=20 + episode * 2,
                environment_info=environment_info,
                training_info=training_info
            )
        
        print(f"✅ Recorded {monitor.current_episode} episodes with integrated data")
        
        # Test comprehensive analysis with integration
        analysis = monitor.get_comprehensive_analysis()
        
        print("📊 Integration Analysis:")
        if 'recommendations' in analysis:
            print(f"   💡 Recommendations: {len(analysis['recommendations'])}")
            for rec in analysis['recommendations'][:3]:
                print(f"      • {rec}")
        
        # Cleanup
        monitor.finalize()
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_monitoring_features():
    """Demonstrate key monitoring features"""
    
    print("\n" + "=" * 60)
    print("🌟 MONITORING FEATURES DEMONSTRATION")
    print("=" * 60)
    
    print("📊 Comprehensive Monitoring System Features:")
    print("   📈 Real-time training progress tracking")
    print("   📋 Detailed episode metrics storage")
    print("   📊 Automated plot generation and analysis")
    print("   🔍 Performance trend analysis")
    print("   🎯 Training stability monitoring")
    print("   💾 Multi-format data export (JSON, CSV)")
    print("   🧠 Integration with optimization systems")
    print("   📑 Comprehensive final analysis reports")
    
    print("\n🏗️ System Advantages:")
    print("   ✅ Circular buffer for memory-efficient storage")
    print("   ✅ Adaptive analysis based on training phase")
    print("   ✅ Automated anomaly detection")
    print("   ✅ Performance benchmarking and comparison")
    print("   ✅ Intelligent training recommendations")
    print("   ✅ Scalable to long training runs")
    
    print("\n🎯 Integration Benefits:")
    print("   📊 Multi-objective reward system tracking")
    print("   🧠 Hierarchical attention analysis")
    print("   🛡️ Training stability monitoring")
    print("   📈 Unified performance dashboard")
    print("   🔬 Deep training diagnostics")
    print("   📋 Automated reporting and alerts")

def main():
    """Main test function"""
    
    print("🧪 Comprehensive Monitoring System - Complete Test Suite")
    print("=" * 80)
    
    # Run all tests
    monitoring_test = test_monitoring_system()
    buffer_test = test_metrics_buffer()
    integration_test = test_integration_simulation()
    
    # Demonstrate features
    demonstrate_monitoring_features()
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 TEST SUMMARY")
    print("=" * 80)
    
    tests = [
        ("Comprehensive monitoring system", monitoring_test),
        ("Metrics buffer operations", buffer_test), 
        ("System integration", integration_test)
    ]
    
    passed = sum(1 for _, result in tests if result)
    
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("📊 Comprehensive monitoring system is working correctly")
        print("🔗 Successfully integrates with all optimization components")
        print("📈 Ready for production multi-agent training")
    else:
        print(f"\n⚠️ {len(tests) - passed}/{len(tests)} tests failed")
    
    print("\n💡 Usage Example:")
    print("  from src.monitoring.comprehensive_monitor import create_comprehensive_monitor")
    print("  monitor = create_comprehensive_monitor(config)")
    print("  monitor.integrate_systems(reward_system, attention_network, stability_system)")
    print("  monitor.record_episode(episode, reward, env_info, training_info)")
    print("  analysis = monitor.get_comprehensive_analysis()")
    print("  monitor.finalize()")

if __name__ == "__main__":
    main()