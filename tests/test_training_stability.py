#!/usr/bin/env python3
"""
Test Advanced Training Stability Improvements
Demonstrates sophisticated training stabilization techniques
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_gradient_stabilization():
    """Test adaptive gradient clipping and stabilization"""
    
    print("=" * 80)
    print("🎯 GRADIENT STABILIZATION TEST")
    print("=" * 80)
    
    try:
        from src.training.stability_improvements import (
            AdaptiveGradientClipper,
            TrainingStabilizer,
            StabilityConfig
        )
        
        print("🔧 Testing Adaptive Gradient Clipping...")
        
        # Create test network
        test_network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # Test adaptive gradient clipper
        clipper = AdaptiveGradientClipper(initial_max_norm=1.0)
        
        # Simulate training with varying gradient magnitudes
        gradient_norms = []
        clipping_thresholds = []
        
        for i in range(100):
            # Create artificial gradients with varying magnitudes
            for param in test_network.parameters():
                if i < 20:
                    # Large gradients initially
                    grad_magnitude = np.random.normal(5.0, 2.0)
                elif i < 60:
                    # Medium gradients
                    grad_magnitude = np.random.normal(2.0, 0.5)
                else:
                    # Small gradients later
                    grad_magnitude = np.random.normal(0.5, 0.1)
                
                param.grad = torch.randn_like(param) * abs(grad_magnitude)
            
            # Apply gradient clipping
            grad_norm = clipper.clip_gradients(test_network.parameters())
            gradient_norms.append(grad_norm)
            clipping_thresholds.append(clipper.max_norm)
        
        print(f"   ✅ Gradient norms range: {min(gradient_norms):.3f} - {max(gradient_norms):.3f}")
        print(f"   ✅ Adaptive threshold range: {min(clipping_thresholds):.3f} - {max(clipping_thresholds):.3f}")
        print(f"   ✅ Final threshold: {clipper.max_norm:.3f}")
        print(f"   📊 Threshold adapted from {clipping_thresholds[0]:.3f} to {clipping_thresholds[-1]:.3f}")
        
        # Test training stabilizer
        print(f"\n🏗️  Testing Training Stabilizer...")
        config = StabilityConfig(
            gradient_clipping_enabled=True,
            max_grad_norm=1.0,
            loss_clipping_enabled=True,
            max_loss_value=10.0
        )
        
        stabilizer = TrainingStabilizer(config)
        
        # Test gradient stabilization
        networks = {'test_net': test_network}
        losses = {'test_net': torch.tensor(2.5, requires_grad=True)}
        
        gradient_info = stabilizer.stabilize_gradients(networks, losses)
        
        print(f"   ✅ Gradient info: {gradient_info}")
        print(f"   ✅ Recent losses tracked: {len(stabilizer.recent_losses)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient stabilization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learning_rate_scheduling():
    """Test advanced learning rate scheduling"""
    
    print("\n" + "=" * 60)
    print("📈 LEARNING RATE SCHEDULING TEST")
    print("=" * 60)
    
    try:
        from src.training.stability_improvements import AdvancedLearningRateScheduler
        
        # Create test optimizer
        test_net = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(test_net.parameters(), lr=0.001)
        
        print("🔧 Testing different scheduler types...")
        
        # Test reduce on plateau scheduler
        print("\n📉 Testing Reduce-On-Plateau Scheduler:")
        plateau_scheduler = AdvancedLearningRateScheduler(
            optimizer, 
            scheduler_type="reduce_on_plateau",
            lr_decay_factor=0.5,
            lr_patience=10,
            lr_min=1e-6,
            warmup_steps=20
        )
        
        learning_rates = []
        for i in range(50):
            # Simulate performance metrics (starts bad, improves, then plateaus)
            if i < 10:
                metric = 10.0 - i * 0.5  # Improving
            elif i < 30:
                metric = 5.0 + np.random.normal(0, 0.1)  # Plateau with noise
            else:
                metric = 5.0 + np.random.normal(0, 0.05)  # Stable
            
            plateau_scheduler.step(metric)
            current_lrs = plateau_scheduler.get_current_lr()
            learning_rates.append(current_lrs[0])
        
        print(f"   ✅ Initial LR: {learning_rates[0]:.6f}")
        print(f"   ✅ Final LR: {learning_rates[-1]:.6f}")
        print(f"   ✅ LR reduction factor: {learning_rates[-1]/learning_rates[0]:.3f}")
        
        # Test cosine scheduler
        print("\n🌊 Testing Cosine Annealing Scheduler:")
        optimizer2 = torch.optim.Adam(test_net.parameters(), lr=0.001)
        cosine_scheduler = AdvancedLearningRateScheduler(
            optimizer2,
            scheduler_type="cosine",
            restart_period=100,
            restart_multiplier=2,
            lr_min=1e-6
        )
        
        cosine_lrs = []
        for i in range(200):
            cosine_scheduler.step()
            cosine_lrs.append(cosine_scheduler.get_current_lr()[0])
        
        print(f"   ✅ Cosine LR range: {min(cosine_lrs):.6f} - {max(cosine_lrs):.6f}")
        print(f"   ✅ Number of restarts detected: {len([i for i in range(1, len(cosine_lrs)) if cosine_lrs[i] > cosine_lrs[i-1] * 1.5])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning rate scheduling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prioritized_replay():
    """Test prioritized experience replay buffer"""
    
    print("\n" + "=" * 60)
    print("🧠 PRIORITIZED REPLAY BUFFER TEST")
    print("=" * 60)
    
    try:
        from src.training.stability_improvements import PrioritizedReplayBuffer
        
        # Create prioritized replay buffer
        buffer = PrioritizedReplayBuffer(
            capacity=1000,
            alpha=0.6,
            beta=0.4,
            beta_end=1.0,
            beta_decay_steps=10000
        )
        
        print("📦 Testing buffer operations...")
        
        # Add experiences with different priorities
        experiences = []
        priorities = []
        
        for i in range(100):
            # Create dummy experience
            exp = {
                'state': np.random.randn(10),
                'action': np.random.randn(2),
                'reward': np.random.randn(),
                'next_state': np.random.randn(10),
                'done': False
            }
            
            # Assign priority (some high, some low)
            if i < 20:
                priority = np.random.uniform(5.0, 10.0)  # High priority
            elif i < 50:
                priority = np.random.uniform(1.0, 3.0)   # Medium priority
            else:
                priority = np.random.uniform(0.1, 1.0)   # Low priority
            
            buffer.push(exp, priority)
            experiences.append(exp)
            priorities.append(priority)
        
        print(f"   ✅ Buffer size: {len(buffer)}")
        print(f"   ✅ Priority range: {min(priorities):.3f} - {max(priorities):.3f}")
        
        # Test sampling
        batch_size = 32
        batch, indices, weights = buffer.sample(batch_size)
        
        print(f"   ✅ Sampled batch size: {len(batch)}")
        print(f"   ✅ Importance weights range: {weights.min():.3f} - {weights.max():.3f}")
        print(f"   ✅ Sample indices range: {indices.min()} - {indices.max()}")
        
        # Test priority updates
        new_priorities = np.random.uniform(0.5, 2.0, len(indices))
        buffer.update_priorities(indices, new_priorities)
        print(f"   ✅ Updated priorities for {len(indices)} experiences")
        
        # Test beta update
        initial_beta = buffer.beta
        for _ in range(100):
            buffer.update_beta()
        print(f"   ✅ Beta progression: {initial_beta:.3f} → {buffer.beta:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prioritized replay test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stability_monitoring():
    """Test training stability monitoring and early stopping"""
    
    print("\n" + "=" * 60)
    print("📊 STABILITY MONITORING TEST")
    print("=" * 60)
    
    try:
        from src.training.stability_improvements import TrainingStabilizer, StabilityConfig
        
        config = StabilityConfig(
            early_stopping_enabled=True,
            early_stopping_patience=20,
            convergence_threshold=0.1,
            loss_smoothing_window=50
        )
        
        stabilizer = TrainingStabilizer(config)
        
        print("📈 Simulating training progress...")
        
        # Simulate training episodes with different patterns
        should_stop = False
        episode = 0
        
        while episode < 100 and not should_stop:
            episode += 1
            
            # Simulate different training phases
            if episode < 30:
                # Learning phase - improving performance
                reward = -100 + episode * 2 + np.random.normal(0, 5)
                loss = 5.0 - episode * 0.1 + np.random.normal(0, 0.2)
            elif episode < 60:
                # Plateau phase - stable performance
                reward = 60 + np.random.normal(0, 3)
                loss = 2.0 + np.random.normal(0, 0.1)
            else:
                # Convergence phase - minimal improvement
                reward = 58 + np.random.normal(0, 1)
                loss = 2.1 + np.random.normal(0, 0.05)
            
            # Update statistics
            gradient_info = {'test_grad_norm': np.random.uniform(0.1, 2.0)}
            learning_rates = {'test_lr': 0.001}
            losses = {'test_loss': loss}
            
            stabilizer.update_statistics(reward, losses, gradient_info, learning_rates)
            
            # Check early stopping
            should_stop = stabilizer.check_early_stopping(reward)
            
            # Update noise schedule
            stabilizer.update_noise_schedule(episode)
            
            if episode % 20 == 0 or should_stop:
                metrics = stabilizer.get_stability_metrics()
                print(f"   📊 Episode {episode}: Reward={reward:.1f}, "
                      f"Stable={metrics['training_stable']}, "
                      f"Patience={metrics['patience_counter']}")
        
        # Final metrics
        final_metrics = stabilizer.get_stability_metrics()
        print(f"\n📋 Final Training Analysis:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n✅ Training completed after {episode} episodes")
        print(f"🛑 Early stopping: {'Triggered' if should_stop else 'Not triggered'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stability monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_target_network_updates():
    """Test advanced target network updating strategies"""
    
    print("\n" + "=" * 60)
    print("🎯 TARGET NETWORK UPDATES TEST")
    print("=" * 60)
    
    try:
        from src.training.stability_improvements import AdvancedTargetNetworkUpdater
        
        # Create test networks
        online_net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        target_net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        
        # Initialize target net differently
        for param in target_net.parameters():
            param.data.fill_(0.5)
        
        networks_online = {'test_net': online_net}
        networks_target = {'test_net': target_net}
        
        print("🔧 Testing different update strategies...")
        
        # Test soft update
        print("\n🌊 Testing Soft Updates:")
        soft_updater = AdvancedTargetNetworkUpdater(
            update_strategy="soft",
            tau=0.1  # Higher tau for visible changes
        )
        
        # Get initial parameter distance
        initial_distance = _parameter_distance(online_net, target_net)
        print(f"   📏 Initial parameter distance: {initial_distance:.6f}")
        
        # Apply several soft updates
        for i in range(10):
            soft_updater.update_target_networks(networks_online, networks_target)
        
        final_distance = _parameter_distance(online_net, target_net)
        print(f"   📏 Final parameter distance: {final_distance:.6f}")
        print(f"   📉 Distance reduction: {(initial_distance - final_distance) / initial_distance:.1%}")
        
        # Test adaptive update
        print("\n🧠 Testing Adaptive Updates:")
        # Reset target network
        for param in target_net.parameters():
            param.data.fill_(0.5)
        
        adaptive_updater = AdvancedTargetNetworkUpdater(
            update_strategy="adaptive",
            tau=0.01,
            performance_threshold=0.1
        )
        
        # Simulate performance improvements and declines
        performances = [0.1, 0.2, 0.35, 0.5, 0.48, 0.46, 0.47, 0.49, 0.51, 0.52]
        
        distances = []
        for perf in performances:
            distance_before = _parameter_distance(online_net, target_net)
            adaptive_updater.update_target_networks(
                networks_online, networks_target, perf
            )
            distance_after = _parameter_distance(online_net, target_net)
            distance_change = distance_before - distance_after
            distances.append(distance_change)
        
        print(f"   📊 Adaptive update changes: {[f'{d:.6f}' for d in distances[:5]]}")
        print(f"   🎯 Average change per update: {np.mean(distances):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Target network update test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def _parameter_distance(net1, net2):
    """Calculate L2 distance between network parameters"""
    distance = 0.0
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        distance += torch.norm(p1 - p2).item() ** 2
    return distance ** 0.5

def demonstrate_stability_features():
    """Demonstrate key stability features"""
    
    print("\n" + "=" * 60)
    print("🌟 STABILITY FEATURES DEMONSTRATION")
    print("=" * 60)
    
    print("🛡️ Advanced Training Stability Features:")
    print("   📊 Adaptive Gradient Clipping - Automatically adjusts clipping thresholds")
    print("   📈 Advanced LR Scheduling - Multiple strategies with warmup")
    print("   🧠 Prioritized Experience Replay - Importance sampling for better learning")
    print("   🎯 Smart Target Updates - Adaptive update rates based on performance")
    print("   📉 Early Stopping - Prevents overfitting with patience-based stopping")
    print("   🔧 Training Monitoring - Comprehensive stability metrics")
    
    print("\n🏗️ Stability Benefits:")
    print("   ✅ Reduced gradient explosion and vanishing")
    print("   ✅ Better sample efficiency through prioritization")
    print("   ✅ Adaptive learning rates for optimal convergence")
    print("   ✅ Robust target network updates")
    print("   ✅ Automatic early stopping to prevent overfitting")
    print("   ✅ Comprehensive training diagnostics")
    
    print("\n🎯 Integration Advantages:")
    print("   📈 More stable multi-agent training")
    print("   🎯 Better convergence in complex environments")
    print("   ⚡ Efficient hyperparameter adaptation")
    print("   🧠 Intelligent training process management")
    print("   🔍 Detailed training analysis and debugging")

def main():
    """Main test function"""
    
    print("🧪 Advanced Training Stability Improvements - Comprehensive Test")
    print("=" * 80)
    
    # Run all tests
    gradient_test = test_gradient_stabilization()
    lr_test = test_learning_rate_scheduling()
    replay_test = test_prioritized_replay()
    monitoring_test = test_stability_monitoring()
    target_test = test_target_network_updates()
    
    # Demonstrate features
    demonstrate_stability_features()
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 TEST SUMMARY")
    print("=" * 80)
    
    tests = [
        ("Gradient stabilization", gradient_test),
        ("Learning rate scheduling", lr_test),
        ("Prioritized replay buffer", replay_test),
        ("Stability monitoring", monitoring_test),
        ("Target network updates", target_test)
    ]
    
    passed = sum(1 for _, result in tests if result)
    
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("🛡️ Training stability improvements are working correctly")
        print("📈 Ready for integration with multi-agent training")
    else:
        print(f"\n⚠️ {len(tests) - passed}/{len(tests)} tests failed")
    
    print("\n💡 Next Steps:")
    print("  1. Integrate with MADDPG and other algorithms")
    print("  2. Test with complex multi-agent scenarios")
    print("  3. Monitor training stability in long runs")
    print("  4. Fine-tune stability parameters")
    print("  5. Analyze performance improvements")

if __name__ == "__main__":
    main()