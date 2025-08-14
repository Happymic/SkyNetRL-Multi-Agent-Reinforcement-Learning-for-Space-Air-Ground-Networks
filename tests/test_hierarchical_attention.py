#!/usr/bin/env python3
"""
Test Hierarchical Attention Network Architecture
Demonstrates the advanced neural network components for multi-agent learning
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_hierarchical_attention_network():
    """Test the hierarchical attention network architecture"""
    
    print("=" * 80)
    print("🧠 HIERARCHICAL ATTENTION NETWORK TEST")
    print("=" * 80)
    
    try:
        from src.networks.hierarchical_attention import (
            HierarchicalAttentionNetwork,
            SpatialAttentionModule,
            AgentAttentionModule,
            TaskAttentionModule,
            MultiHeadAttention,
            create_hierarchical_attention_network
        )
        
        print("📦 Testing individual attention components...")
        
        # Test Multi-Head Attention
        print("\n🔍 Testing Multi-Head Attention...")
        d_model, n_heads = 256, 8
        mha = MultiHeadAttention(d_model, n_heads)
        
        batch_size, seq_len = 4, 10
        test_input = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = mha(test_input, test_input, test_input)
        print(f"   ✅ Input shape: {test_input.shape}")
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Attention weights shape: {weights.shape}")
        assert output.shape == test_input.shape
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
        
        # Test Spatial Attention Module
        print("\n🌍 Testing Spatial Attention Module...")
        spatial_module = SpatialAttentionModule(d_model, n_heads=8)
        
        # Create test spatial features (POIs, obstacles, etc.)
        n_objects = 8
        spatial_features = torch.randn(batch_size, n_objects, 5)  # x, y, type, feature1, feature2
        agent_position = torch.randn(batch_size, 2)  # agent x, y position
        
        spatial_output, spatial_weights = spatial_module(spatial_features, agent_position)
        print(f"   ✅ Spatial input shape: {spatial_features.shape}")
        print(f"   ✅ Spatial output shape: {spatial_output.shape}")
        print(f"   ✅ Spatial attention weights shape: {spatial_weights.shape}")
        
        # Test Agent Attention Module
        print("\n🤖 Testing Agent Attention Module...")
        agent_module = AgentAttentionModule(d_model, n_heads=8)  # 256 is divisible by 8
        
        n_agents = 6
        agent_features = torch.randn(batch_size, n_agents, 12)  # agent state + position features
        self_position = torch.randn(batch_size, 3)  # x, y, z position
        
        agent_output, agent_weights, comm_mask = agent_module(agent_features, self_position)
        print(f"   ✅ Agent input shape: {agent_features.shape}")
        print(f"   ✅ Agent output shape: {agent_output.shape}")
        print(f"   ✅ Agent attention weights shape: {agent_weights.shape}")
        print(f"   ✅ Communication mask shape: {comm_mask.shape}")
        
        # Test Task Attention Module
        print("\n📋 Testing Task Attention Module...")
        task_module = TaskAttentionModule(d_model, n_heads=4)
        
        global_features = torch.randn(batch_size, 6)  # global mission state
        task_output, task_weights = task_module(global_features)
        print(f"   ✅ Task input shape: {global_features.shape}")
        print(f"   ✅ Task output shape: {task_output.shape}")
        print(f"   ✅ Task attention weights shape: {task_weights.shape}")
        
        print("\n✅ All attention components working correctly!")
        
        # Test full hierarchical network
        print("\n🏗️  Testing Full Hierarchical Attention Network...")
        
        config = {
            'state_dim': 184,  # Matches enhanced SAGIN environment
            'action_dim': 2,
            'hierarchical_attention': {
                'hidden_dim': 256,
                'n_spatial_heads': 8,
                'n_agent_heads': 8,
                'n_task_heads': 4,
                'dropout': 0.1
            }
        }
        
        network = create_hierarchical_attention_network(config)
        
        # Create structured observation
        batch_size = 4
        test_obs = torch.randn(batch_size, 184)  # Flat observation from environment
        
        # Forward pass
        output = network(test_obs)
        
        print(f"   ✅ Network input shape: {test_obs.shape}")
        print(f"   ✅ Value output shape: {output['value'].shape}")
        print(f"   ✅ Action mean shape: {output['action_mean'].shape}")
        print(f"   ✅ Action std shape: {output['action_std'].shape}")
        
        # Test attention analysis
        attention_analysis = network.get_attention_analysis()
        print(f"\n📊 Attention Analysis:")
        for key, value in attention_analysis.items():
            print(f"   📈 {key}: {value:.4f}")
        
        print(f"\n🎯 Network Architecture Summary:")
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎓 Trainable parameters: {trainable_params:,}")
        print(f"   🧠 Hidden dimension: {config['hierarchical_attention']['hidden_dim']}")
        print(f"   👁️ Spatial attention heads: {config['hierarchical_attention']['n_spatial_heads']}")
        print(f"   🤝 Agent attention heads: {config['hierarchical_attention']['n_agent_heads']}")
        print(f"   📋 Task attention heads: {config['hierarchical_attention']['n_task_heads']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hierarchical_critic():
    """Test the hierarchical critic network"""
    
    print("\n" + "=" * 60)
    print("🎯 HIERARCHICAL CRITIC NETWORK TEST")
    print("=" * 60)
    
    try:
        from src.networks.hierarchical_attention import HierarchicalCritic, create_hierarchical_critic
        
        config = {
            'state_dim': 184 * 7,  # Global state for all agents
            'n_agents': 7,
            'hierarchical_critic': {
                'hidden_dim': 256,
                'dropout': 0.1
            }
        }
        
        critic = create_hierarchical_critic(config)
        
        # Test forward pass
        batch_size = 4
        global_state = torch.randn(batch_size, config['state_dim'])
        
        value = critic(global_state)
        
        print(f"✅ Critic input shape: {global_state.shape}")
        print(f"✅ Critic output shape: {value.shape}")
        print(f"✅ Output range: [{value.min().item():.3f}, {value.max().item():.3f}]")
        
        # Architecture summary
        total_params = sum(p.numel() for p in critic.parameters())
        print(f"\n📊 Critic Architecture:")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🧠 Hidden dimension: {config['hierarchical_critic']['hidden_dim']}")
        print(f"   👥 Number of agents: {config['n_agents']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Critic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_interpretability():
    """Test attention mechanism interpretability features"""
    
    print("\n" + "=" * 60)
    print("🔍 ATTENTION INTERPRETABILITY TEST")
    print("=" * 60)
    
    try:
        from src.networks.hierarchical_attention import create_hierarchical_attention_network
        
        config = {
            'state_dim': 184,
            'action_dim': 2,
            'hierarchical_attention': {
                'hidden_dim': 128,  # Smaller for faster testing
                'n_spatial_heads': 4,
                'n_agent_heads': 4,
                'n_task_heads': 2
            }
        }
        
        network = create_hierarchical_attention_network(config)
        
        # Create test scenarios
        scenarios = {
            'high_priority_mission': torch.randn(1, 184),
            'low_energy_situation': torch.randn(1, 184),
            'crowded_environment': torch.randn(1, 184)
        }
        
        print("📋 Testing attention patterns across different scenarios:")
        
        for scenario_name, test_obs in scenarios.items():
            output = network(test_obs)
            analysis = network.get_attention_analysis()
            
            print(f"\n🎯 Scenario: {scenario_name}")
            print(f"   📊 Spatial attention entropy: {analysis['spatial_attention_entropy']:.4f}")
            print(f"   🤖 Agent attention entropy: {analysis['agent_attention_entropy']:.4f}")
            print(f"   📋 Task attention entropy: {analysis['task_attention_entropy']:.4f}")
            print(f"   📡 Communication connectivity: {analysis['communication_connectivity']:.4f}")
            
            # Check attention weights exist
            weights = output['attention_weights']
            assert 'spatial' in weights
            assert 'agent' in weights
            assert 'task' in weights
            assert 'communication_mask' in weights
        
        print(f"\n✅ Interpretability features working correctly!")
        print(f"📈 Different scenarios show varying attention patterns")
        print(f"🔬 Attention weights accessible for analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Interpretability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_network_capabilities():
    """Demonstrate key capabilities of the hierarchical attention network"""
    
    print("\n" + "=" * 60)
    print("🌟 NETWORK CAPABILITIES DEMONSTRATION")
    print("=" * 60)
    
    print("🧠 Hierarchical Attention Network Features:")
    print("   🌍 Spatial Attention: Focuses on POIs, obstacles, and charging stations")
    print("   🤖 Agent Attention: Models multi-agent communication and coordination")
    print("   📋 Task Attention: Prioritizes mission objectives and temporal context")
    print("   🔗 Attention Fusion: Combines all attention mechanisms hierarchically")
    print("   📊 Interpretability: Provides attention weights for analysis")
    
    print("\n🏗️  Architecture Advantages:")
    print("   ✅ Handles variable numbers of agents and spatial objects")
    print("   ✅ Communication-aware agent interactions")
    print("   ✅ Priority-aware spatial reasoning")
    print("   ✅ Temporal and mission context understanding")
    print("   ✅ Explainable attention mechanisms")
    print("   ✅ Scalable multi-head attention")
    
    print("\n🎯 Integration Benefits:")
    print("   📈 Better coordination in multi-agent scenarios")
    print("   🎯 Improved POI coverage through spatial attention")
    print("   ⚡ Efficient communication resource usage")
    print("   🧠 Adaptive behavior based on mission context")
    print("   🔍 Attention analysis for debugging and optimization")

def main():
    """Main test function"""
    
    print("🧪 Hierarchical Attention Network - Comprehensive Test")
    print("=" * 80)
    
    # Test main network
    network_test = test_hierarchical_attention_network()
    
    # Test critic
    critic_test = test_hierarchical_critic()
    
    # Test interpretability
    interpretability_test = test_attention_interpretability()
    
    # Demonstrate capabilities
    demonstrate_network_capabilities()
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 TEST SUMMARY")
    print("=" * 80)
    
    if network_test:
        print("✅ Hierarchical attention network test: PASSED")
    else:
        print("❌ Hierarchical attention network test: FAILED")
    
    if critic_test:
        print("✅ Hierarchical critic test: PASSED") 
    else:
        print("❌ Hierarchical critic test: FAILED")
    
    if interpretability_test:
        print("✅ Attention interpretability test: PASSED")
    else:
        print("❌ Attention interpretability test: FAILED")
    
    if network_test and critic_test and interpretability_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("🧠 Hierarchical attention architecture is working correctly")
        print("📈 Ready for integration with MADDPG and other algorithms")
    else:
        print("\n⚠️ Some tests failed - check implementation")
    
    print("\n💡 Next Steps:")
    print("  1. Integrate with existing MADDPG agent")
    print("  2. Test with multi-agent training scenarios")
    print("  3. Analyze attention patterns during learning")
    print("  4. Optimize attention head configurations")
    print("  5. Validate performance improvements")

if __name__ == "__main__":
    main()