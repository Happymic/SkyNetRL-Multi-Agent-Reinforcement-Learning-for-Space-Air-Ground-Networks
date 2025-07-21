"""
Research Training Script for SkyNetRL
Generates comprehensive results, visualizations, and analysis for paper/research purposes
"""

from research_config import ResearchConfig
from trainer import MADDPGTrainer
import torch
import numpy as np
import random
import time
import os
import json


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config_summary(config):
    """Save configuration summary to results directory"""
    config_summary = {
        "experiment_info": {
            "timestamp": config.timestamp,
            "device": str(config.device),
            "total_episodes": config.num_episodes,
            "max_steps_per_episode": config.max_time_steps
        },
        "environment": {
            "area_size": config.area_size,
            "num_satellites": config.num_satellites,
            "num_uavs": config.num_uavs,
            "num_ground_stations": config.num_ground_stations,
            "num_pois": config.num_pois,
            "num_obstacles": config.num_obstacles,
            "num_charging_stations": config.num_charging_stations
        },
        "agent_parameters": {
            "satellite_range": config.satellite_range,
            "uav_range": config.uav_range,
            "ground_station_range": config.ground_station_range,
            "satellite_speed": config.satellite_speed,
            "uav_speed": config.uav_speed,
            "ground_station_speed": config.ground_station_speed
        },
        "training_parameters": {
            "actor_lr": config.actor_lr,
            "critic_lr": config.critic_lr,
            "gamma": config.gamma,
            "tau": config.tau,
            "batch_size": config.batch_size,
            "buffer_size": config.buffer_size
        },
        "visualization": {
            "enabled": config.real_time_visualization,
            "animation_saving": config.save_animation,
            "visualization_frequency": config.visualize_frequency
        }
    }
    
    config_path = os.path.join(config.results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")


def run_research_experiment():
    """Run the complete research experiment with full visualization and analysis"""
    
    print("="*80)
    print("SkyNetRL Research Experiment")
    print("Multi-Agent Reinforcement Learning for Space-Air-Ground Networks")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create research configuration
    config = ResearchConfig()
    print(config)
    
    # Save configuration
    save_config_summary(config)
    
    # Initialize trainer
    print("\nInitializing MADDPG trainer with full visualization...")
    start_time = time.time()
    
    try:
        trainer = MADDPGTrainer(config)
        print("✓ Trainer initialized successfully")
        print(f"✓ Output directory: {config.base_dir}")
        print(f"✓ Visualization enabled: {config.real_time_visualization}")
        print(f"✓ Animation saving: {config.save_animation}")
        
        # Start training
        print(f"\nStarting research training ({config.num_episodes} episodes)...")
        print("This will generate:")
        print("  - Training metrics and learning curves")
        print("  - Real-time visualizations")
        print("  - Saved model checkpoints")
        print("  - Performance analysis")
        print("  - Animation files")
        print("\nTraining in progress...")
        
        # Run the complete training
        best_reward, best_episode = trainer.train()
        
        # Training completed
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "="*80)
        print("✓ RESEARCH EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print summary
        print(f"\nExperiment Summary:")
        print(f"  Total training time: {training_time/60:.2f} minutes")
        print(f"  Best reward: {best_reward:.2f} (Episode {best_episode})")
        print(f"  Results directory: {config.base_dir}")
        
        # Generate final analysis
        if hasattr(trainer, 'metrics'):
            print(f"\nGenerating final analysis...")
            summary = trainer.metrics.get_summary()
            
            # Save detailed summary
            summary_path = os.path.join(config.results_dir, "training_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"Final Performance Metrics:")
            for category in summary:
                print(f"\n{category}:")
                for metric, values in summary[category].items():
                    if isinstance(values, dict) and 'current' in values:
                        print(f"  {metric}: {values['current']:.3f} (avg: {values['mean']:.3f} ± {values['std']:.3f})")
        
        # List generated files
        print(f"\nGenerated Files:")
        if os.path.exists(config.visualization_dir):
            viz_files = os.listdir(config.visualization_dir)
            print(f"  Visualizations ({len(viz_files)} files): {config.visualization_dir}")
            
        if os.path.exists(config.model_save_path):
            model_dirs = [d for d in os.listdir(config.model_save_path) if os.path.isdir(os.path.join(config.model_save_path, d))]
            print(f"  Saved models ({len(model_dirs)} checkpoints): {config.model_save_path}")
            
        if os.path.exists(config.results_dir):
            result_files = os.listdir(config.results_dir)
            print(f"  Analysis results ({len(result_files)} files): {config.results_dir}")
        
        print(f"\n📊 All results are ready for research analysis and paper figures!")
        print(f"📁 Base directory: {config.base_dir}")
        
        return True, config.base_dir
        
    except Exception as e:
        print(f"\n✗ Research experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    success, results_dir = run_research_experiment()
    
    if success:
        print(f"\n🎉 Research experiment completed successfully!")
        print(f"📁 Results available in: {results_dir}")
    else:
        print(f"\n❌ Research experiment failed!")
    
    exit(0 if success else 1)