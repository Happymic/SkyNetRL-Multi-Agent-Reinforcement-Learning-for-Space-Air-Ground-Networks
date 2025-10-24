#!/usr/bin/env python3
"""
Clean Training Runner with Comprehensive Visualization
====================================================
Runs a full training session with professional visualization output.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from experiments.enhanced_complete_experiment import EnhancedCompleteExperiment
from visualization.clean_training_visualizer import CleanTrainingVisualizer


def run_clean_training():
    """Run training with clean visualization"""
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"clean_training_{timestamp}"
    output_dir = Path("outputs") / experiment_name
    
    print(f"🚀 Starting clean training experiment: {experiment_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Training configuration
    config = {
        "experiment_name": experiment_name,
        "algorithm": "ae_maddpg",
        "num_episodes": 100,
        "environment": {
            "num_agents": 6,
            "area_size": 1000,
            "num_pois": 20,
            "num_satellites": 2,
            "num_uavs": 3,
            "num_ground_stations": 1,
            "max_episode_steps": 200
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "gamma": 0.95,
            "tau": 0.01,
            "buffer_size": 100000,
            "update_frequency": 1
        },
        "attention": {
            "use_spatial": True,
            "use_agent": True,
            "use_task": True,
            "hidden_dim": 128,
            "num_heads": 4
        },
        "visualization": {
            "save_videos": True,
            "save_plots": True,
            "plot_frequency": 10,
            "create_dashboard": True
        }
    }
    
    # Save configuration
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("🔧 Configuration saved")
    
    # Initialize experiment
    experiment = EnhancedCompleteExperiment(config)
    
    # Run training
    print("🎯 Starting training...")
    start_time = time.time()
    
    results = experiment.run_single_algorithm("ae_maddpg")
    
    training_time = time.time() - start_time
    results['training_time'] = f"{training_time:.2f} seconds"
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Create comprehensive visualizations
    print("🎨 Creating comprehensive visualizations...")
    
    visualizer = CleanTrainingVisualizer(str(output_dir / "visualizations"))
    
    # Generate all visualizations
    viz_files = []
    
    # 1. Main training dashboard
    dashboard_file = visualizer.create_comprehensive_training_dashboard(
        results, 
        save_name='comprehensive_training_dashboard.png'
    )
    viz_files.append(dashboard_file)
    
    # 2. 3D performance landscape
    performance_3d = visualizer.create_3d_performance_landscape(
        results.get('training_stats', {}),
        save_name='3d_performance_landscape.png'
    )
    viz_files.append(performance_3d)
    
    # 3. Training animation (if episode data available)
    if 'episode_data' in results:
        animation_file = visualizer.create_training_animation(
            results['episode_data'],
            save_name='training_progress_animation.gif'
        )
        viz_files.append(animation_file)
    
    # 4. Generate comprehensive report
    report_file = visualizer.generate_training_report(
        results,
        save_name='comprehensive_training_report.json'
    )
    viz_files.append(report_file)
    
    # Save final results
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("📊 Results Summary:")
    print(f"   • Experiment: {experiment_name}")
    print(f"   • Episodes: {config['num_episodes']}")
    print(f"   • Training Time: {training_time:.2f}s")
    
    # Print final metrics if available
    if 'evaluation_stats' in results:
        final_metrics = results['evaluation_stats'].get('final_metrics', {})
        print(f"   • Final Coverage: {final_metrics.get('avg_coverage_rate', 0):.3f}")
        print(f"   • Energy Efficiency: {final_metrics.get('avg_energy_efficiency', 0):.3f}")
    
    print(f"   • Output Directory: {output_dir}")
    print(f"   • Visualizations: {len(viz_files)} files created")
    
    print("\n🎉 Clean training experiment completed successfully!")
    print(f"🔍 Check {output_dir}/visualizations/ for all visualization files")
    
    return str(output_dir)


if __name__ == "__main__":
    try:
        output_path = run_clean_training()
        print(f"\n✨ All files saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise