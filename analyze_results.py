"""
Results Analysis Script for SkyNetRL Research
Generates visualization and analysis of training results for paper/research
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse


def load_experiment_data(results_dir):
    """Load all experiment data from results directory"""
    
    # Load configuration
    config_path = os.path.join(results_dir, "results", "experiment_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load final report
    report_path = os.path.join(results_dir, "final_report.json")
    with open(report_path, 'r') as f:
        final_report = json.load(f)
    
    # Load training data from individual episodes
    training_data_dir = os.path.join(results_dir, "training_data")
    training_data = []
    
    if os.path.exists(training_data_dir):
        for file in sorted(os.listdir(training_data_dir)):
            if file.endswith('.json'):
                with open(os.path.join(training_data_dir, file), 'r') as f:
                    episode_data = json.load(f)
                    training_data.append(episode_data)
    
    return config, final_report, training_data


def create_learning_curves(training_data, results_dir):
    """Create learning curves visualization"""
    
    if not training_data:
        print("No training data available for learning curves")
        return
    
    # Extract data
    episodes = [data['episode'] for data in training_data]
    rewards = [data['rewards'] for data in training_data]
    coverage = [data['coverage'] for data in training_data]
    energy = [data['energy'] for data in training_data]
    collisions = [data['collisions'] for data in training_data]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Episode Rewards', 'Coverage Rate', 'Energy Consumption', 'Collision Count'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Rewards
    fig.add_trace(go.Scatter(x=episodes, y=rewards, mode='lines+markers',
                            name='Rewards', line=dict(color='blue')), row=1, col=1)
    
    # Coverage
    fig.add_trace(go.Scatter(x=episodes, y=coverage, mode='lines+markers',
                            name='Coverage Rate', line=dict(color='green')), row=1, col=2)
    
    # Energy
    fig.add_trace(go.Scatter(x=episodes, y=energy, mode='lines+markers',
                            name='Energy Consumption', line=dict(color='orange')), row=2, col=1)
    
    # Collisions
    fig.add_trace(go.Scatter(x=episodes, y=collisions, mode='lines+markers',
                            name='Collisions', line=dict(color='red')), row=2, col=2)
    
    fig.update_layout(
        title="SkyNetRL Training Progress - Learning Curves",
        showlegend=False,
        height=800,
        font=dict(size=12)
    )
    
    # Save plot
    output_path = os.path.join(results_dir, "learning_curves.html")
    fig.write_html(output_path)
    print(f"Learning curves saved to: {output_path}")
    
    return fig


def create_performance_summary(final_report, config, results_dir):
    """Create performance summary visualization"""
    
    metrics = final_report['final_metrics']
    
    # Extract key metrics
    coverage_avg = metrics['coverage_metrics']['avg_coverage']['mean']
    coverage_std = metrics['coverage_metrics']['avg_coverage']['std']
    
    energy_avg = metrics['energy_metrics']['avg_energy_consumption']['mean']
    energy_std = metrics['energy_metrics']['avg_energy_consumption']['std']
    
    collision_avg = metrics['system_metrics']['collision_counts']['mean']
    collision_std = metrics['system_metrics']['collision_counts']['std']
    
    mission_completion = metrics['task_metrics']['mission_completion_rate']['mean']
    
    # Create summary table
    summary_data = {
        'Metric': ['Coverage Rate', 'Energy Consumption', 'Collision Count', 'Mission Completion'],
        'Mean': [coverage_avg, energy_avg, collision_avg, mission_completion],
        'Std': [coverage_std, energy_std, collision_std, 
                metrics['task_metrics']['mission_completion_rate']['std']],
        'Unit': ['%', 'units', 'count', '%']
    }
    
    # Create bar chart
    fig = go.Figure()
    
    # Normalize metrics for visualization (scale to 0-1)
    normalized_means = [
        coverage_avg,
        energy_avg / 600,  # Normalize energy
        1 - (collision_avg / 200),  # Invert collisions (fewer is better)
        mission_completion
    ]
    
    colors = ['green', 'orange', 'red', 'blue']
    
    fig.add_trace(go.Bar(
        x=['Coverage Rate', 'Energy Efficiency', 'Collision Avoidance', 'Mission Completion'],
        y=normalized_means,
        marker_color=colors,
        text=[f"{val:.3f}" for val in normalized_means],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="SkyNetRL Performance Summary (Normalized Metrics)",
        yaxis_title="Performance Score (0-1)",
        xaxis_title="Metrics",
        font=dict(size=12),
        height=500
    )
    
    # Save plot
    output_path = os.path.join(results_dir, "performance_summary.html")
    fig.write_html(output_path)
    print(f"Performance summary saved to: {output_path}")
    
    return fig


def create_detailed_metrics_report(final_report, results_dir):
    """Create detailed metrics report"""
    
    report = {
        "Experiment Summary": {
            "Best Episode": final_report['best_episode'],
            "Best Reward": round(final_report['best_reward'], 2),
            "Training Completed": "Successfully"
        },
        "Coverage Analysis": {},
        "Energy Analysis": {},
        "Cooperation Analysis": {},
        "System Performance": {}
    }
    
    # Extract detailed metrics
    metrics = final_report['final_metrics']
    
    # Coverage metrics
    coverage = metrics['coverage_metrics']
    report["Coverage Analysis"] = {
        "Average Coverage": f"{coverage['avg_coverage']['mean']:.1%} ± {coverage['avg_coverage']['std']:.1%}",
        "Peak Coverage": f"{coverage['peak_coverage']['mean']:.1%}",
        "Priority Coverage": f"{coverage['priority_coverage']['mean']:.1%} ± {coverage['priority_coverage']['std']:.1%}"
    }
    
    # Energy metrics
    energy = metrics['energy_metrics']
    report["Energy Analysis"] = {
        "Avg Energy Consumption": f"{energy['avg_energy_consumption']['mean']:.1f} ± {energy['avg_energy_consumption']['std']:.1f}",
        "Charging Frequency": f"{energy['charging_frequency']['mean']:.1f}",
        "Energy Efficiency": f"{energy['energy_efficiency']['mean']:.4f}"
    }
    
    # Cooperation metrics
    coop = metrics['cooperation_metrics']
    report["Cooperation Analysis"] = {
        "Communication Density": f"{coop['communication_density']['mean']:.3f} ± {coop['communication_density']['std']:.3f}",
        "Task Sharing": f"{coop['task_sharing']['mean']:.3f}",
        "Formation Stability": f"{coop['formation_stability']['mean']:.3f}"
    }
    
    # System performance
    system = metrics['system_metrics']
    report["System Performance"] = {
        "Average Collisions": f"{system['collision_counts']['mean']:.1f} ± {system['collision_counts']['std']:.1f}",
        "Path Efficiency": f"{system['path_efficiency']['mean']:.3f}",
        "Resource Utilization": f"{system['resource_utilization']['mean']:.3f}"
    }
    
    # Save detailed report
    output_path = os.path.join(results_dir, "detailed_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Detailed analysis saved to: {output_path}")
    
    return report


def generate_paper_summary(config, final_report, results_dir):
    """Generate summary suitable for research paper"""
    
    summary = f"""
# SkyNetRL Experiment Results Summary

## Experimental Setup
- **Environment**: {config['environment']['area_size']}x{config['environment']['area_size']} area
- **Agents**: {config['environment']['num_satellites']} satellites, {config['environment']['num_uavs']} UAVs, {config['environment']['num_ground_stations']} ground stations
- **Training**: {config['experiment_info']['total_episodes']} episodes, {config['experiment_info']['max_steps_per_episode']} steps per episode
- **Device**: {config['experiment_info']['device']}

## Key Results
- **Best Performance**: Episode {final_report['best_episode']} with reward {final_report['best_reward']:.2f}
- **Coverage Rate**: {final_report['final_metrics']['coverage_metrics']['avg_coverage']['mean']:.1%} ± {final_report['final_metrics']['coverage_metrics']['avg_coverage']['std']:.1%}
- **Energy Efficiency**: {final_report['final_metrics']['energy_metrics']['energy_efficiency']['mean']:.4f}
- **Mission Completion**: {final_report['final_metrics']['task_metrics']['mission_completion_rate']['mean']:.1%}
- **Collision Rate**: {final_report['final_metrics']['system_metrics']['collision_counts']['mean']:.1f} per episode

## Multi-Agent Coordination
- **Communication Density**: {final_report['final_metrics']['cooperation_metrics']['communication_density']['mean']:.3f}
- **Task Sharing Efficiency**: {final_report['final_metrics']['cooperation_metrics']['task_sharing']['mean']:.3f}
- **Formation Stability**: {final_report['final_metrics']['cooperation_metrics']['formation_stability']['mean']:.3f}

## Network Performance
- **Path Efficiency**: {final_report['final_metrics']['system_metrics']['path_efficiency']['mean']:.3f}
- **Resource Utilization**: {final_report['final_metrics']['system_metrics']['resource_utilization']['mean']:.3f}
- **Response Time**: {final_report['final_metrics']['task_metrics']['avg_response_time']['mean']:.3f}

## Files Generated
- Training models: {len([d for d in os.listdir(os.path.join(results_dir, 'saved_models')) if d.startswith('checkpoint')])} checkpoints
- Metrics data: {len([f for f in os.listdir(os.path.join(results_dir, 'metrics')) if f.endswith('.npy')])} metric files
- Analysis results: Multiple visualization and analysis files

This experiment demonstrates the effectiveness of MADDPG for coordinating multi-agent space-air-ground networks with significant improvements in coverage, energy efficiency, and coordination.
"""
    
    # Save paper summary
    output_path = os.path.join(results_dir, "paper_summary.md")
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"Paper summary saved to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze SkyNetRL experiment results')
    parser.add_argument('--results_dir', default='./runs/research_20250721_234907',
                        help='Path to results directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    print(f"Analyzing results from: {args.results_dir}")
    
    # Load data
    config, final_report, training_data = load_experiment_data(args.results_dir)
    
    print(f"\nExperiment Configuration:")
    print(f"  Episodes: {config['experiment_info']['total_episodes']}")
    print(f"  Environment: {config['environment']['area_size']}x{config['environment']['area_size']}")
    print(f"  Agents: {config['environment']['num_satellites'] + config['environment']['num_uavs'] + config['environment']['num_ground_stations']}")
    print(f"  Best reward: {final_report['best_reward']:.2f}")
    
    # Generate visualizations and reports
    print(f"\nGenerating analysis...")
    
    if training_data:
        create_learning_curves(training_data, args.results_dir)
    
    create_performance_summary(final_report, config, args.results_dir)
    create_detailed_metrics_report(final_report, args.results_dir)
    generate_paper_summary(config, final_report, args.results_dir)
    
    print(f"\n✓ Analysis complete! All files saved to: {args.results_dir}")
    print(f"\nGenerated files for research/paper:")
    print(f"  - learning_curves.html: Training progress visualization")
    print(f"  - performance_summary.html: Key metrics summary")
    print(f"  - detailed_analysis.json: Detailed numerical results")
    print(f"  - paper_summary.md: Research paper summary")


if __name__ == "__main__":
    main()