#!/usr/bin/env python3
"""
Main entry point for SkyNetRL - Multi-Agent Reinforcement Learning for Space-Air-Ground Networks

This script provides a unified interface to run different experiments and algorithms.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from experiments.complete_experiment import CompleteExperiment
from experiments.enhanced_complete_experiment import EnhancedCompleteExperiment
from experiments.comparison_experiment import ComparisonExperiment


def main():
    parser = argparse.ArgumentParser(
        description="SkyNetRL: Multi-Agent RL for Space-Air-Ground Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --algorithm ae_maddpg --episodes 100
  python main.py --algorithm all --episodes 50 --comparison
  python main.py --development-mode --episodes 10
  python main.py --comparison --algorithm all --3gpp-channels --mac-protocols
  python main.py --comparison --algorithm ae_maddpg --statistical-runs 10
        """
    )
    
    parser.add_argument(
        "--algorithm", 
        choices=["ae_maddpg", "baseline_maddpg", "qmix", "independent_ppo", "greedy_heuristic", "random_policy", "demo_algorithm", "all"],
        default="ae_maddpg",
        help="Algorithm to run (default: ae_maddpg)"
    )
    
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run algorithm comparison with statistical analysis and visualization"
    )
    
    parser.add_argument(
        "--3gpp-channels",
        action="store_true",
        help="Enable 3GPP-compliant channel modeling"
    )
    
    parser.add_argument(
        "--mac-protocols",
        action="store_true",
        help="Enable MAC layer and QoS protocol simulation"
    )
    
    parser.add_argument(
        "--statistical-runs",
        type=int,
        default=5,
        help="Number of independent runs for statistical analysis (default: 5)"
    )
    
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=50,
        help="Number of training episodes (default: 50)"
    )
    
    parser.add_argument(
        "--eval-episodes", 
        type=int, 
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    
    parser.add_argument(
        "--development-mode",
        action="store_true", 
        help="Run in development mode with reduced episodes"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip generating visualizations"
    )
    
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate video visualization of agent movements"
    )
    
    parser.add_argument(
        "--video-mode",
        choices=["overview", "tracking", "orbiting", "split_screen"],
        default="overview",
        help="Video camera mode (default: overview)"
    )
    
    parser.add_argument(
        "--realtime-3d",
        action="store_true",
        help="Enable real-time 3D visualization (requires OpenGL)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Adjust parameters for development mode
    if args.development_mode:
        args.episodes = min(args.episodes, 10)
        args.eval_episodes = min(args.eval_episodes, 5)
    
    # Load default configuration and update with command line args
    import json
    with open("configs/default_config.json", "r") as f:
        config = json.load(f)
    
    # Update with command line arguments
    config["seed"] = args.seed
    config["num_episodes"] = args.episodes
    config["num_eval_episodes"] = args.eval_episodes
    config["enable_visualization"] = not args.no_visualization
    config["enable_video"] = args.video
    config["video_mode"] = args.video_mode
    config["enable_realtime_3d"] = args.realtime_3d
    config["comparison_mode"] = args.comparison
    config["enable_3gpp_channels"] = args.__dict__.get('3gpp_channels', False)
    config["enable_mac_protocols"] = args.mac_protocols
    config["num_statistical_runs"] = args.statistical_runs
    
    # Add visualization config
    config["visualization"] = {
        "enable_video": args.video,
        "video_mode": args.video_mode,
        "enable_realtime_3d": args.realtime_3d
    }
    
    # Choose experiment type based on arguments
    if args.comparison:
        print("📊 Using Algorithm Comparison with Statistical Analysis")
        experiment = ComparisonExperiment(config)
    elif args.video or args.realtime_3d:
        print("🎬 Using Enhanced Experiment with Video Visualization")
        experiment = EnhancedCompleteExperiment(config)
    else:
        print("📊 Using Standard Experiment")
        experiment = CompleteExperiment(config)
    
    if args.comparison:
        # Comparison mode with statistical analysis
        if args.algorithm == "all":
            print("Running complete algorithm comparison with statistical analysis...")
            results = experiment.run_complete_comparison()
        else:
            print(f"Running single algorithm with statistical analysis: {args.algorithm}")
            results = experiment.run_single_algorithm(args.algorithm, args.statistical_runs)
    elif args.algorithm == "all":
        print("Running complete algorithm comparison...")
        if hasattr(experiment, 'run_complete_comparison'):
            results = experiment.run_complete_comparison()
        else:
            results = experiment.run_comparison_study()
    else:
        print(f"Running single algorithm: {args.algorithm}")
        if hasattr(experiment, 'run_single_algorithm'):
            results = experiment.run_single_algorithm(args.algorithm)
        else:
            # For standard experiment, run comparison but only with specified algorithm
            config["algorithms"] = [args.algorithm]
            results = experiment.run_comparison_study()
    
    print("\nExperiment completed successfully!")
    if hasattr(experiment, 'output_dir'):
        print(f"Results saved to: {experiment.output_dir}")
    else:
        print(f"Results saved to: results/{config.get('experiment_name', 'experiment')}")


if __name__ == "__main__":
    main()