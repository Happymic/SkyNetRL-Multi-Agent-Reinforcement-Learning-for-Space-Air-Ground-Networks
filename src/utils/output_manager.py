#!/usr/bin/env python3
"""
Standardized Output Management System for SkyNetRL
Handles all experiment outputs in a consistent, professional format
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import zipfile


@dataclass
class ExperimentConfig:
    """Configuration for experiment output management"""
    name: str
    algorithm: str
    episodes: int
    start_time: datetime
    config: Dict[str, Any]
    output_dir: str


class StandardizedOutputManager:
    """
    Manages all experiment outputs in standardized format
    """
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.output_dir = Path(experiment_config.output_dir)
        self.experiment_dir = self.output_dir / experiment_config.name
        
        # Create standardized directory structure
        self._create_directory_structure()
        
        # Initialize tracking variables
        self.metrics_data = []
        self.episode_videos = []
        self.plots_generated = []
        self.models_saved = []
        
        # Save initial experiment config
        self._save_experiment_config()
        
        print(f"📁 Standardized Output Manager initialized: {self.experiment_dir}")
    
    def _create_directory_structure(self):
        """Create the standardized directory structure"""
        directories = [
            self.experiment_dir,
            self.experiment_dir / "logs",
            self.experiment_dir / "models" / "checkpoints",
            self.experiment_dir / "plots",
            self.experiment_dir / "videos",
            self.experiment_dir / "analysis"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _save_experiment_config(self):
        """Save experiment configuration"""
        config_data = {
            "experiment": {
                "name": self.config.name,
                "algorithm": self.config.algorithm,
                "start_time": self.config.start_time.isoformat(),
                "episodes": self.config.episodes,
                "status": "running",
                "output_directory": str(self.experiment_dir)
            },
            "configuration": self.config.config
        }
        
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def record_episode_metrics(self, episode: int, metrics: Dict[str, Any]):
        """Record metrics for a single episode"""
        # Add timestamp and episode number
        metrics_with_meta = {
            "episode": episode,
            "timestamp": datetime.now().timestamp(),
            **metrics
        }
        
        self.metrics_data.append(metrics_with_meta)
        
        # Save to CSV periodically (every 10 episodes)
        if episode % 10 == 0 or episode == 1:
            self._save_metrics_csv()
    
    def _save_metrics_csv(self):
        """Save metrics to CSV file"""
        if not self.metrics_data:
            return
            
        df = pd.DataFrame(self.metrics_data)
        csv_file = self.experiment_dir / "logs" / "metrics.csv"
        df.to_csv(csv_file, index=False)
    
    def save_episode_video(self, episode: int, video_path: str, video_type: str = "gif"):
        """Save episode video with standardized naming"""
        if not os.path.exists(video_path):
            return
        
        # Standardized video filename
        video_name = f"episode_{episode:03d}.{video_type}"
        target_path = self.experiment_dir / "videos" / video_name
        
        # Copy video to standardized location
        shutil.copy2(video_path, target_path)
        
        # Track saved videos
        self.episode_videos.append({
            "episode": episode,
            "filename": video_name,
            "path": str(target_path),
            "type": video_type,
            "size_mb": os.path.getsize(target_path) / (1024 * 1024)
        })
        
        print(f"🎥 Saved episode video: {video_name}")
    
    def save_training_plot(self, plot_name: str, figure: plt.Figure = None):
        """Save training plots with standardized naming"""
        plot_file = self.experiment_dir / "plots" / f"{plot_name}.png"
        
        if figure is None:
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        else:
            figure.savefig(plot_file, dpi=300, bbox_inches='tight',
                         facecolor='white', edgecolor='none')
        
        # Track saved plots
        self.plots_generated.append({
            "name": plot_name,
            "filename": f"{plot_name}.png",
            "path": str(plot_file),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"📊 Saved training plot: {plot_name}.png")
    
    def save_model_checkpoint(self, episode: int, model_state: Dict, 
                            is_best: bool = False, is_final: bool = False):
        """Save model checkpoints with standardized naming"""
        if is_final:
            model_name = "final_model.pth"
            model_path = self.experiment_dir / "models" / model_name
        elif is_best:
            model_name = "best_model.pth"
            model_path = self.experiment_dir / "models" / model_name
        else:
            model_name = f"episode_{episode:03d}.pth"
            model_path = self.experiment_dir / "models" / "checkpoints" / model_name
        
        # Save model (placeholder - would use torch.save in real implementation)
        model_info = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "model_config": model_state,
            "is_best": is_best,
            "is_final": is_final
        }
        
        with open(str(model_path).replace('.pth', '.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Track saved models
        self.models_saved.append({
            "episode": episode,
            "filename": model_name,
            "path": str(model_path),
            "is_best": is_best,
            "is_final": is_final
        })
        
        print(f"🤖 Saved model: {model_name}")
    
    def generate_training_curves(self):
        """Generate comprehensive training curves"""
        if len(self.metrics_data) < 2:
            return
        
        df = pd.DataFrame(self.metrics_data)
        
        # Set style for professional plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comprehensive training curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.config.algorithm.upper()} Training Progress - {self.config.name}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Total Reward
        if 'total_reward' in df.columns:
            axes[0,0].plot(df['episode'], df['total_reward'], label='Episode Reward', alpha=0.7)
            axes[0,0].plot(df['episode'], df['total_reward'].rolling(window=10, min_periods=1).mean(), 
                          label='Moving Average (10)', linewidth=2)
            axes[0,0].set_title('Total Reward')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Reward')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Coverage Rate
        if 'coverage_rate' in df.columns:
            axes[0,1].plot(df['episode'], df['coverage_rate'] * 100, label='Coverage Rate', color='green')
            axes[0,1].set_title('Coverage Rate')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Coverage (%)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Energy Efficiency
        if 'energy_efficiency' in df.columns:
            axes[0,2].plot(df['episode'], df['energy_efficiency'], label='Energy Efficiency', color='orange')
            axes[0,2].set_title('Energy Efficiency')
            axes[0,2].set_xlabel('Episode')
            axes[0,2].set_ylabel('Efficiency')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Loss (if available)
        if 'total_loss' in df.columns:
            axes[1,0].plot(df['episode'], df['total_loss'], label='Training Loss', color='red')
            axes[1,0].set_title('Training Loss')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Loss')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Gradient Norm (if available)
        if 'gradient_norm' in df.columns:
            axes[1,1].plot(df['episode'], df['gradient_norm'], label='Gradient Norm', color='purple')
            axes[1,1].set_title('Gradient Norm')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Gradient Norm')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Learning Rate (if available)
        if 'learning_rate' in df.columns:
            axes[1,2].plot(df['episode'], df['learning_rate'], label='Learning Rate', color='brown')
            axes[1,2].set_title('Learning Rate')
            axes[1,2].set_xlabel('Episode')
            axes[1,2].set_ylabel('Learning Rate')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_training_plot("training_curves", fig)
        plt.close()
    
    def generate_reward_breakdown_plot(self):
        """Generate reward component breakdown analysis"""
        if len(self.metrics_data) < 2:
            return
        
        df = pd.DataFrame(self.metrics_data)
        
        # Find reward components
        reward_columns = [col for col in df.columns if col.startswith('reward_')]
        
        if not reward_columns:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Reward Component Analysis - {self.config.name}', 
                    fontsize=16, fontweight='bold')
        
        # Stacked area plot of reward components
        reward_data = df[reward_columns]
        axes[0].stackplot(df['episode'], *[reward_data[col] for col in reward_columns], 
                         labels=[col.replace('reward_', '').title() for col in reward_columns],
                         alpha=0.7)
        axes[0].set_title('Reward Components Over Time (Stacked)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward Value')
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[0].grid(True, alpha=0.3)
        
        # Individual reward components
        for col in reward_columns[:6]:  # Show top 6 components
            axes[1].plot(df['episode'], reward_data[col].rolling(window=5, min_periods=1).mean(), 
                        label=col.replace('reward_', '').title(), alpha=0.8)
        
        axes[1].set_title('Individual Reward Components (Moving Average)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Reward Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_training_plot("reward_breakdown", fig)
        plt.close()
    
    def create_experiment_summary(self):
        """Create comprehensive experiment summary"""
        end_time = datetime.now()
        duration = end_time - self.config.start_time
        
        # Calculate summary statistics
        summary_stats = {}
        if self.metrics_data:
            df = pd.DataFrame(self.metrics_data)
            
            # Performance metrics
            if 'total_reward' in df.columns:
                summary_stats['reward'] = {
                    'best': float(df['total_reward'].max()),
                    'final': float(df['total_reward'].iloc[-1]) if len(df) > 0 else 0,
                    'average': float(df['total_reward'].mean()),
                    'std': float(df['total_reward'].std())
                }
            
            if 'coverage_rate' in df.columns:
                summary_stats['coverage'] = {
                    'best': float(df['coverage_rate'].max() * 100),
                    'final': float(df['coverage_rate'].iloc[-1] * 100) if len(df) > 0 else 0,
                    'average': float(df['coverage_rate'].mean() * 100)
                }
            
            if 'energy_efficiency' in df.columns:
                summary_stats['efficiency'] = {
                    'best': float(df['energy_efficiency'].max()),
                    'final': float(df['energy_efficiency'].iloc[-1]) if len(df) > 0 else 0,
                    'average': float(df['energy_efficiency'].mean())
                }
        
        # Create comprehensive summary
        summary = {
            "experiment_info": {
                "name": self.config.name,
                "algorithm": self.config.algorithm,
                "start_time": self.config.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_formatted": str(duration),
                "total_episodes": len(self.metrics_data),
                "status": "completed"
            },
            "performance": summary_stats,
            "files_generated": {
                "videos": self.episode_videos,
                "plots": self.plots_generated,
                "models": self.models_saved,
                "total_files": len(self.episode_videos) + len(self.plots_generated) + len(self.models_saved)
            },
            "directory_structure": self._get_directory_structure()
        }
        
        # Save summary
        summary_file = self.experiment_dir / "analysis" / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update config with completion status
        self._update_experiment_config(end_time, "completed")
        
        print(f"📋 Experiment summary created: {summary_file}")
        return summary
    
    def _get_directory_structure(self):
        """Get directory structure information"""
        structure = {}
        for item in self.experiment_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(self.experiment_dir)
                category = str(rel_path.parts[0]) if len(rel_path.parts) > 1 else "root"
                if category not in structure:
                    structure[category] = []
                structure[category].append({
                    "filename": item.name,
                    "size_mb": round(item.stat().st_size / (1024 * 1024), 3),
                    "path": str(rel_path)
                })
        return structure
    
    def _update_experiment_config(self, end_time: datetime, status: str):
        """Update experiment config with completion info"""
        config_file = self.experiment_dir / "config.json"
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        config_data["experiment"]["end_time"] = end_time.isoformat()
        config_data["experiment"]["status"] = status
        config_data["experiment"]["duration"] = str(end_time - self.config.start_time)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def create_html_report(self):
        """Create HTML report for easy viewing"""
        summary = self.create_experiment_summary()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SkyNetRL Experiment Report - {self.config.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; }}
        .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .section {{ margin: 20px 0; }}
        .files-list {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛰️ SkyNetRL Experiment Report</h1>
            <h2>{self.config.name}</h2>
            <p><strong>Algorithm:</strong> {self.config.algorithm.upper()}</p>
        </div>
        
        <div class="section">
            <h3>📊 Experiment Overview</h3>
            <div class="grid">
                <div class="metric">
                    <h4>Duration</h4>
                    <p>{summary['experiment_info']['duration_formatted']}</p>
                </div>
                <div class="metric">
                    <h4>Episodes</h4>
                    <p>{summary['experiment_info']['total_episodes']}</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>🎯 Performance Results</h3>
"""
        
        # Add performance metrics if available
        if 'reward' in summary['performance']:
            html_content += f"""
            <div class="grid">
                <div class="metric">
                    <h4>Best Reward</h4>
                    <p>{summary['performance']['reward']['best']:.2f}</p>
                </div>
                <div class="metric">
                    <h4>Average Reward</h4>
                    <p>{summary['performance']['reward']['average']:.2f}</p>
                </div>
            </div>
"""
        
        if 'coverage' in summary['performance']:
            html_content += f"""
            <div class="grid">
                <div class="metric">
                    <h4>Best Coverage</h4>
                    <p>{summary['performance']['coverage']['best']:.1f}%</p>
                </div>
                <div class="metric">
                    <h4>Average Coverage</h4>
                    <p>{summary['performance']['coverage']['average']:.1f}%</p>
                </div>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="section">
            <h3>📁 Generated Files</h3>
            <div class="files-list">
                <p><strong>Videos:</strong> {len(summary['files_generated']['videos'])} files</p>
                <p><strong>Plots:</strong> {len(summary['files_generated']['plots'])} files</p>
                <p><strong>Models:</strong> {len(summary['files_generated']['models'])} files</p>
                <p><strong>Total:</strong> {summary['files_generated']['total_files']} files</p>
            </div>
        </div>
        
        <div class="section">
            <h3>📈 Available Plots</h3>
            <ul>
"""
        
        for plot in summary['files_generated']['plots']:
            html_content += f"<li>{plot['name']}.png</li>"
        
        html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h3>📝 Notes</h3>
            <p>This experiment was conducted using the SkyNetRL Multi-Agent Reinforcement Learning framework. 
            All results, plots, and videos are saved in the experiment directory for detailed analysis.</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_file = self.experiment_dir / "analysis" / "experiment_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML report created: {html_file}")
    
    def compress_experiment(self):
        """Create compressed archive of entire experiment"""
        archive_name = f"{self.config.name}.zip"
        archive_path = self.output_dir / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.experiment_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.output_dir)
                    zipf.write(file_path, arcname)
        
        print(f"📦 Experiment archived: {archive_path}")
        return archive_path
    
    def finalize_experiment(self):
        """Finalize experiment with all outputs and reports"""
        print(f"\n🏁 Finalizing experiment: {self.config.name}")
        
        # Generate final visualizations
        self.generate_training_curves()
        self.generate_reward_breakdown_plot()
        
        # Save final metrics
        self._save_metrics_csv()
        
        # Create comprehensive summary
        summary = self.create_experiment_summary()
        
        # Create HTML report
        self.create_html_report()
        
        print(f"✅ Experiment finalized successfully!")
        print(f"📁 Results saved to: {self.experiment_dir}")
        print(f"📊 Total metrics recorded: {len(self.metrics_data)}")
        print(f"🎥 Videos generated: {len(self.episode_videos)}")
        print(f"📈 Plots created: {len(self.plots_generated)}")
        
        return summary


def create_output_manager(experiment_name: str, algorithm: str, episodes: int, 
                         config: Dict[str, Any], output_dir: str = "outputs") -> StandardizedOutputManager:
    """
    Factory function to create a standardized output manager
    """
    exp_config = ExperimentConfig(
        name=experiment_name,
        algorithm=algorithm,
        episodes=episodes,
        start_time=datetime.now(),
        config=config,
        output_dir=output_dir
    )
    
    return StandardizedOutputManager(exp_config)