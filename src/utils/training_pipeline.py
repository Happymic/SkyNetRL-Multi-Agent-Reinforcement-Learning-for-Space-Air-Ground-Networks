"""
Standardized Training Output Pipeline
=====================================
Automated system for generating consistent, professional outputs during training.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import logging


@dataclass
class TrainingOutputConfig:
    """Configuration for training output structure"""
    # Base paths
    base_output_dir: str = "outputs"
    experiment_name: str = "experiment"
    
    # Output subdirectories
    create_videos: bool = True
    create_plots: bool = True
    create_reports: bool = True
    create_checkpoints: bool = True
    create_logs: bool = True
    
    # Visualization settings
    save_gif: bool = True
    save_mp4: bool = True
    gif_fps: int = 10
    video_fps: int = 24
    
    # Checkpoint settings
    checkpoint_frequency: int = 100  # Episodes
    keep_best_n_checkpoints: int = 5
    
    # Report settings
    report_frequency: int = 50  # Episodes
    generate_html_dashboard: bool = True
    
    # Clean mode
    clean_previous: bool = False


class StandardizedTrainingPipeline:
    """Manages standardized output structure for training runs"""
    
    def __init__(self, config: Optional[TrainingOutputConfig] = None):
        self.config = config or TrainingOutputConfig()
        self.logger = self._setup_logger()
        
        # Create output structure
        self.paths = self._setup_output_structure()
        
        # Initialize tracking
        self.episode_data = []
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'coverage': [],
            'communication_success': [],
            'collision_rate': [],
            'timestamps': []
        }
        self.best_reward = -float('inf')
        self.checkpoint_scores = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup training logger"""
        logger = logging.getLogger('TrainingPipeline')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_output_structure(self) -> Dict[str, Path]:
        """Create standardized output directory structure"""
        # Generate unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.experiment_name}_{timestamp}"
        
        # Base directory
        base_path = Path(self.config.base_output_dir) / exp_name
        
        # Create directory structure
        paths = {
            'base': base_path,
            'videos': base_path / 'videos',
            'plots': base_path / 'plots',
            'reports': base_path / 'reports',
            'checkpoints': base_path / 'checkpoints',
            'logs': base_path / 'logs',
            'configs': base_path / 'configs'
        }
        
        # Create directories
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = paths['configs'] / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        self.logger.info(f"Created output structure at: {base_path}")
        
        return paths
    
    def log_episode(self, episode: int, data: Dict[str, Any]):
        """Log episode data and generate outputs if needed"""
        # Store episode data
        self.episode_data.append(data)
        
        # Update metrics
        self.training_metrics['episodes'].append(episode)
        self.training_metrics['rewards'].append(data.get('total_reward', 0))
        self.training_metrics['coverage'].append(data.get('avg_coverage', 0))
        self.training_metrics['communication_success'].append(
            data.get('communication_success', 0)
        )
        self.training_metrics['collision_rate'].append(
            data.get('collision_rate', 0)
        )
        self.training_metrics['timestamps'].append(datetime.now().isoformat())
        
        # Check if we should generate outputs
        if episode % self.config.checkpoint_frequency == 0:
            self._save_checkpoint(episode, data)
        
        if episode % self.config.report_frequency == 0:
            self._generate_report(episode)
        
        # Log to file
        if self.config.create_logs:
            self._log_to_file(episode, data)
    
    def _save_checkpoint(self, episode: int, data: Dict):
        """Save model checkpoint with metadata"""
        checkpoint_path = self.paths['checkpoints'] / f"checkpoint_ep{episode}.pt"
        
        checkpoint_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'reward': data.get('total_reward', 0),
                'coverage': data.get('avg_coverage', 0),
                'communication': data.get('communication_success', 0)
            },
            'model_state': data.get('model_state', None)
        }
        
        # Save checkpoint (placeholder - actual saving depends on framework)
        with open(checkpoint_path.with_suffix('.json'), 'w') as f:
            json.dump({k: v for k, v in checkpoint_data.items() 
                      if k != 'model_state'}, f, indent=2)
        
        # Track checkpoint scores for pruning
        score = data.get('total_reward', 0)
        self.checkpoint_scores.append((score, checkpoint_path))
        
        # Prune old checkpoints if needed
        if len(self.checkpoint_scores) > self.config.keep_best_n_checkpoints:
            self.checkpoint_scores.sort(key=lambda x: x[0], reverse=True)
            to_remove = self.checkpoint_scores[self.config.keep_best_n_checkpoints:]
            for _, path in to_remove:
                if path.exists():
                    path.unlink()
                json_path = path.with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
            self.checkpoint_scores = self.checkpoint_scores[:self.config.keep_best_n_checkpoints]
        
        self.logger.info(f"Saved checkpoint at episode {episode}")
    
    def _generate_report(self, episode: int):
        """Generate comprehensive training report"""
        if not self.config.create_reports:
            return
        
        # Create plots
        if self.config.create_plots:
            self._create_training_plots(episode)
        
        # Generate HTML dashboard
        if self.config.generate_html_dashboard:
            self._create_html_dashboard(episode)
        
        self.logger.info(f"Generated report at episode {episode}")
    
    def _create_training_plots(self, episode: int):
        """Create training progress plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')
        
        # Reward plot
        axes[0, 0].plot(self.training_metrics['episodes'], 
                       self.training_metrics['rewards'],
                       color='#E74C3C', linewidth=2)
        axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Coverage plot
        axes[0, 1].plot(self.training_metrics['episodes'],
                       self.training_metrics['coverage'],
                       color='#3498DB', linewidth=2)
        axes[0, 1].set_title('Coverage Rate', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Coverage (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Communication success plot
        axes[1, 0].plot(self.training_metrics['episodes'],
                       self.training_metrics['communication_success'],
                       color='#27AE60', linewidth=2)
        axes[1, 0].set_title('Communication Success', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Collision rate plot
        axes[1, 1].plot(self.training_metrics['episodes'],
                       self.training_metrics['collision_rate'],
                       color='#F39C12', linewidth=2)
        axes[1, 1].set_title('Collision Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Collisions per Episode')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress - Episode {episode}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.paths['plots'] / f"training_progress_ep{episode}.png"
        plt.savefig(plot_path, dpi=100, facecolor='white', bbox_inches='tight')
        plt.close()
    
    def _create_html_dashboard(self, episode: int):
        """Create HTML dashboard for training progress"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Training Dashboard - Episode {episode}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    background: #ffffff;
                    color: #333333;
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .metric-card {{
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                    transition: transform 0.2s;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}
                
                .metric-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .metric-label {{
                    color: #6c757d;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .section {{
                    background: white;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                
                .section h2 {{
                    color: #495057;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #e9ecef;
                }}
                
                .plot-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .plot-container img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #6c757d;
                    border-top: 1px solid #dee2e6;
                    margin-top: 40px;
                }}
                
                .status-badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: bold;
                }}
                
                .status-training {{
                    background: #28a745;
                    color: white;
                }}
                
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                
                .table th, .table td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }}
                
                .table th {{
                    background: #f8f9fa;
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 SkyNetRL Training Dashboard</h1>
                    <p>Multi-Agent Reinforcement Learning for Space-Air-Ground Networks</p>
                    <p>Episode: {episode} | <span class="status-badge status-training">TRAINING</span></p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Reward</div>
                        <div class="metric-value" style="color: #E74C3C;">
                            {self.training_metrics['rewards'][-1] if self.training_metrics['rewards'] else 0:.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Coverage Rate</div>
                        <div class="metric-value" style="color: #3498DB;">
                            {self.training_metrics['coverage'][-1] if self.training_metrics['coverage'] else 0:.1f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Communication Success</div>
                        <div class="metric-value" style="color: #27AE60;">
                            {self.training_metrics['communication_success'][-1] if self.training_metrics['communication_success'] else 0:.1f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Collision Rate</div>
                        <div class="metric-value" style="color: #F39C12;">
                            {self.training_metrics['collision_rate'][-1] if self.training_metrics['collision_rate'] else 0:.2f}
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Training Progress</h2>
                    <div class="plot-container">
                        <img src="../plots/training_progress_ep{episode}.png" alt="Training Progress">
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Recent Performance</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Episode</th>
                                <th>Reward</th>
                                <th>Coverage</th>
                                <th>Communication</th>
                                <th>Collisions</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add recent episodes to table
        for i in range(max(0, len(self.training_metrics['episodes']) - 10), 
                      len(self.training_metrics['episodes'])):
            html_content += f"""
                            <tr>
                                <td>{self.training_metrics['episodes'][i]}</td>
                                <td>{self.training_metrics['rewards'][i]:.2f}</td>
                                <td>{self.training_metrics['coverage'][i]:.1f}%</td>
                                <td>{self.training_metrics['communication_success'][i]:.1f}%</td>
                                <td>{self.training_metrics['collision_rate'][i]:.2f}</td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>📁 Output Files</h2>
                    <ul>
                        <li>Videos: <code>videos/</code></li>
                        <li>Plots: <code>plots/</code></li>
                        <li>Checkpoints: <code>checkpoints/</code></li>
                        <li>Logs: <code>logs/</code></li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                    <p>SkyNetRL © 2024 | Professional Training Pipeline</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        dashboard_path = self.paths['reports'] / f"dashboard_ep{episode}.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        # Also save as latest
        latest_path = self.paths['reports'] / "latest_dashboard.html"
        shutil.copy(dashboard_path, latest_path)
    
    def _log_to_file(self, episode: int, data: Dict):
        """Log episode data to file"""
        log_path = self.paths['logs'] / f"training_log.jsonl"
        
        log_entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'reward': data.get('total_reward', 0),
                'coverage': data.get('avg_coverage', 0),
                'communication': data.get('communication_success', 0),
                'collisions': data.get('collision_rate', 0)
            }
        }
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def generate_episode_video(self, episode: int, frames: List[Dict],
                              visualizer=None):
        """Generate episode video with professional visualization"""
        if not self.config.create_videos:
            return
        
        if visualizer is None:
            from src.visualization.professional_visualizer import ProfessionalVisualizer
            visualizer = ProfessionalVisualizer(env=None)
        
        # Generate GIF
        if self.config.save_gif:
            gif_path = self.paths['videos'] / f"episode_{episode:04d}.gif"
            visualizer.config.fps = self.config.gif_fps
            visualizer.generate_episode_animation(frames, str(gif_path), format='gif')
            self.logger.info(f"Generated GIF: {gif_path}")
        
        # Generate MP4
        if self.config.save_mp4:
            mp4_path = self.paths['videos'] / f"episode_{episode:04d}.mp4"
            visualizer.config.fps = self.config.video_fps
            visualizer.generate_episode_animation(frames, str(mp4_path), format='mp4')
            self.logger.info(f"Generated MP4: {mp4_path}")
    
    def finalize_training(self):
        """Generate final reports and clean up"""
        self.logger.info("Finalizing training outputs...")
        
        # Generate final report
        if len(self.training_metrics['episodes']) > 0:
            final_episode = self.training_metrics['episodes'][-1]
            self._generate_report(final_episode)
        
        # Save complete metrics
        metrics_path = self.paths['reports'] / 'complete_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Create summary
        summary = {
            'experiment_name': self.config.experiment_name,
            'total_episodes': len(self.training_metrics['episodes']),
            'best_reward': max(self.training_metrics['rewards']) if self.training_metrics['rewards'] else 0,
            'avg_reward': np.mean(self.training_metrics['rewards']) if self.training_metrics['rewards'] else 0,
            'final_coverage': self.training_metrics['coverage'][-1] if self.training_metrics['coverage'] else 0,
            'output_path': str(self.paths['base'])
        }
        
        summary_path = self.paths['base'] / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training finalized. Outputs saved to: {self.paths['base']}")
        
        return str(self.paths['base'])