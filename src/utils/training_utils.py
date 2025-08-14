"""
Training Utilities for Multi-Agent Reinforcement Learning
"""

import os
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import json


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_preference: str = 'auto') -> torch.device:
    """
    Get appropriate device for training
    
    Args:
        device_preference: 'auto', 'cpu', 'cuda', or specific device
        
    Returns:
        device: PyTorch device
    """
    if device_preference == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_preference == 'cpu':
        device = torch.device('cpu')
    elif device_preference == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(device_preference)
    
    print(f"Using device: {device}")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(agents: List, optimizer_states: List, episode: int, 
                   save_dir: str, filename: Optional[str] = None):
    """
    Save training checkpoint
    
    Args:
        agents: List of agents to save
        optimizer_states: List of optimizer states
        episode: Current episode
        save_dir: Directory to save checkpoint
        filename: Optional custom filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_episode_{episode}.pth'
    
    checkpoint = {
        'episode': episode,
        'agents': [],
        'optimizer_states': optimizer_states
    }
    
    for i, agent in enumerate(agents):
        agent_data = {
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'target_actor_state_dict': agent.target_actor.state_dict(),
            'target_critic_state_dict': agent.target_critic.state_dict()
        }
        
        if hasattr(agent, 'training_stats'):
            agent_data['training_stats'] = agent.training_stats
        
        checkpoint['agents'].append(agent_data)
    
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(agents: List, checkpoint_path: str) -> int:
    """
    Load training checkpoint
    
    Args:
        agents: List of agents to load into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        episode: Episode number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    for i, agent in enumerate(agents):
        agent_data = checkpoint['agents'][i]
        
        agent.actor.load_state_dict(agent_data['actor_state_dict'])
        agent.critic.load_state_dict(agent_data['critic_state_dict'])
        agent.target_actor.load_state_dict(agent_data['target_actor_state_dict'])
        agent.target_critic.load_state_dict(agent_data['target_critic_state_dict'])
        
        if 'training_stats' in agent_data and hasattr(agent, 'training_stats'):
            agent.training_stats = agent_data['training_stats']
    
    episode = checkpoint['episode']
    print(f"Checkpoint loaded from episode {episode}")
    
    return episode


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute gradient norm for monitoring"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def linear_schedule(start_value: float, end_value: float, 
                   current_step: int, total_steps: int) -> float:
    """Linear scheduling function"""
    if current_step >= total_steps:
        return end_value
    
    fraction = current_step / total_steps
    return start_value + fraction * (end_value - start_value)


def exponential_schedule(start_value: float, end_value: float,
                        current_step: int, decay_rate: float) -> float:
    """Exponential scheduling function"""
    return end_value + (start_value - end_value) * np.exp(-decay_rate * current_step)


def cosine_schedule(start_value: float, end_value: float,
                   current_step: int, total_steps: int) -> float:
    """Cosine annealing schedule"""
    if current_step >= total_steps:
        return end_value
    
    cosine_decay = 0.5 * (1 + np.cos(np.pi * current_step / total_steps))
    return end_value + (start_value - end_value) * cosine_decay


class TrainingLogger:
    """Logger for training metrics and visualization"""
    
    def __init__(self, log_dir: str):
        """
        Initialize training logger
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            'episode_rewards': [],
            'coverage_rates': [],
            'energy_efficiency': [],
            'actor_losses': [],
            'critic_losses': [],
            'attention_entropy': []
        }
        
        self.episode_data = []
    
    def log_episode(self, episode: int, metrics: Dict):
        """Log metrics for an episode"""
        episode_log = {'episode': episode, **metrics}
        self.episode_data.append(episode_log)
        
        # Update running metrics
        for key, value in metrics.items():
            if key in self.metrics and isinstance(value, (int, float)):
                self.metrics[key].append(value)
    
    def save_metrics(self, filename: str = 'training_metrics.json'):
        """Save metrics to file"""
        filepath = os.path.join(self.log_dir, filename)
        
        data = {
            'metrics': self.metrics,
            'episode_data': self.episode_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None, show: bool = False):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.metrics['episode_rewards']:
            axes[0, 0].plot(self.metrics['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Coverage rates
        if self.metrics['coverage_rates']:
            axes[0, 1].plot(self.metrics['coverage_rates'])
            axes[0, 1].set_title('Coverage Rates')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Coverage Rate')
            axes[0, 1].grid(True)
        
        # Losses
        if self.metrics['actor_losses'] and self.metrics['critic_losses']:
            axes[1, 0].plot(self.metrics['actor_losses'], label='Actor Loss')
            axes[1, 0].plot(self.metrics['critic_losses'], label='Critic Loss')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Energy efficiency
        if self.metrics['energy_efficiency']:
            axes[1, 1].plot(self.metrics['energy_efficiency'])
            axes[1, 1].set_title('Energy Efficiency')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Efficiency')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_recent_average(self, metric: str, window: int = 10) -> float:
        """Get recent average of a metric"""
        if metric not in self.metrics or len(self.metrics[metric]) == 0:
            return 0.0
        
        recent_values = self.metrics[metric][-window:]
        return np.mean(recent_values)
    
    def get_smoothed_curve(self, metric: str, window: int = 10) -> List[float]:
        """Get smoothed version of metric curve"""
        if metric not in self.metrics or len(self.metrics[metric]) == 0:
            return []
        
        values = self.metrics[metric]
        smoothed = []
        
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i + 1]
            smoothed.append(np.mean(window_values))
        
        return smoothed


class EarlyStopping:
    """Early stopping utility for training"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 mode: str = 'max'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of episodes to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics that should increase, 'min' for decrease
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should be stopped
        
        Args:
            current_value: Current value of monitored metric
            
        Returns:
            early_stop: Whether to stop training
        """
        improved = False
        
        if self.mode == 'max':
            if current_value > self.best_value + self.min_delta:
                improved = True
                self.best_value = current_value
        else:  # mode == 'min'
            if current_value < self.best_value - self.min_delta:
                improved = True
                self.best_value = current_value
        
        if improved:
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class AdaptiveLearningRate:
    """Adaptive learning rate scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, mode: str = 'plateau',
                 factor: float = 0.5, patience: int = 10, 
                 min_lr: float = 1e-6):
        """
        Initialize adaptive learning rate scheduler
        
        Args:
            optimizer: PyTorch optimizer
            mode: Scheduling mode ('plateau', 'step', 'exponential')
            factor: Factor by which to reduce learning rate
            patience: Number of episodes to wait before reducing
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0
        self.best_value = float('-inf')
        
    def step(self, current_value: float):
        """Step the scheduler"""
        if self.mode == 'plateau':
            if current_value > self.best_value:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
                
                if self.counter >= self.patience:
                    self._reduce_lr()
                    self.counter = 0
    
    def _reduce_lr(self):
        """Reduce learning rate"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")


def print_training_summary(episode: int, metrics: Dict, window: int = 10):
    """Print formatted training summary"""
    print(f"\n{'='*60}")
    print(f"Episode {episode} Summary")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"{'='*60}\n")


def validate_config(config: Dict) -> Dict:
    """Validate and set default values for configuration"""
    # Set default values
    defaults = {
        'seed': 42,
        'device': 'auto',
        'num_episodes': 200,
        'eval_frequency': 10,
        'save_frequency': 50,
        'buffer_size': 100000,
        'batch_size': 32,
        'learning_starts': 1000,
        'gradient_clip': 1.0
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    # Validate critical parameters
    assert config['num_episodes'] > 0, "num_episodes must be positive"
    assert config['batch_size'] > 0, "batch_size must be positive"
    assert config['buffer_size'] > config['batch_size'], "buffer_size must be larger than batch_size"
    
    return config