"""
Advanced Training Stability Improvements for Multi-Agent Reinforcement Learning
Implements sophisticated techniques to improve training stability and convergence
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, defaultdict
import copy
import math
from dataclasses import dataclass, field
import warnings


@dataclass
class StabilityConfig:
    """Configuration for training stability improvements"""
    
    # Gradient management
    gradient_clipping_enabled: bool = True
    max_grad_norm: float = 1.0
    gradient_penalty_weight: float = 0.0
    
    # Learning rate scheduling
    lr_scheduling_enabled: bool = True
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "exponential"
    lr_decay_factor: float = 0.5
    lr_patience: int = 100
    lr_min: float = 1e-6
    
    # Experience replay improvements
    prioritized_replay: bool = True
    importance_sampling_beta: float = 0.4
    importance_sampling_beta_end: float = 1.0
    importance_sampling_decay_steps: int = 100000
    
    # Training stabilization
    target_network_update_freq: int = 100
    soft_target_update_tau: float = 0.005
    noise_scheduling: bool = True
    noise_decay_rate: float = 0.99
    
    # Early stopping and convergence
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 500
    convergence_threshold: float = 1e-4
    
    # Regularization
    weight_decay: float = 1e-5
    dropout_scheduling: bool = True
    initial_dropout: float = 0.2
    final_dropout: float = 0.05
    
    # Loss stabilization
    loss_clipping_enabled: bool = True
    max_loss_value: float = 100.0
    loss_smoothing_window: int = 100


class AdaptiveGradientClipper:
    """Adaptive gradient clipping that adjusts based on gradient statistics"""
    
    def __init__(self, initial_max_norm: float = 1.0, adaptation_rate: float = 0.01):
        self.max_norm = initial_max_norm
        self.adaptation_rate = adaptation_rate
        self.gradient_norms = deque(maxlen=1000)
        self.target_percentile = 75  # Clip at 75th percentile of recent gradients
        
    def clip_gradients(self, parameters):
        """Clip gradients with adaptive threshold"""
        
        # Calculate current gradient norm
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        
        # Store gradient norm for adaptation
        self.gradient_norms.append(total_norm)
        
        # Adapt clipping threshold based on recent gradient statistics
        if len(self.gradient_norms) >= 50:
            percentile_norm = np.percentile(self.gradient_norms, self.target_percentile)
            # Exponential moving average for smooth adaptation
            self.max_norm = (1 - self.adaptation_rate) * self.max_norm + \
                           self.adaptation_rate * percentile_norm
            self.max_norm = max(self.max_norm, 0.1)  # Minimum threshold
        
        # Apply gradient clipping
        if total_norm > self.max_norm:
            clip_coef = self.max_norm / (total_norm + 1e-8)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm


class AdvancedLearningRateScheduler:
    """Advanced learning rate scheduling with multiple strategies"""
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        scheduler_type: str = "reduce_on_plateau",
        **kwargs
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Create scheduler based on type
        if scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('lr_decay_factor', 0.5),
                patience=kwargs.get('lr_patience', 100),
                min_lr=kwargs.get('lr_min', 1e-6),
                verbose=True
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('restart_period', 1000),
                T_mult=kwargs.get('restart_multiplier', 2),
                eta_min=kwargs.get('lr_min', 1e-6)
            )
        elif scheduler_type == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.999)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Warmup settings
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        self.warmup_enabled = self.warmup_steps > 0
        self.current_step = 0
        
    def step(self, metric: Optional[float] = None):
        """Step the learning rate scheduler"""
        self.current_step += 1
        
        # Apply warmup if enabled
        if self.warmup_enabled and self.current_step <= self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.initial_lrs[i] * warmup_factor
            return
        
        # Apply main scheduling
        if self.scheduler_type == "reduce_on_plateau":
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_current_lr(self) -> List[float]:
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]


class PrioritizedReplayBuffer:
    """Prioritized experience replay with importance sampling"""
    
    def __init__(
        self, 
        capacity: int, 
        alpha: float = 0.6, 
        beta: float = 0.4,
        beta_end: float = 1.0,
        beta_decay_steps: int = 100000
    ):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_end = beta_end
        self.beta_decay_steps = beta_decay_steps
        self.beta_step = 0
        
        # Storage
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        
    def push(self, experience, priority: Optional[float] = None):
        """Add experience with priority"""
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = priority
        self.max_priority = max(self.max_priority, priority)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Sample experiences
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def update_beta(self):
        """Update importance sampling coefficient"""
        self.beta_step += 1
        progress = min(1.0, self.beta_step / self.beta_decay_steps)
        self.beta = self.beta + progress * (self.beta_end - self.beta)
    
    def __len__(self):
        return len(self.buffer)


class TrainingStabilizer:
    """Main class for training stability improvements"""
    
    def __init__(self, config: StabilityConfig):
        self.config = config
        
        # Gradient management
        self.gradient_clipper = AdaptiveGradientClipper(
            initial_max_norm=config.max_grad_norm
        ) if config.gradient_clipping_enabled else None
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': deque(maxlen=config.loss_smoothing_window),
            'losses': deque(maxlen=config.loss_smoothing_window),
            'gradient_norms': deque(maxlen=config.loss_smoothing_window),
            'learning_rates': deque(maxlen=config.loss_smoothing_window)
        }
        
        # Early stopping
        self.best_performance = float('-inf')
        self.patience_counter = 0
        self.should_stop_early = False
        
        # Noise scheduling
        self.current_noise_level = 1.0
        
        # Loss tracking for stability
        self.recent_losses = deque(maxlen=config.loss_smoothing_window)
        
    def setup_optimizers_and_schedulers(
        self, 
        networks: Dict[str, nn.Module],
        learning_rates: Dict[str, float]
    ) -> Tuple[Dict[str, optim.Optimizer], Dict[str, AdvancedLearningRateScheduler]]:
        """Setup optimizers and learning rate schedulers"""
        
        optimizers = {}
        schedulers = {}
        
        for name, network in networks.items():
            # Create optimizer with weight decay
            optimizer = optim.Adam(
                network.parameters(),
                lr=learning_rates.get(name, 3e-4),
                weight_decay=self.config.weight_decay,
                eps=1e-8
            )
            optimizers[name] = optimizer
            
            # Create scheduler if enabled
            if self.config.lr_scheduling_enabled:
                scheduler = AdvancedLearningRateScheduler(
                    optimizer,
                    scheduler_type=self.config.scheduler_type,
                    lr_decay_factor=self.config.lr_decay_factor,
                    lr_patience=self.config.lr_patience,
                    lr_min=self.config.lr_min
                )
                schedulers[name] = scheduler
        
        return optimizers, schedulers
    
    def stabilize_gradients(
        self, 
        networks: Dict[str, nn.Module], 
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Apply gradient stabilization techniques"""
        
        gradient_info = {}
        
        for name, network in networks.items():
            if name in losses:
                loss = losses[name]
                
                # Clip loss if enabled
                if self.config.loss_clipping_enabled:
                    loss = torch.clamp(loss, -self.config.max_loss_value, self.config.max_loss_value)
                
                # Backward pass
                loss.backward(retain_graph=True)
                
                # Gradient clipping
                if self.gradient_clipper is not None:
                    grad_norm = self.gradient_clipper.clip_gradients(network.parameters())
                    gradient_info[f'{name}_grad_norm'] = grad_norm
                
                # Record loss
                self.recent_losses.append(loss.item())
        
        return gradient_info
    
    def update_schedulers(
        self, 
        schedulers: Dict[str, AdvancedLearningRateScheduler],
        performance_metric: Optional[float] = None
    ):
        """Update learning rate schedulers"""
        
        for scheduler in schedulers.values():
            scheduler.step(performance_metric)
    
    def update_noise_schedule(self, episode: int):
        """Update exploration noise schedule"""
        if self.config.noise_scheduling:
            self.current_noise_level *= self.config.noise_decay_rate
            self.current_noise_level = max(0.01, self.current_noise_level)  # Minimum noise
    
    def check_early_stopping(self, current_performance: float) -> bool:
        """Check if training should stop early"""
        
        if not self.config.early_stopping_enabled:
            return False
        
        if current_performance > self.best_performance + self.config.convergence_threshold:
            self.best_performance = current_performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.early_stopping_patience:
            self.should_stop_early = True
            return True
        
        return False
    
    def update_statistics(
        self, 
        episode_reward: float,
        losses: Dict[str, float],
        gradient_info: Dict[str, float],
        learning_rates: Dict[str, float]
    ):
        """Update training statistics"""
        
        self.training_stats['episode_rewards'].append(episode_reward)
        
        if losses:
            avg_loss = np.mean(list(losses.values()))
            self.training_stats['losses'].append(avg_loss)
        
        if gradient_info:
            avg_grad_norm = np.mean([v for k, v in gradient_info.items() if 'grad_norm' in k])
            self.training_stats['gradient_norms'].append(avg_grad_norm)
        
        if learning_rates:
            avg_lr = np.mean(list(learning_rates.values()))
            self.training_stats['learning_rates'].append(avg_lr)
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get comprehensive stability metrics"""
        
        metrics = {}
        
        # Performance metrics
        if self.training_stats['episode_rewards']:
            rewards = list(self.training_stats['episode_rewards'])
            metrics['reward_mean'] = np.mean(rewards)
            metrics['reward_std'] = np.std(rewards)
            metrics['reward_trend'] = self._calculate_trend(rewards)
        
        # Loss metrics
        if self.training_stats['losses']:
            losses = list(self.training_stats['losses'])
            metrics['loss_mean'] = np.mean(losses)
            metrics['loss_std'] = np.std(losses)
            metrics['loss_trend'] = self._calculate_trend(losses)
        
        # Gradient metrics
        if self.training_stats['gradient_norms']:
            grad_norms = list(self.training_stats['gradient_norms'])
            metrics['grad_norm_mean'] = np.mean(grad_norms)
            metrics['grad_norm_std'] = np.std(grad_norms)
            
        # Learning rate metrics
        if self.training_stats['learning_rates']:
            lrs = list(self.training_stats['learning_rates'])
            metrics['lr_current'] = lrs[-1] if lrs else 0.0
            metrics['lr_change'] = (lrs[-1] - lrs[0]) / lrs[0] if len(lrs) > 1 and lrs[0] > 0 else 0.0
        
        # Stability indicators
        metrics['training_stable'] = self._is_training_stable()
        metrics['early_stopping_triggered'] = self.should_stop_early
        metrics['current_noise_level'] = self.current_noise_level
        metrics['patience_counter'] = self.patience_counter
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values (positive = improving)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope
    
    def _is_training_stable(self) -> bool:
        """Determine if training appears stable"""
        
        if len(self.recent_losses) < 20:
            return False
        
        recent_losses = list(self.recent_losses)
        
        # Check for exploding losses
        if any(abs(loss) > self.config.max_loss_value for loss in recent_losses[-10:]):
            return False
        
        # Check for excessive variance
        recent_std = np.std(recent_losses[-20:])
        recent_mean = np.mean(recent_losses[-20:])
        if recent_mean != 0 and recent_std / abs(recent_mean) > 2.0:  # CV > 200%
            return False
        
        # Check for NaN or inf
        if any(not np.isfinite(loss) for loss in recent_losses[-10:]):
            return False
        
        return True


class DynamicDropoutScheduler:
    """Dynamic dropout scheduling during training"""
    
    def __init__(
        self, 
        networks: Dict[str, nn.Module],
        initial_dropout: float = 0.2,
        final_dropout: float = 0.05,
        total_episodes: int = 10000
    ):
        self.networks = networks
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.total_episodes = total_episodes
        
        # Store original dropout modules
        self.dropout_modules = {}
        for name, network in networks.items():
            self.dropout_modules[name] = []
            for module in network.modules():
                if isinstance(module, nn.Dropout):
                    self.dropout_modules[name].append(module)
    
    def update_dropout(self, episode: int):
        """Update dropout rates based on training progress"""
        
        progress = min(1.0, episode / self.total_episodes)
        current_dropout = self.initial_dropout + progress * (self.final_dropout - self.initial_dropout)
        
        for name, modules in self.dropout_modules.items():
            for module in modules:
                module.p = current_dropout


def create_stability_system(config: Dict) -> TrainingStabilizer:
    """Factory function to create training stability system"""
    
    stability_config = StabilityConfig(**config.get('stability', {}))
    return TrainingStabilizer(stability_config)


class AdvancedTargetNetworkUpdater:
    """Advanced target network updating with multiple strategies"""
    
    def __init__(
        self, 
        update_strategy: str = "soft",  # "hard", "soft", "adaptive"
        update_frequency: int = 100,
        tau: float = 0.005,
        performance_threshold: float = 0.1
    ):
        self.update_strategy = update_strategy
        self.update_frequency = update_frequency
        self.tau = tau
        self.performance_threshold = performance_threshold
        self.step_count = 0
        self.performance_history = deque(maxlen=100)
    
    def update_target_networks(
        self, 
        online_networks: Dict[str, nn.Module],
        target_networks: Dict[str, nn.Module],
        current_performance: Optional[float] = None
    ):
        """Update target networks based on strategy"""
        
        self.step_count += 1
        if current_performance is not None:
            self.performance_history.append(current_performance)
        
        if self.update_strategy == "hard":
            if self.step_count % self.update_frequency == 0:
                self._hard_update(online_networks, target_networks)
        
        elif self.update_strategy == "soft":
            self._soft_update(online_networks, target_networks)
        
        elif self.update_strategy == "adaptive":
            self._adaptive_update(online_networks, target_networks, current_performance)
    
    def _hard_update(
        self, 
        online_networks: Dict[str, nn.Module],
        target_networks: Dict[str, nn.Module]
    ):
        """Hard update - copy parameters completely"""
        for name in online_networks.keys():
            if name in target_networks:
                target_networks[name].load_state_dict(online_networks[name].state_dict())
    
    def _soft_update(
        self, 
        online_networks: Dict[str, nn.Module],
        target_networks: Dict[str, nn.Module]
    ):
        """Soft update - exponential moving average"""
        for name in online_networks.keys():
            if name in target_networks:
                for target_param, online_param in zip(
                    target_networks[name].parameters(),
                    online_networks[name].parameters()
                ):
                    target_param.data.copy_(
                        self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                    )
    
    def _adaptive_update(
        self, 
        online_networks: Dict[str, nn.Module],
        target_networks: Dict[str, nn.Module],
        current_performance: Optional[float]
    ):
        """Adaptive update based on performance"""
        
        if current_performance is None or len(self.performance_history) < 10:
            # Fallback to soft update
            self._soft_update(online_networks, target_networks)
            return
        
        # Check performance improvement
        recent_performance = np.mean(list(self.performance_history)[-5:])
        older_performance = np.mean(list(self.performance_history)[-10:-5])
        
        performance_improvement = recent_performance - older_performance
        
        if performance_improvement > self.performance_threshold:
            # Good performance - use more aggressive updates
            adaptive_tau = min(0.02, self.tau * 2)
        elif performance_improvement < -self.performance_threshold:
            # Poor performance - use conservative updates
            adaptive_tau = max(0.001, self.tau * 0.5)
        else:
            # Stable performance - use normal updates
            adaptive_tau = self.tau
        
        # Apply soft update with adaptive tau
        for name in online_networks.keys():
            if name in target_networks:
                for target_param, online_param in zip(
                    target_networks[name].parameters(),
                    online_networks[name].parameters()
                ):
                    target_param.data.copy_(
                        adaptive_tau * online_param.data + (1.0 - adaptive_tau) * target_param.data
                    )