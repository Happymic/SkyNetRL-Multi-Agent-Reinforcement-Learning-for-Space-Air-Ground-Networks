import os
import torch
from datetime import datetime


class QuickTestConfig:
    """Minimal configuration for quick testing"""
    def __init__(self):
        # Mode setting
        self.mode = 'train'
        
        # Reproducibility
        self.seed = 42
        
        # Minimal environment settings for fastest training
        self.area_size = 300
        self.num_satellites = 1
        self.num_uavs = 2
        self.num_ground_stations = 2
        self.num_pois = 3
        self.num_obstacles = 1
        self.num_charging_stations = 2
        
        # Agent ranges
        self.satellite_range = 150
        self.uav_range = 80
        self.ground_station_range = 50
        self.charging_station_range = 75
        
        # Movement speeds
        self.satellite_speed = 2
        self.uav_speed = 5
        self.ground_station_speed = 1
        
        # UAV energy settings
        self.uav_energy_capacity = 1000
        self.uav_energy_consumption_rate = 0.3
        self.base_energy_consumption = 0.1
        self.movement_energy_consumption = 0.2
        self.charging_rate = 80
        
        # Obstacle settings
        self.obstacle_size = 20
        
        # Simulation settings
        self.max_time_steps = 50  # Very short episodes
        
        # Agent settings
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2
        self.individual_obs_dim = 9
        self.hidden_dim = 128  # Smaller network
        
        # MADDPG settings
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.gamma = 0.95
        self.tau = 0.01
        
        # Minimal training settings
        self.num_episodes = 5  # Only 5 episodes for quick test
        self.batch_size = 16
        self.buffer_size = 1000
        self.log_frequency = 1
        self.eval_frequency = 2
        self.eval_episodes = 1
        self.save_frequency = 5
        
        # Memory management
        self.memory_cleanup_freq = 5
        self.gradient_accumulation_steps = 1
        self.empty_cache_freq = 5
        
        # Exploration settings
        self.exploration_noise = 0.1
        self.exploration_decay = 0.99
        
        # Advanced features
        self.communication_range = 150
        self.poi_priority_levels = 3
        
        # Disable visualization for speed
        self.visualize_frequency = 1000  # Basically never
        self.real_time_visualization = False
        self.real_time_frequency = 10
        self.save_animation = False
        self.animation_fps = 10
        
        # Device settings
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
            
        # Mixed precision
        self.use_mixed_precision = False  # Disable for simplicity
        self.scaler = torch.amp.GradScaler(enabled=False)
        
        # Directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"./runs/quick_test_{self.timestamp}"
        self.model_save_path = os.path.join(self.base_dir, "saved_models")
        self.visualization_dir = os.path.join(self.base_dir, "visualizations")
        self.model_load_path = None
        
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)