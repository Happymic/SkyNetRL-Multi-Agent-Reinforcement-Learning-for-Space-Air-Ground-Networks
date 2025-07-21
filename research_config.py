import os
import torch
from datetime import datetime


class ResearchConfig:
    """Configuration optimized for generating research results and visualizations"""
    def __init__(self):
        # Mode setting
        self.mode = 'train'
        
        # Reproducibility
        self.seed = 42
        
        # Environment settings - Balanced for meaningful research results
        self.area_size = 800  # Larger area for more complex scenarios
        self.num_satellites = 2  # Multiple satellites for better coverage analysis
        self.num_uavs = 4  # Multiple UAVs to show coordination
        self.num_ground_stations = 4  # Ground infrastructure
        self.num_pois = 8  # More POIs for complex coverage scenarios
        self.num_obstacles = 4  # Obstacles to show navigation capabilities
        self.num_charging_stations = 4  # Adequate charging infrastructure
        
        # Agent ranges - Well-designed for paper figures
        self.satellite_range = 300  # Wide satellite coverage
        self.uav_range = 150  # UAV tactical range
        self.ground_station_range = 100  # Ground station range
        self.charging_station_range = 120  # Charging accessibility
        
        # Movement speeds - Realistic and observable
        self.satellite_speed = 4  # Faster satellite movement
        self.uav_speed = 8  # Agile UAV movement
        self.ground_station_speed = 2  # Mobile ground stations
        
        # UAV energy settings - For energy efficiency analysis
        self.uav_energy_capacity = 1500  # Higher capacity for longer missions
        self.uav_energy_consumption_rate = 0.3  # Efficient consumption
        self.base_energy_consumption = 0.1  # Low baseline consumption
        self.movement_energy_consumption = 0.2  # Movement cost
        self.charging_rate = 80  # Fast charging for research scenarios
        
        # Obstacle settings
        self.obstacle_size = 30  # Visible obstacles in visualizations
        
        # Simulation settings - Long enough for learning analysis
        self.max_time_steps = 300  # Longer episodes for detailed analysis
        
        # Agent settings
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2
        self.individual_obs_dim = 9
        self.hidden_dim = 256  # Larger networks for better performance
        
        # MADDPG settings - Optimized for stable learning
        self.actor_lr = 0.0002  # Conservative learning rate
        self.critic_lr = 0.0008  # Higher critic learning rate
        self.gamma = 0.99  # Standard discount factor
        self.tau = 0.005  # Soft update rate
        
        # Training settings - Sufficient for research analysis
        self.num_episodes = 50  # Enough episodes to show learning trends
        self.batch_size = 64  # Larger batch for stable gradients
        self.buffer_size = 20000  # Large buffer for experience diversity
        self.log_frequency = 2  # Frequent logging for detailed analysis
        self.eval_frequency = 5  # Regular evaluation
        self.eval_episodes = 3  # Multiple evaluation episodes
        self.save_frequency = 5  # Regular model saving
        
        # Memory management settings
        self.memory_cleanup_freq = 15
        self.gradient_accumulation_steps = 2
        self.empty_cache_freq = 10
        
        # Exploration settings - Balanced exploration
        self.exploration_noise = 0.2  # Higher initial exploration
        self.exploration_decay = 0.995  # Gradual decay
        
        # Advanced features
        self.communication_range = 250  # Wide communication range
        self.poi_priority_levels = 5  # Multiple priority levels for analysis
        
        # Visualization settings - FULL VISUALIZATION ENABLED
        self.visualize_frequency = 10  # Frequent visualization updates
        self.real_time_visualization = True  # Enable real-time visualization
        self.real_time_frequency = 5  # Update every 5 episodes
        self.save_animation = True  # Save animations for paper
        self.animation_fps = 15  # Smooth animations
        
        # Device settings
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            
            # Set memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_fraction = 0.8  # Use more memory for research
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        else:
            self.device = torch.device("cpu")
            print("Using CPU for training - consider GPU for faster results")
            
        # Mixed precision for efficiency
        self.use_mixed_precision = False  # Disable for stability
        self.scaler = torch.amp.GradScaler(enabled=False)
        
        # Research directories and paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"./runs/research_{self.timestamp}"
        self.model_save_path = os.path.join(self.base_dir, "saved_models")
        self.visualization_dir = os.path.join(self.base_dir, "visualizations")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.model_load_path = None
        
        # Create all necessary directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"\n=== Research Configuration ===")
        print(f"Area size: {self.area_size}x{self.area_size}")
        print(f"Agents: {self.num_agents} total ({self.num_satellites} satellites, {self.num_uavs} UAVs, {self.num_ground_stations} ground stations)")
        print(f"Episodes: {self.num_episodes}")
        print(f"Steps per episode: {self.max_time_steps}")
        print(f"Device: {self.device}")
        print(f"Results will be saved to: {self.base_dir}")
        print(f"Visualization enabled: {self.real_time_visualization}")
        print(f"Animation saving: {self.save_animation}")
        
    def __str__(self):
        """String representation of the configuration"""
        config_str = "\n=== SkyNetRL Research Configuration ===\n"
        config_str += f"Environment: {self.area_size}x{self.area_size} area\n"
        config_str += f"Agents: {self.num_agents} total\n"
        config_str += f"Training: {self.num_episodes} episodes\n"
        config_str += f"Device: {self.device}\n"
        config_str += f"Output: {self.base_dir}\n"
        return config_str