import os
from datetime import datetime

class Config:
    def __init__(self):
        # Reproducibility
        self.seed = 42

        # Environment settings
        self.area_size = 50  # 减小区域大小
        self.num_satellites = 1  # 减少卫星数量
        self.num_uavs = 2  # 减少UAV数量
        self.num_ground_stations = 2  # 减少地面站数量
        self.num_pois = 10  # 减少兴趣点数量
        self.num_obstacles = 5  # 减少障碍物数量
        self.num_charging_stations = 1

        # Agent ranges
        self.satellite_range = 20
        self.uav_range = 10
        self.ground_station_range = 8
        self.charging_station_range = 5

        # Movement speeds
        self.satellite_speed = 0.5
        self.uav_speed = 1
        self.ground_station_speed = 0.2

        # UAV energy settings
        self.uav_energy_capacity = 50
        self.uav_energy_consumption_rate = 0.5
        self.base_energy_consumption = 0.2
        self.movement_energy_consumption = 0.05
        self.charging_rate = 1

        # Obstacle settings
        self.obstacle_size = 1

        # Simulation settings
        self.max_time_steps = 500  # 减少每个episode的最大步数

        # Agent settings
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2
        self.individual_obs_dim = 9
        self.hidden_dim = 64  # 减小隐藏层大小

        # MADDPG settings
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.gamma = 0.95
        self.tau = 0.01

        # Training settings
        self.num_episodes = 50  # 减少训练的episode数量
        self.batch_size = 64
        self.buffer_size = 10000
        self.log_frequency = 5
        self.eval_frequency = 10
        self.eval_episodes = 3
        self.save_frequency = 10

        # Exploration settings
        self.exploration_noise = 0.2
        self.exploration_decay = 0.99

        # Advanced features
        self.communication_range = 15
        self.poi_priority_levels = 2

        # Visualization settings
        self.visualize_frequency = 10
        self.real_time_visualization = False
        self.real_time_frequency = 5
        self.save_animation = True
        self.animation_fps = 5

        # Create timestamp and directories for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"./runs/{self.timestamp}"
        self.model_save_path = os.path.join(self.base_dir, "saved_models")
        self.visualization_dir = os.path.join(self.base_dir, "visualizations")

        # Ensure directories exist
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)

    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in vars(self).items())

# If you want to use the config, you can create an instance like this:
# config = Config()
# print(config)  # This will print all the configurations