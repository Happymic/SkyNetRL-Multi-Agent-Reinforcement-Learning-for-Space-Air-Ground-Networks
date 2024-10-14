import os
from datetime import datetime

class Config:
    def __init__(self):
        # Reproducibility
        self.seed = 42

        # Environment settings
        self.area_size = 1000  # 减小区域大小
        self.num_satellites = 1  # 减少卫星数量
        self.num_uavs = 5  # 减少UAV数量
        self.num_ground_stations = 5  # 减少地面站数量
        self.num_pois = 5  # 减少兴趣点数量
        self.num_obstacles = 3  # 减少障碍物数量
        self.num_charging_stations = 5

        # Agent ranges
        self.satellite_range = 200
        self.uav_range = 100
        self.ground_station_range = 50
        self.charging_station_range = 75

        # Movement speeds
        self.satellite_speed = 2
        self.uav_speed = 5
        self.ground_station_speed = 1

        # UAV energy settings
        self.uav_energy_capacity = 1000
        self.uav_energy_consumption_rate = 0.5
        self.base_energy_consumption = 0.2
        self.movement_energy_consumption = 0.3
        self.charging_rate = 50

        # Obstacle settings
        self.obstacle_size = 20

        # Simulation settings
        self.max_time_steps = 100  # 减少每个episode的最大步数

        # Agent settings
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2
        self.individual_obs_dim = 9
        self.hidden_dim = 256  # 减小隐藏层大小

        # MADDPG settings
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        self.gamma = 0.99
        self.tau = 0.01

        # Training settings
        self.num_episodes = 10  # 增加训练的episode数量，以充分利用6小时训练时间
        self.batch_size = 64    # 增加批次大小，充分利用GPU
        self.buffer_size = 200 # 略微减小缓冲区大小，以适应32GB RAM
        self.log_frequency = 5   # 保持不变，每10个episode记录一次
        self.eval_frequency = 5  # 更频繁地进行评估
        self.eval_episodes = 5    # 保持不变
        self.save_frequency = 5

        # Exploration settings
        self.exploration_noise = 0.3
        self.exploration_decay = 0.9999

        # Advanced features
        self.communication_range = 150
        self.poi_priority_levels = 5

        # Visualization settings
        self.visualize_frequency = 100
        self.real_time_visualization = False
        self.real_time_frequency = 10
        self.save_animation = True
        self.animation_fps = 10

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