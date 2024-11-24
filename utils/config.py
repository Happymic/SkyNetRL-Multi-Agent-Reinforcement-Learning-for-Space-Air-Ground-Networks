import os
from datetime import datetime
import torch

'''
class Config:
    def __init__(self, mode='train'):
        # Mode setting
        self.mode = mode  # 'train' or 'test'

        # Reproducibility
        self.seed = 42

        # Environment settings
        self.area_size = 1000
        self.num_satellites = 1
        self.num_uavs = 5
        self.num_ground_stations = 5
        self.num_pois = 5
        self.num_obstacles = 3
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
        self.max_time_steps = 100

        # Agent settings
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2
        self.individual_obs_dim = 9
        self.hidden_dim = 256

        # MADDPG settings
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        self.gamma = 0.99
        self.tau = 0.01

        # Training settings
        self.num_episodes = 100 if mode == 'train' else 5  # Fewer episodes for testing
        self.batch_size = 16
        self.buffer_size = 1000
        self.log_frequency = 5
        self.eval_frequency = 20
        self.eval_episodes = 5
        self.save_frequency = 5

        # Memory management settings
        self.memory_cleanup_freq = 10
        self.gradient_accumulation_steps = 2

        # Exploration settings
        self.exploration_noise = 0.1
        self.exploration_decay = 0.995

        # Advanced features
        self.communication_range = 150
        self.poi_priority_levels = 5

        # Visualization settings
        self.visualize_frequency = 100
        self.real_time_visualization = True if mode == 'test' else False
        self.real_time_frequency = 10
        self.save_animation = True
        self.animation_fps = 10

        # Create timestamp and directories for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"./runs/{self.timestamp}"
        self.model_save_path = os.path.join(self.base_dir, "saved_models")
        self.visualization_dir = os.path.join(self.base_dir, "visualizations")

        # For testing mode, allow specifying a different model path
        self.model_load_path = None  # Will be set by test script

        # Ensure directories exist
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.scaler = torch.amp.GradScaler('cuda')  # Updated for deprecation warning


    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in vars(self).items())
'''

'''
class Config:
    def __init__(self, mode='train'):
        # Mode setting
        self.mode = mode  # 'train' or 'test'

        # Reproducibility
        self.seed = 42

        # Environment settings - Reduced for memory efficiency
        self.area_size = 500
        self.num_satellites = 1
        self.num_uavs = 3
        self.num_ground_stations = 3
        self.num_pois = 3
        self.num_obstacles = 2
        self.num_charging_stations = 3

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
        self.max_time_steps = 50

        # Agent settings
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2
        self.individual_obs_dim = 9
        self.hidden_dim = 128

        # MADDPG settings
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        self.gamma = 0.99
        self.tau = 0.01

        # Training settings
        self.num_episodes = 100 if mode == 'train' else 5
        self.batch_size = 8
        self.buffer_size = 500
        self.log_frequency = 5
        self.eval_frequency = 20
        self.eval_episodes = 3
        self.save_frequency = 10

        # Memory management settings
        self.memory_cleanup_freq = 5
        self.gradient_accumulation_steps = 4
        self.empty_cache_freq = 10

        # Exploration settings
        self.exploration_noise = 0.1
        self.exploration_decay = 0.995

        # Advanced features
        self.communication_range = 150
        self.poi_priority_levels = 3

        # Visualization settings
        self.visualize_frequency = 100
        self.real_time_visualization = True if mode == 'test' else False
        self.real_time_frequency = 10
        self.save_animation = False
        self.animation_fps = 10

        # GPU Memory Management
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # 明确指定使用第一个GPU

            # Set PyTorch memory optimization flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Initialize CUDA device
            torch.cuda.set_device(0)  # 使用第一个GPU
            torch.cuda.empty_cache()

            # Set memory fraction limit (60% of total memory)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_fraction = 0.6
            allocated_memory = int(total_memory * memory_fraction)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")

        # Mixed precision settings
        self.use_mixed_precision = True
        self.scaler = torch.amp.GradScaler(
            enabled=True,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )

        # Create timestamp and directories for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"./runs/{self.timestamp}"
        self.model_save_path = os.path.join(self.base_dir, "saved_models")
        self.visualization_dir = os.path.join(self.base_dir, "visualizations")
        self.model_load_path = None

        # Ensure directories exist
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)

    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in vars(self).items())

    def print_memory_stats(self):
        """打印当前内存使用情况"""
        if torch.cuda.is_available():
            print("\nGPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")
            props = torch.cuda.get_device_properties(self.device)
            print(f"Total GPU Memory: {props.total_memory / 1024 ** 2:.1f} MB")
'''

import os
import torch
from datetime import datetime


class Config:
    def __init__(self, mode='train'):
        # Mode setting
        self.mode = mode  # 'train' or 'test'

        # Reproducibility
        self.seed = 42

        # Environment settings - Balanced for training effect
        self.area_size = 600  # 略微增加以获得更好的训练场景
        self.num_satellites = 2  # 增加到2个以提供更好的覆盖
        self.num_uavs = 4  # 略微增加UAV数量
        self.num_ground_stations = 3  # 保持不变
        self.num_pois = 5  # 增加兴趣点以提供更多学习机会
        self.num_obstacles = 3  # 略微增加障碍物
        self.num_charging_stations = 4  # 与UAV数量匹配

        # Agent ranges - Adjusted for area size
        self.satellite_range = 250  # 增加覆盖范围
        self.uav_range = 120  # 略微增加UAV范围
        self.ground_station_range = 80  # 增加地面站范围
        self.charging_station_range = 100  # 增加充电站范围

        # Movement speeds - More balanced
        self.satellite_speed = 3  # 略微增加速度
        self.uav_speed = 6  # 增加UAV机动性
        self.ground_station_speed = 1  # 保持不变

        # UAV energy settings - Optimized
        self.uav_energy_capacity = 1200  # 增加能量容量
        self.uav_energy_consumption_rate = 0.4  # 降低消耗率
        self.base_energy_consumption = 0.15  # 降低基础消耗
        self.movement_energy_consumption = 0.25  # 降低移动消耗
        self.charging_rate = 60  # 增加充电速率

        # Obstacle settings
        self.obstacle_size = 25  # 略微增加障碍物大小

        # Simulation settings
        self.max_time_steps = 75  # 增加每个episode的步数

        # Agent settings
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2
        self.individual_obs_dim = 9
        self.hidden_dim = 256  # 增加网络容量

        # MADDPG settings - Optimized for stability
        self.actor_lr = 0.0003  # 略微增加学习率
        self.critic_lr = 0.001  # 调整critic学习率
        self.gamma = 0.99
        self.tau = 0.005  # 降低软更新率以提高稳定性

        # Training settings - Balanced for efficiency
        self.num_episodes = 300 if mode == 'train' else 5  # 增加训练轮数
        self.batch_size = 32  # 增加批次大小
        self.buffer_size = 10000  # 增加缓冲区大小
        self.log_frequency = 5
        self.eval_frequency = 20
        self.eval_episodes = 3
        self.save_frequency = 20  # 增加保存频率

        # Memory management settings - Enhanced
        self.memory_cleanup_freq = 10  # 增加清理频率
        self.gradient_accumulation_steps = 4
        self.empty_cache_freq = 15  # 增加清理频率

        # Exploration settings - More exploration
        self.exploration_noise = 0.15  # 增加探索噪声
        self.exploration_decay = 0.997  # 降低衰减速率

        # Advanced features
        self.communication_range = 200  # 增加通信范围
        self.poi_priority_levels = 3

        # Visualization settings
        self.visualize_frequency = 50  # 减少可视化频率
        self.real_time_visualization = True if mode == 'test' else False
        self.real_time_frequency = 10
        self.save_animation = False
        self.animation_fps = 10

        # GPU Memory Management
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_fraction = 0.7  # 略微增加内存使用比例
            allocated_memory = int(total_memory * memory_fraction)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")

        # Mixed precision settings
        self.use_mixed_precision = True
        self.scaler = torch.amp.GradScaler(
            enabled=True,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )

        # Directories and paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"./runs/{self.timestamp}"
        self.model_save_path = os.path.join(self.base_dir, "saved_models")
        self.visualization_dir = os.path.join(self.base_dir, "visualizations")
        self.model_load_path = None

        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)