class Config:
    def __init__(self):
        # Environment settings
        self.area_size = 100
        self.num_satellites = 2
        self.num_uavs = 5
        self.num_ground_stations = 5
        self.num_pois = 20

        self.satellite_range = 30
        self.uav_range = 15
        self.ground_station_range = 10

        self.satellite_speed = 1
        self.uav_speed = 2
        self.ground_station_speed = 0.5

        self.uav_energy_capacity = 100
        self.uav_energy_consumption_rate = 1

        self.max_time_steps = 200

        # Agent settings
        self.hidden_dim = 128
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        self.action_dim = 2  # Assuming 2D actions for each agent
        self.individual_obs_dim = 5  # agent's own position (2) + closest POI position (2) + energy level (1)

        # MADDPG settings
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.gamma = 0.95
        self.tau = 0.01

        # Training settings
        self.num_episodes = 1000  # Reduced for faster initial training
        self.batch_size = 64
        self.buffer_size = 10000
        self.log_frequency = 10
        self.eval_frequency = 100
        self.eval_episodes = 5

        # Visualization settings
        self.visualize_frequency = 100

        # Model save settings
        self.save_frequency = 100
        self.model_save_path = "./saved_models/"