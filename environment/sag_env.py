import numpy as np

class SAGEnvironment:
    def __init__(self, config):
        self.config = config
        self.area_size = config.area_size
        self.num_satellites = config.num_satellites
        self.num_uavs = config.num_uavs
        self.num_ground_stations = config.num_ground_stations
        self.num_pois = config.num_pois
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations

        self.satellite_range = config.satellite_range
        self.uav_range = config.uav_range
        self.ground_station_range = config.ground_station_range

        self.charging_stations = self._init_positions(config.num_charging_stations)
        self.obstacles = self._init_positions(config.num_obstacles)
        self.poi_priorities = np.random.randint(1, config.poi_priority_levels + 1, config.num_pois)

        self.satellite_speed = config.satellite_speed
        self.uav_speed = config.uav_speed
        self.ground_station_speed = config.ground_station_speed

        self.uav_energy_capacity = config.uav_energy_capacity
        self.uav_energy_consumption_rate = config.uav_energy_consumption_rate

        self.max_time_steps = config.max_time_steps
        self.individual_obs_dim = config.individual_obs_dim
        self.action_dim = config.action_dim

        self.agent_trajectories = {
            'satellites': [],
            'uavs': [],
            'ground_stations': []
        }

        self.total_reward = 0
        self.collision_count = 0

        self._cached_adj = None

    def reset(self):
        self.satellites = self._init_positions(self.num_satellites)
        self.uavs = self._init_positions(self.num_uavs)
        self.ground_stations = self._init_positions(self.num_ground_stations)
        self.pois = self._init_positions(self.num_pois)

        self.agent_energy = np.zeros(self.num_agents)
        self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs] = self.uav_energy_capacity

        self.time_step = 0
        self.previous_uav_positions = self.uavs.copy()

        self.agent_trajectories = {
            'satellites': [self.satellites.copy()],
            'uavs': [self.uavs.copy()],
            'ground_stations': [self.ground_stations.copy()]
        }

        self.total_reward = 0
        self.collision_count = 0
        self._cached_adj = None

        return self._get_obs()

    def step(self, actions):
        self.previous_uav_positions = self.uavs.copy()
        self._update_positions(actions)
        self._update_energy()
        self._recharge_uavs()
        self.time_step += 1

        self.agent_trajectories['satellites'].append(self.satellites.copy())
        self.agent_trajectories['uavs'].append(self.uavs.copy())
        self.agent_trajectories['ground_stations'].append(self.ground_stations.copy())

        self._cached_adj = None

        reward = self._compute_reward()
        self.total_reward += np.sum(reward)

        # Update collision count (you need to implement this based on your logic)
        self.collision_count += self._check_collisions()

        obs = self._get_obs()
        done = self._check_done()
        info = self._get_info()

        return obs, reward, done, info
    def _check_collisions(self):
        # Implement your collision detection logic here
        # This is just a placeholder
        return 0
    def _init_positions(self, num_entities):
        return np.random.uniform(0, self.area_size, (num_entities, 2))

    def _update_positions(self, actions):
        # Satellites move in a fixed orbit
        self.satellites += self.config.satellite_speed
        self.satellites %= self.area_size

        # UAVs move based on actions
        uav_actions = actions[self.num_satellites:self.num_satellites + self.num_uavs]

        # Ensure uav_actions is a 2D array
        if uav_actions.ndim == 1:
            uav_actions = uav_actions.reshape(self.num_uavs, -1)

        new_uav_positions = self.uavs + uav_actions * self.config.uav_speed
        for i, uav in enumerate(new_uav_positions):
            if self._is_valid_position(uav):
                self.uavs[i] = uav
        self.uavs = np.clip(self.uavs, 0, self.area_size)

        # Ground stations move slowly based on actions
        ground_actions = actions[self.num_satellites + self.num_uavs:]

        # Ensure ground_actions is a 2D array
        if ground_actions.ndim == 1:
            ground_actions = ground_actions.reshape(self.num_ground_stations, -1)

        self.ground_stations += ground_actions * self.config.ground_station_speed
        self.ground_stations = np.clip(self.ground_stations, 0, self.area_size)

    def _is_valid_position(self, position):
        for obstacle in self.obstacles:
            if np.linalg.norm(position - obstacle) < self.config.obstacle_size:
                return False
        return True

    def _update_energy(self):
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        movement = np.linalg.norm(self.uavs - self.previous_uav_positions, axis=1)
        energy_consumption = self.config.base_energy_consumption + self.config.movement_energy_consumption * movement
        uav_energy -= energy_consumption
        uav_energy = np.maximum(uav_energy, 0)
        self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs] = uav_energy

    def _recharge_uavs(self):
        for i, uav in enumerate(self.uavs):
            for charging_station in self.charging_stations:
                if np.linalg.norm(uav - charging_station) < self.config.charging_station_range:
                    self.agent_energy[self.num_satellites + i] = min(
                        self.agent_energy[self.num_satellites + i] + self.config.charging_rate,
                        self.config.uav_energy_capacity
                    )
                    break

    def _get_obs(self):
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        individual_obs = []

        for i, pos in enumerate(all_positions):
            distances = np.linalg.norm(self.pois - pos, axis=1)
            closest_poi = self.pois[np.argmin(distances)]
            closest_poi_priority = self.poi_priorities[np.argmin(distances)]

            obstacle_distances = np.linalg.norm(self.obstacles - pos, axis=1)
            closest_obstacle = self.obstacles[np.argmin(obstacle_distances)]

            charging_station_direction = self.charging_stations[0] - pos
            charging_station_distance = np.linalg.norm(charging_station_direction)
            if charging_station_distance > 0:
                charging_station_direction /= charging_station_distance
            else:
                charging_station_direction = np.zeros_like(charging_station_direction)

            obs = np.concatenate([
                pos,  # Own position (2)
                closest_poi,  # Closest POI position (2)
                [self.agent_energy[i]],  # Energy level (1)
                closest_obstacle,  # Closest obstacle position (2)
                [closest_poi_priority],  # POI priority (1)
                [np.dot(charging_station_direction, pos / (np.linalg.norm(pos) + 1e-8))]  # Charging station direction (1)
            ])
            individual_obs.append(obs)

        return np.array(individual_obs)

    def get_adj(self):
        if self._cached_adj is None:
            self._cached_adj = self._calculate_adj()
        return self._cached_adj.copy()

    def _calculate_adj(self):
        adj = np.zeros((self.num_agents, self.num_agents))
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        all_ranges = np.concatenate([
            np.full(self.num_satellites, self.satellite_range),
            np.full(self.num_uavs, self.uav_range),
            np.full(self.num_ground_stations, self.ground_station_range)
        ])

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = np.linalg.norm(all_positions[i] - all_positions[j])
                if distance <= max(all_ranges[i], all_ranges[j]):
                    adj[i, j] = adj[j, i] = 1

        return adj

    def _compute_reward(self):
        covered_pois = self._get_covered_pois()
        total_priorities = np.sum(self.poi_priorities)
        coverage_reward = np.sum(covered_pois * self.poi_priorities) / (total_priorities + 1e-8)

        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        energy_penalty = np.sum(self.config.uav_energy_capacity - uav_energy) / (self.num_uavs * self.config.uav_energy_capacity + 1e-8)

        collision_penalty = self._get_collision_penalty()

        reward = coverage_reward - 0.1 * energy_penalty - 0.5 * collision_penalty
        return np.full(self.num_agents, reward)

    def _get_collision_penalty(self):
        penalty = 0
        for uav in self.uavs:
            for obstacle in self.obstacles:
                if np.linalg.norm(uav - obstacle) < self.config.obstacle_size:
                    penalty += 1
        return penalty / self.num_uavs

    def _get_covered_pois(self):
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        all_ranges = np.concatenate([
            np.full(self.num_satellites, self.satellite_range),
            np.full(self.num_uavs, self.uav_range),
            np.full(self.num_ground_stations, self.ground_station_range)
        ])

        covered_pois = np.zeros(self.num_pois)
        for i, poi in enumerate(self.pois):
            distances = np.linalg.norm(all_positions - poi, axis=1)
            covered_pois[i] = np.any(distances <= all_ranges)

        return covered_pois

    def _check_done(self):
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        return self.time_step >= self.max_time_steps or np.all(uav_energy == 0)

    def _get_info(self):
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        return {
            'coverage': np.mean(self._get_covered_pois()),
            'uav_energy': np.mean(uav_energy),
            'time_step': self.time_step
        }

    def get_trajectories(self):
        return {
            'satellites': [pos.copy() for pos in self.agent_trajectories['satellites']],
            'uavs': [pos.copy() for pos in self.agent_trajectories['uavs']],
            'ground_stations': [pos.copy() for pos in self.agent_trajectories['ground_stations']]
        }