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

        self.satellite_speed = config.satellite_speed
        self.uav_speed = config.uav_speed
        self.ground_station_speed = config.ground_station_speed

        self.uav_energy_capacity = config.uav_energy_capacity
        self.uav_energy_consumption_rate = config.uav_energy_consumption_rate

        self.max_time_steps = config.max_time_steps
        self.individual_obs_dim = config.individual_obs_dim
        self.action_dim = config.action_dim

    def reset(self):
        self.satellites = self._init_positions(self.num_satellites)
        self.uavs = self._init_positions(self.num_uavs)
        self.ground_stations = self._init_positions(self.num_ground_stations)
        self.pois = self._init_positions(self.num_pois)

        self.agent_energy = np.zeros(self.num_agents)
        self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs] = self.uav_energy_capacity

        self.time_step = 0

        return self._get_obs()

    def step(self, actions):
        assert actions.shape == (self.num_agents,
                                 self.action_dim), f"Action shape mismatch. Expected {(self.num_agents, self.action_dim)}, got {actions.shape}"

        self._update_positions(actions)
        self._update_energy()
        self.time_step += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        return obs, reward, done, info
    def _init_positions(self, num_entities):
        return np.random.uniform(0, self.area_size, (num_entities, 2))

    def _update_positions(self, actions):
        # Satellites move in a fixed orbit
        self.satellites += self.satellite_speed
        self.satellites %= self.area_size

        # UAVs move based on actions
        uav_actions = actions[self.num_satellites:self.num_satellites + self.num_uavs]
        self.uavs += uav_actions * self.uav_speed
        self.uavs = np.clip(self.uavs, 0, self.area_size)

        # Ground stations move slowly based on actions
        ground_actions = actions[self.num_satellites + self.num_uavs:]
        self.ground_stations += ground_actions * self.ground_station_speed
        self.ground_stations = np.clip(self.ground_stations, 0, self.area_size)

    def _update_energy(self):
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        uav_energy -= self.uav_energy_consumption_rate
        uav_energy = np.maximum(uav_energy, 0)
        self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs] = uav_energy

    def _get_obs(self):
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        individual_obs = []

        for i, pos in enumerate(all_positions):
            distances = np.linalg.norm(self.pois - pos, axis=1)
            closest_poi = self.pois[np.argmin(distances)]

            obs = np.concatenate([
                pos,  # Own position (2)
                closest_poi,  # Closest POI position (2)
                [self.agent_energy[i]]  # Energy level (1)
            ])
            individual_obs.append(obs)

        obs = np.array(individual_obs)
        assert obs.shape == (self.num_agents,
                             self.individual_obs_dim), f"Observation shape mismatch. Expected {(self.num_agents, self.individual_obs_dim)}, got {obs.shape}"
        return obs

    def _get_adj(self):
        adj = np.zeros((self.num_agents, self.num_agents))
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    distance = np.linalg.norm(all_positions[i] - all_positions[j])
                    if i < self.num_satellites:
                        adj[i, j] = distance <= self.satellite_range
                    elif i < self.num_satellites + self.num_uavs:
                        adj[i, j] = distance <= self.uav_range
                    else:
                        adj[i, j] = distance <= self.ground_station_range

        return adj

    def _compute_reward(self):
        covered_pois = self._get_covered_pois()
        coverage_reward = np.sum(covered_pois) / self.num_pois

        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        energy_penalty = np.sum(self.uav_energy_capacity - uav_energy) / (self.num_uavs * self.uav_energy_capacity)

        reward = coverage_reward - 0.1 * energy_penalty
        return np.full(self.num_agents, reward)  # Each agent receives the same global reward

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