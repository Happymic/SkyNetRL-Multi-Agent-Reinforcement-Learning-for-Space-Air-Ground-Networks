import numpy as np

class SAGEnvironment:
    def __init__(self, config):
        """初始化空天地一体化网络环境"""
        self.config = config

        # 环境基础参数
        self.area_size = config.area_size
        self.num_satellites = config.num_satellites
        self.num_uavs = config.num_uavs
        self.num_ground_stations = config.num_ground_stations
        self.num_pois = config.num_pois
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations

        # 范围设置
        self.satellite_range = config.satellite_range
        self.uav_range = config.uav_range
        self.ground_station_range = config.ground_station_range
        self.charging_station_range = config.charging_station_range

        # 能量设置
        self.uav_energy_capacity = config.uav_energy_capacity
        self.base_energy_consumption = config.base_energy_consumption
        self.movement_energy_consumption = config.movement_energy_consumption
        self.charging_rate = config.charging_rate

        # 运动设置
        self.satellite_speed = config.satellite_speed
        self.uav_speed = config.uav_speed
        self.ground_station_speed = config.ground_station_speed

        # 障碍物设置
        self.obstacle_size = config.obstacle_size

        # 初始化环境组件
        self.charging_stations = None
        self.obstacles = None
        self.pois = None
        self.poi_priorities = None
        self.satellites = None
        self.uavs = None
        self.ground_stations = None
        self.agent_energy = None

        # 状态记录
        self.time_step = 0
        self.max_time_steps = config.max_time_steps
        self.previous_uav_positions = None
        self.collision_count = 0
        self.total_reward = 0

        # 轨迹记录
        self.agent_trajectories = {
            'satellites': [],
            'uavs': [],
            'ground_stations': []
        }

        # 指标记录
        self.episode_data = {
            'coverage': [],
            'energy': [],
            'collisions': 0,
            'charging_events': 0,
            'communication_events': 0
        }

        self.observation_dim = 9  # 总共9维特征
        self.action_dim = 2  # x,y两个方向的动作

    def reset(self):
        """重置环境状态"""
        # 初始化固定位置的组件
        self.charging_stations = self._init_positions(self.config.num_charging_stations)
        self.obstacles = self._init_positions(self.config.num_obstacles)
        self.pois = self._init_positions(self.config.num_pois)
        self.poi_priorities = np.random.randint(1, self.config.poi_priority_levels + 1, self.config.num_pois)

        # 初始化移动智能体
        self.satellites = self._init_positions(self.num_satellites)
        self.uavs = self._init_positions(self.num_uavs)
        self.ground_stations = self._init_positions(self.num_ground_stations)

        # 初始化能量
        self.agent_energy = np.zeros(self.num_agents)
        self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs] = self.uav_energy_capacity

        # 重置状态记录
        self.time_step = 0
        self.previous_uav_positions = self.uavs.copy()
        self.collision_count = 0
        self.total_reward = 0

        # 重置轨迹记录
        self.agent_trajectories = {
            'satellites': [self.satellites.copy()],
            'uavs': [self.uavs.copy()],
            'ground_stations': [self.ground_stations.copy()]
        }

        # 重置指标记录
        self.episode_data = {
            'coverage': [],
            'energy': [],
            'collisions': 0,
            'charging_events': 0,
            'communication_events': 0
        }

        return self._get_obs()

    def step(self, actions):
        """环境步进"""
        self.time_step += 1
        self.previous_uav_positions = self.uavs.copy()

        # 更新位置
        self._update_positions(actions)

        # 更新能量
        self._update_energy()

        # 检查充电
        self._recharge_uavs()

        # 记录轨迹
        self.agent_trajectories['satellites'].append(self.satellites.copy())
        self.agent_trajectories['uavs'].append(self.uavs.copy())
        self.agent_trajectories['ground_stations'].append(self.ground_stations.copy())

        # 计算奖励
        reward = self._compute_reward()
        self.total_reward += np.sum(reward)

        # 检查碰撞
        current_collisions = self._check_collisions()
        self.collision_count += current_collisions

        # 更新状态记录
        self.episode_data['coverage'].append(self._get_covered_pois())
        self.episode_data['energy'].append(self.agent_energy.copy())
        self.episode_data['collisions'] = self.collision_count

        # 获取观察、完成状态和信息
        obs = self._get_obs()
        done = self._check_done()
        info = self._get_info()

        return obs, reward, done, info

    def _init_positions(self, num_entities):
        """初始化实体位置"""
        return np.random.uniform(0, self.area_size, (num_entities, 2))

    def _update_positions(self, actions):
        """更新所有智能体的位置"""
        # 更新卫星位置 (轨道运动)
        self.satellites += self.satellite_speed
        self.satellites %= self.area_size

        # 更新UAV位置
        uav_actions = actions[self.num_satellites:self.num_satellites + self.num_uavs]
        if uav_actions.ndim == 1:
            uav_actions = uav_actions.reshape(-1, 2)
        new_uav_positions = self.uavs + uav_actions * self.uav_speed

        # 检查UAV新位置的有效性并更新
        for i, uav in enumerate(new_uav_positions):
            if self._is_valid_position(uav):
                self.uavs[i] = uav
        self.uavs = np.clip(self.uavs, 0, self.area_size)

        # 更新地面站位置
        ground_actions = actions[self.num_satellites + self.num_uavs:]
        if ground_actions.ndim == 1:
            ground_actions = ground_actions.reshape(-1, 2)
        self.ground_stations += ground_actions * self.ground_station_speed
        self.ground_stations = np.clip(self.ground_stations, 0, self.area_size)

    def _is_valid_position(self, position):
        """检查位置是否有效（不与障碍物碰撞）"""
        for obstacle in self.obstacles:
            if np.linalg.norm(position - obstacle) < self.obstacle_size:
                return False
        return True

    def _update_energy(self):
        """更新UAV能量"""
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        movement = np.linalg.norm(self.uavs - self.previous_uav_positions, axis=1)
        energy_consumption = self.base_energy_consumption + self.movement_energy_consumption * movement
        uav_energy -= energy_consumption
        uav_energy = np.maximum(uav_energy, 0)
        self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs] = uav_energy

    def _recharge_uavs(self):
        """为靠近充电站的UAV充电"""
        charging_events = 0
        for i, uav in enumerate(self.uavs):
            for station in self.charging_stations:
                if np.linalg.norm(uav - station) < self.charging_station_range:
                    self.agent_energy[self.num_satellites + i] = min(
                        self.agent_energy[self.num_satellites + i] + self.charging_rate,
                        self.uav_energy_capacity
                    )
                    charging_events += 1
                    break
        self.episode_data['charging_events'] += charging_events

    def _compute_reward(self):
        """计算奖励"""
        # 计算覆盖奖励
        covered_pois = self._get_covered_pois()
        total_priorities = np.sum(self.poi_priorities)
        coverage_reward = 2 * np.sum(covered_pois * self.poi_priorities) / (total_priorities + 1e-8)

        # 计算能量惩罚
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        energy_penalty = 0.05 * np.sum(self.uav_energy_capacity - uav_energy) / \
                        (self.num_uavs * self.uav_energy_capacity + 1e-8)

        # 计算碰撞惩罚
        collision_penalty = 0.3 * np.log1p(self._check_collisions())

        # 计算时间因子
        time_factor = 1 + 0.001 * self.time_step

        # 计算任务完成奖励
        task_completion_reward = 0.5 * np.sum(covered_pois *
                                            (self.poi_priorities == self.config.poi_priority_levels))

        # 总奖励
        reward = (coverage_reward + task_completion_reward - energy_penalty - collision_penalty) * time_factor
        return np.full(self.num_agents, reward)

    def _check_collisions(self):
        """检查碰撞"""
        collisions = 0
        # UAV与障碍物碰撞
        for uav in self.uavs:
            for obstacle in self.obstacles:
                if np.linalg.norm(uav - obstacle) < self.obstacle_size:
                    collisions += 1
        # UAV之间的碰撞
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                if np.linalg.norm(self.uavs[i] - self.uavs[j]) < self.obstacle_size:
                    collisions += 1
        return collisions

    def _get_covered_pois(self):
        """计算POI的覆盖情况"""
        covered = np.zeros(self.num_pois)
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        all_ranges = np.concatenate([
            np.full(self.num_satellites, self.satellite_range),
            np.full(self.num_uavs, self.uav_range),
            np.full(self.num_ground_stations, self.ground_station_range)
        ])

        for i, poi in enumerate(self.pois):
            distances = np.linalg.norm(all_positions - poi, axis=1)
            covered[i] = np.any(distances <= all_ranges)

        return covered

    def _get_obs(self):
        """获取环境观察"""
        observations = []
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])

        for i, pos in enumerate(all_positions):
            # 获取最近的POI信息
            poi_distances = np.linalg.norm(self.pois - pos, axis=1)
            nearest_poi_idx = np.argmin(poi_distances)
            nearest_poi = self.pois[nearest_poi_idx]
            nearest_poi_priority = self.poi_priorities[nearest_poi_idx]

            # 获取最近的障碍物信息
            obstacle_distances = np.linalg.norm(self.obstacles - pos, axis=1)
            nearest_obstacle = self.obstacles[np.argmin(obstacle_distances)]

            # 获取最近的充电站信息
            charging_station_distances = np.linalg.norm(self.charging_stations - pos, axis=1)
            nearest_charging = self.charging_stations[np.argmin(charging_station_distances)]

            # 标准化距离到[0,1]范围
            max_distance = np.sqrt(2) * self.area_size
            normalized_distances = min(np.min(poi_distances) / max_distance, 1.0)

            # 组合观察 (总共9维)
            obs = np.concatenate([
                pos / self.area_size,  # 归一化位置 (2)
                nearest_poi / self.area_size,  # 归一化最近POI位置 (2)
                [nearest_poi_priority / self.config.poi_priority_levels],  # 归一化优先级 (1)
                [self.agent_energy[i] / self.uav_energy_capacity],  # 归一化能量 (1)
                [normalized_distances],  # 归一化最近POI距离 (1)
                nearest_charging / self.area_size,  # 归一化最近充电站位置 (2)
            ])
            observations.append(obs)

        return np.array(observations)

    def _get_info(self):
        """获取环境信息"""
        # 获取当前覆盖率
        covered_pois = self._get_covered_pois()
        coverage = np.mean(covered_pois)

        # 构建位置信息
        positions = {
            'satellites': self.satellites.copy(),
            'uavs': self.uavs.copy(),
            'ground_stations': self.ground_stations.copy(),
            'pois': self.pois.copy(),
            'charging_stations': self.charging_stations.copy(),
            'obstacles': self.obstacles.copy()
        }

        # 获取UAV能量信息
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]

        return {
            # 覆盖率指标
            'coverage': coverage,
            'covered_pois': covered_pois,
            'priority_coverage': np.mean(covered_pois * (self.poi_priorities >= self.config.poi_priority_levels * 0.7)),
            'energy': self.agent_energy.copy(),
            'uav_energy': uav_energy,
            'energy_mean': np.mean(uav_energy),
            'energy_min': np.min(uav_energy),

            # 位置信息
            'positions': positions,
            'trajectories': self.agent_trajectories,

            # 碰撞指标
            'collisions': self.collision_count,
            'current_collisions': self._check_collisions(),

            # 充电指标
            'charging_events': self.episode_data['charging_events'],
            'charging_efficiency': self.episode_data['charging_events'] / (self.time_step + 1),

            # 通信指标
            'communication_density': self._calculate_communication_density(),
            'communication_links': self._get_communication_links(),

            # 任务完成指标
            'time_step': self.time_step,
            'total_reward': self.total_reward,
            'completion_rate': np.mean(self._get_covered_pois() > 0.8),

            # 系统效率指标
            'resource_utilization': self._calculate_resource_utilization(),
            'load_balance': self._calculate_load_balance()
        }

    def _check_done(self):
        """检查环境是否结束"""
        # 时间步数达到上限
        if self.time_step >= self.max_time_steps:
            return True

        # 所有UAV能量耗尽
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        if np.all(uav_energy == 0):
            return True

        # 碰撞次数过多
        if self.collision_count > self.config.max_time_steps * 0.5:
            return True

        return False

    def _calculate_communication_density(self):
        """计算通信密度"""
        communication_range = self.config.communication_range
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])

        total_links = 0
        possible_links = (self.num_agents * (self.num_agents - 1)) / 2

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(all_positions[i] - all_positions[j]) <= communication_range:
                    total_links += 1

        return total_links / possible_links

    def _get_communication_links(self):
        """获取通信链接"""
        communication_range = self.config.communication_range
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        links = []

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(all_positions[i] - all_positions[j]) <= communication_range:
                    links.append((all_positions[i], all_positions[j]))

        return links

    def _calculate_resource_utilization(self):
        """计算资源利用率"""
        # UAV能量利用率
        uav_energy = self.agent_energy[self.num_satellites:self.num_satellites + self.num_uavs]
        energy_utilization = 1 - np.mean(uav_energy / self.uav_energy_capacity)

        # 覆盖效率
        coverage = np.mean(self._get_covered_pois())

        # 通信效率
        comm_density = self._calculate_communication_density()

        return (energy_utilization + coverage + comm_density) / 3

    def _calculate_load_balance(self):
        """计算负载均衡性"""
        # 计算每个智能体的覆盖负载
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        all_ranges = np.concatenate([
            np.full(self.num_satellites, self.satellite_range),
            np.full(self.num_uavs, self.uav_range),
            np.full(self.num_ground_stations, self.ground_station_range)
        ])

        loads = np.zeros(self.num_agents)
        for i, pos in enumerate(all_positions):
            for poi, priority in zip(self.pois, self.poi_priorities):
                if np.linalg.norm(pos - poi) <= all_ranges[i]:
                    loads[i] += priority

        # 计算负载的变异系数 (标准差/平均值)
        if np.mean(loads) > 0:
            return 1 - (np.std(loads) / np.mean(loads))
        return 0

    def get_coverage_map(self):
        """生成覆盖地图"""
        resolution = 100
        coverage_map = np.zeros((resolution, resolution))
        x = np.linspace(0, self.area_size, resolution)
        y = np.linspace(0, self.area_size, resolution)
        xx, yy = np.meshgrid(x, y)

        # 所有智能体的位置和范围
        all_positions = np.concatenate([self.satellites, self.uavs, self.ground_stations])
        all_ranges = np.concatenate([
            np.full(self.num_satellites, self.satellite_range),
            np.full(self.num_uavs, self.uav_range),
            np.full(self.num_ground_stations, self.ground_station_range)
        ])

        # 计算每个点的覆盖情况
        for pos, range_val in zip(all_positions, all_ranges):
            dist = np.sqrt((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2)
            coverage_map += (dist <= range_val).astype(float)

        return coverage_map

    def get_state_dict(self):
        """获取环境完整状态字典"""
        return {
            'time_step': self.time_step,
            'satellites': self.satellites.copy(),
            'uavs': self.uavs.copy(),
            'ground_stations': self.ground_stations.copy(),
            'pois': self.pois.copy(),
            'poi_priorities': self.poi_priorities.copy(),
            'charging_stations': self.charging_stations.copy(),
            'obstacles': self.obstacles.copy(),
            'agent_energy': self.agent_energy.copy(),
            'collision_count': self.collision_count,
            'total_reward': self.total_reward,
            'trajectories': {k: v.copy() for k, v in self.agent_trajectories.items()},
            'episode_data': self.episode_data.copy()
        }

    def load_state_dict(self, state_dict):
        """加载环境状态"""
        self.time_step = state_dict['time_step']
        self.satellites = state_dict['satellites'].copy()
        self.uavs = state_dict['uavs'].copy()
        self.ground_stations = state_dict['ground_stations'].copy()
        self.pois = state_dict['pois'].copy()
        self.poi_priorities = state_dict['poi_priorities'].copy()
        self.charging_stations = state_dict['charging_stations'].copy()
        self.obstacles = state_dict['obstacles'].copy()
        self.agent_energy = state_dict['agent_energy'].copy()
        self.collision_count = state_dict['collision_count']
        self.total_reward = state_dict['total_reward']
        self.agent_trajectories = {k: v.copy() for k, v in state_dict['trajectories'].items()}
        self.episode_data = state_dict['episode_data'].copy()

    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def render(self):
        """渲染环境（可以根据需要实现可视化）"""
        pass

    def close(self):
        """关闭环境"""
        pass