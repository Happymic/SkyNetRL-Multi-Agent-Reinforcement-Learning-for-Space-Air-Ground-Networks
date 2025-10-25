"""
Enhanced SAGIN Environment with Proper Observation Structure
Designed to work with attention-enhanced algorithms
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import copy


@dataclass
class AgentConfig:
    """Configuration for different agent types"""
    coverage_radius: float
    max_speed: float
    energy_capacity: Optional[float] = None
    energy_consumption_rate: Optional[float] = None
    

@dataclass
class POI:
    """Point of Interest"""
    x: float
    y: float
    priority: int
    importance: float = 1.0
    covered: bool = False
    coverage_time: int = 0


@dataclass
class Obstacle:
    """Obstacle in the environment"""
    x: float
    y: float
    radius: float


@dataclass
class ChargingStation:
    """Charging station for UAVs"""
    x: float
    y: float
    availability: bool = True
    importance: float = 1.0


class EnhancedSAGINEnvironment(gym.Env):
    """Enhanced SAGIN Environment with proper attention-compatible observations"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.area_size = config['area_size']
        self.max_episode_steps = config.get('max_episode_steps', 200)
        
        # Agent configuration
        self.num_satellites = config['num_satellites']
        self.num_uavs = config['num_uavs']
        self.num_ground_stations = config['num_ground_stations']
        self.num_agents = self.num_satellites + self.num_uavs + self.num_ground_stations
        
        # Environment objects
        self.num_pois = config['num_pois']
        self.num_obstacles = config.get('num_obstacles', 3)
        self.num_charging_stations = config.get('num_charging_stations', 4)
        
        # Agent type configurations
        self.agent_configs = {
            'satellite': AgentConfig(
                coverage_radius=config.get('satellite_coverage_radius', 250),
                max_speed=config.get('satellite_max_speed', 3)
            ),
            'uav': AgentConfig(
                coverage_radius=config.get('uav_coverage_radius', 120),
                max_speed=config.get('uav_max_speed', 6),
                energy_capacity=config.get('uav_energy_capacity', 1200),
                energy_consumption_rate=config.get('uav_energy_consumption', 5)
            ),
            'ground_station': AgentConfig(
                coverage_radius=config.get('ground_station_coverage_radius', 80),
                max_speed=config.get('ground_station_max_speed', 2)
            )
        }
        
        # Communication range
        self.comm_range = config.get('communication_range', 200)
        
        # Reward weights
        self.reward_weights = {
            'coverage': config.get('coverage_reward_weight', 1.0),
            'task_priority': config.get('task_priority_weight', 0.5),
            'energy_penalty': config.get('energy_penalty_weight', 0.05),
            'collision_penalty': config.get('collision_penalty_weight', 0.3)
        }
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32  # [vel_x, vel_y]
        )
        
        # Enhanced observation space: 184 dimensions
        # [self_obs(9) + spatial_obs(80) + agent_obs(90) + task_obs(5)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(184,), dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.agents = {}
        self.pois = []
        self.obstacles = []
        self.charging_stations = []
        
        # History tracking
        self.coverage_history = []
        self.energy_history = []
        self.collision_history = []
        self.position_history = []
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, return_info: bool = False) -> Dict[int, np.ndarray]:
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # Initialize agents
        self._initialize_agents()
        
        # Initialize POIs
        self._initialize_pois()
        
        # Initialize obstacles
        self._initialize_obstacles()
        
        # Initialize charging stations
        self._initialize_charging_stations()
        
        # Clear history
        self.coverage_history = []
        self.energy_history = []
        self.collision_history = []
        self.position_history = []
        
        # Get initial observations
        observations = self._get_observations()
        
        if return_info:
            return observations, self._get_info()
        return observations
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Apply actions
        self._apply_actions(actions)
        
        # Update environment state
        self._update_coverage()
        self._update_energy()
        collisions = self._check_collisions()
        
        # Compute rewards
        rewards = self._compute_rewards(collisions)
        
        # Check if episode is done
        dones = self._check_dones()
        
        # Get new observations
        observations = self._get_observations()
        
        # Store history
        self._record_history(collisions)
        
        # Additional info
        info = self._get_info()
        
        return observations, rewards, dones, info
    
    def _initialize_agents(self):
        """Initialize all agents with proper positions and states"""
        self.agents = {}
        
        # Initialize satellites
        for i in range(self.num_satellites):
            self.agents[i] = {
                'type': 'satellite',
                'position': np.random.uniform(0, self.area_size, 2),
                'velocity': np.zeros(2),
                'energy': None,  # Satellites have unlimited energy
                'active': True
            }
        
        # Initialize UAVs
        for i in range(self.num_satellites, self.num_satellites + self.num_uavs):
            self.agents[i] = {
                'type': 'uav',
                'position': np.random.uniform(0, self.area_size, 2),
                'velocity': np.zeros(2),
                'energy': self.agent_configs['uav'].energy_capacity,
                'active': True
            }
        
        # Initialize ground stations
        for i in range(self.num_satellites + self.num_uavs, self.num_agents):
            self.agents[i] = {
                'type': 'ground_station',
                'position': np.random.uniform(0, self.area_size, 2),
                'velocity': np.zeros(2),
                'energy': None,  # Ground stations have unlimited energy
                'active': True
            }
    
    def _initialize_pois(self):
        """Initialize Points of Interest"""
        self.pois = []
        for i in range(self.num_pois):
            poi = POI(
                x=np.random.uniform(0, self.area_size),
                y=np.random.uniform(0, self.area_size),
                priority=np.random.randint(1, 6),  # Priority 1-5
                importance=np.random.uniform(0.5, 2.0)
            )
            self.pois.append(poi)
    
    def _initialize_obstacles(self):
        """Initialize obstacles"""
        self.obstacles = []
        for i in range(self.num_obstacles):
            obstacle = Obstacle(
                x=np.random.uniform(50, self.area_size - 50),
                y=np.random.uniform(50, self.area_size - 50),
                radius=np.random.uniform(20, 50)
            )
            self.obstacles.append(obstacle)
    
    def _initialize_charging_stations(self):
        """Initialize charging stations"""
        self.charging_stations = []
        for i in range(self.num_charging_stations):
            station = ChargingStation(
                x=np.random.uniform(0, self.area_size),
                y=np.random.uniform(0, self.area_size),
                importance=np.random.uniform(0.8, 1.5)
            )
            self.charging_stations.append(station)
    
    def _apply_actions(self, actions: Dict[int, np.ndarray]):
        """Apply actions to agents"""
        for agent_id, action in actions.items():
            if agent_id not in self.agents or not self.agents[agent_id]['active']:
                continue
            
            agent = self.agents[agent_id]
            agent_type = agent['type']
            config = self.agent_configs[agent_type]
            
            # Update velocity based on action
            agent['velocity'] = np.clip(action, -1, 1) * config.max_speed
            
            # Update position
            new_position = agent['position'] + agent['velocity']
            
            # Keep agents within bounds
            new_position = np.clip(new_position, 0, self.area_size)
            
            # Check obstacle collisions
            if not self._position_in_obstacle(new_position):
                agent['position'] = new_position
    
    def _position_in_obstacle(self, position: np.ndarray) -> bool:
        """Check if position is inside an obstacle"""
        for obstacle in self.obstacles:
            dist = np.linalg.norm(position - np.array([obstacle.x, obstacle.y]))
            if dist < obstacle.radius:
                return True
        return False
    
    def _update_coverage(self):
        """Update POI coverage status"""
        for poi in self.pois:
            poi.covered = False
            
            for agent_id, agent in self.agents.items():
                if not agent['active']:
                    continue
                
                agent_type = agent['type']
                coverage_radius = self.agent_configs[agent_type].coverage_radius
                
                # Check if agent covers this POI
                dist = np.linalg.norm(agent['position'] - np.array([poi.x, poi.y]))
                if dist <= coverage_radius:
                    poi.covered = True
                    poi.coverage_time += 1
                    break
    
    def _update_energy(self):
        """Update energy levels for UAVs"""
        for agent_id, agent in self.agents.items():
            if agent['type'] == 'uav' and agent['active']:
                # Consume energy based on movement
                speed = np.linalg.norm(agent['velocity'])
                energy_consumption = self.agent_configs['uav'].energy_consumption_rate * (1 + speed / 10)
                agent['energy'] = max(0, agent['energy'] - energy_consumption)
                
                # Check if near charging station
                for station in self.charging_stations:
                    dist = np.linalg.norm(agent['position'] - np.array([station.x, station.y]))
                    if dist <= 30 and station.availability:  # Within 30m of charging station
                        # Charge the UAV
                        charge_rate = 50  # Energy units per step
                        agent['energy'] = min(
                            self.agent_configs['uav'].energy_capacity,
                            agent['energy'] + charge_rate
                        )
                
                # Deactivate if out of energy
                if agent['energy'] <= 0:
                    agent['active'] = False
    
    def _check_collisions(self) -> int:
        """Check for agent collisions"""
        collisions = 0
        agent_positions = [(agent_id, agent['position']) for agent_id, agent in self.agents.items() 
                          if agent['active']]
        
        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                pos1 = agent_positions[i][1]
                pos2 = agent_positions[j][1]
                dist = np.linalg.norm(pos1 - pos2)
                
                if dist < 20:  # Collision threshold
                    collisions += 1
        
        return collisions
    
    def _compute_rewards(self, collisions: int) -> Dict[int, float]:
        """Compute rewards for all agents"""
        rewards = {}
        
        # Global rewards
        coverage_reward = self._compute_coverage_reward()
        collision_penalty = self.reward_weights['collision_penalty'] * collisions
        
        for agent_id, agent in self.agents.items():
            reward = coverage_reward - collision_penalty
            
            # Individual energy penalty for UAVs
            if agent['type'] == 'uav' and agent['active']:
                energy_ratio = agent['energy'] / self.agent_configs['uav'].energy_capacity
                if energy_ratio < 0.2:  # Low energy penalty
                    reward -= self.reward_weights['energy_penalty'] * (0.2 - energy_ratio)
            
            # Movement efficiency reward (reward for purposeful movement)
            velocity_magnitude = np.linalg.norm(agent['velocity'])
            if velocity_magnitude > 0.1:  # Avoid division by zero
                # Reward for moving towards uncovered POIs
                closest_uncovered_poi = self._find_closest_uncovered_poi(agent['position'])
                if closest_uncovered_poi is not None:
                    direction_to_poi = closest_uncovered_poi - agent['position']
                    if np.linalg.norm(direction_to_poi) > 0:
                        direction_to_poi = direction_to_poi / np.linalg.norm(direction_to_poi)
                        velocity_direction = agent['velocity'] / velocity_magnitude
                        alignment = np.dot(direction_to_poi, velocity_direction)
                        reward += 0.1 * alignment  # Small reward for moving towards POIs
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _find_closest_uncovered_poi(self, position: np.ndarray) -> np.ndarray:
        """Find the closest uncovered POI to the given position"""
        uncovered_pois = [poi for poi in self.pois if not poi.covered]
        if not uncovered_pois:
            return None
        
        distances = [np.linalg.norm(position - np.array([poi.x, poi.y])) for poi in uncovered_pois]
        closest_idx = np.argmin(distances)
        return np.array([uncovered_pois[closest_idx].x, uncovered_pois[closest_idx].y])
    
    def _compute_coverage_reward(self) -> float:
        """Compute coverage-based reward"""
        covered_pois = sum(1 for poi in self.pois if poi.covered)
        coverage_rate = covered_pois / len(self.pois)
        
        # Priority-weighted coverage
        priority_reward = sum(poi.priority * poi.importance for poi in self.pois if poi.covered)
        max_priority_reward = sum(poi.priority * poi.importance for poi in self.pois)
        
        base_coverage_reward = self.reward_weights['coverage'] * coverage_rate
        priority_coverage_reward = self.reward_weights['task_priority'] * (priority_reward / max_priority_reward)
        
        return base_coverage_reward + priority_coverage_reward
    
    def _check_dones(self) -> Dict[int, bool]:
        """Check if episode is done"""
        done = self.current_step >= self.max_episode_steps
        
        # Check if all POIs are covered
        all_covered = all(poi.covered for poi in self.pois)
        done = done or all_covered
        
        # Check if all UAVs are out of energy
        active_uavs = sum(1 for agent in self.agents.values() 
                         if agent['type'] == 'uav' and agent['active'])
        if active_uavs == 0 and self.num_uavs > 0:
            done = True
        
        return {agent_id: done for agent_id in self.agents.keys()}
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents"""
        observations = {}
        
        for agent_id, agent in self.agents.items():
            obs = self._get_agent_observation(agent_id)
            observations[agent_id] = obs
        
        return observations
    
    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent following the enhanced structure"""
        agent = self.agents[agent_id]
        
        # 1. Self observation (9 dimensions)
        self_obs = self._get_self_observation(agent_id)
        
        # 2. Spatial observation (80 dimensions: 20 objects × 4 features)
        spatial_obs = self._get_spatial_observation(agent_id)
        
        # 3. Agent observation (90 dimensions: 10 agents × 9 features)
        agent_obs = self._get_agent_observation_features(agent_id)
        
        # 4. Task observation (5 dimensions)
        task_obs = self._get_task_observation(agent_id)
        
        # Combine all observations
        full_obs = np.concatenate([self_obs, spatial_obs, agent_obs, task_obs])
        
        # Ensure correct dimension
        assert len(full_obs) == 184, f"Observation dimension mismatch: {len(full_obs)} != 184"
        
        return full_obs.astype(np.float32)
    
    def _get_self_observation(self, agent_id: int) -> np.ndarray:
        """Get agent's own state observation (9 dimensions)"""
        agent = self.agents[agent_id]
        
        # Position (2), velocity (2), energy (1), agent type (3), active status (1)
        obs = np.zeros(9)
        
        # Position (normalized)
        obs[0] = agent['position'][0] / self.area_size
        obs[1] = agent['position'][1] / self.area_size
        
        # Velocity (normalized)
        max_speed = self.agent_configs[agent['type']].max_speed
        obs[2] = agent['velocity'][0] / max_speed if max_speed > 0 else 0
        obs[3] = agent['velocity'][1] / max_speed if max_speed > 0 else 0
        
        # Energy (normalized, 0 if unlimited energy)
        if agent['energy'] is not None:
            obs[4] = agent['energy'] / self.agent_configs[agent['type']].energy_capacity
        else:
            obs[4] = 1.0  # Full energy for unlimited energy agents
        
        # Agent type (one-hot encoding)
        if agent['type'] == 'satellite':
            obs[5] = 1.0
        elif agent['type'] == 'uav':
            obs[6] = 1.0
        else:  # ground_station
            obs[7] = 1.0
        
        # Active status
        obs[8] = 1.0 if agent['active'] else 0.0
        
        return obs
    
    def _get_spatial_observation(self, agent_id: int) -> np.ndarray:
        """Get spatial observation (80 dimensions: 20 objects × 4 features)"""
        agent = self.agents[agent_id]
        agent_pos = agent['position']
        
        spatial_obs = np.zeros(80)  # 20 objects × 4 features
        
        # Fill first 10 slots with POIs (prioritize closest and highest priority)
        poi_data = []
        for poi in self.pois:
            dist = np.linalg.norm(agent_pos - np.array([poi.x, poi.y]))
            poi_data.append((dist, poi))
        
        # Sort by distance, then by priority
        poi_data.sort(key=lambda x: (x[0], -x[1].priority))
        
        for i, (dist, poi) in enumerate(poi_data[:10]):
            idx = i * 4
            spatial_obs[idx] = poi.x / self.area_size      # x position
            spatial_obs[idx + 1] = poi.y / self.area_size  # y position
            spatial_obs[idx + 2] = poi.priority / 5.0      # priority (normalized)
            spatial_obs[idx + 3] = 1.0 if poi.covered else 0.0  # coverage status
        
        # Fill next 5 slots with obstacles
        for i, obstacle in enumerate(self.obstacles[:5]):
            idx = 40 + i * 4
            spatial_obs[idx] = obstacle.x / self.area_size
            spatial_obs[idx + 1] = obstacle.y / self.area_size
            spatial_obs[idx + 2] = obstacle.radius / 100.0  # normalized radius
            spatial_obs[idx + 3] = 1.0  # obstacle indicator
        
        # Fill last 5 slots with charging stations
        for i, station in enumerate(self.charging_stations[:5]):
            idx = 60 + i * 4
            spatial_obs[idx] = station.x / self.area_size
            spatial_obs[idx + 1] = station.y / self.area_size
            spatial_obs[idx + 2] = 1.0 if station.availability else 0.0
            spatial_obs[idx + 3] = station.importance / 2.0  # normalized importance
        
        return spatial_obs
    
    def _get_agent_observation_features(self, agent_id: int) -> np.ndarray:
        """Get other agents' observation (90 dimensions: 10 agents × 9 features)"""
        agent_obs = np.zeros(90)
        agent_pos = self.agents[agent_id]['position']
        
        # Get other agents within communication range
        other_agents = []
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id:
                dist = np.linalg.norm(agent_pos - other_agent['position'])
                if dist <= self.comm_range:
                    other_agents.append((dist, other_id, other_agent))
        
        # Sort by distance
        other_agents.sort(key=lambda x: x[0])
        
        # Fill observation for up to 10 closest agents
        for i, (dist, other_id, other_agent) in enumerate(other_agents[:10]):
            idx = i * 9
            
            # Position (relative)
            rel_pos = other_agent['position'] - agent_pos
            agent_obs[idx] = rel_pos[0] / self.comm_range
            agent_obs[idx + 1] = rel_pos[1] / self.comm_range
            
            # Velocity
            max_speed = self.agent_configs[other_agent['type']].max_speed
            agent_obs[idx + 2] = other_agent['velocity'][0] / max_speed if max_speed > 0 else 0
            agent_obs[idx + 3] = other_agent['velocity'][1] / max_speed if max_speed > 0 else 0
            
            # Energy
            if other_agent['energy'] is not None:
                agent_obs[idx + 4] = other_agent['energy'] / self.agent_configs[other_agent['type']].energy_capacity
            else:
                agent_obs[idx + 4] = 1.0
            
            # Agent type (one-hot)
            if other_agent['type'] == 'satellite':
                agent_obs[idx + 5] = 1.0
            elif other_agent['type'] == 'uav':
                agent_obs[idx + 6] = 1.0
            else:  # ground_station
                agent_obs[idx + 7] = 1.0
            
            # Active status
            agent_obs[idx + 8] = 1.0 if other_agent['active'] else 0.0
        
        return agent_obs
    
    def _get_task_observation(self, agent_id: int) -> np.ndarray:
        """Get task-related observation (5 dimensions)"""
        task_obs = np.zeros(5)
        
        # Overall coverage rate
        covered_pois = sum(1 for poi in self.pois if poi.covered)
        task_obs[0] = covered_pois / len(self.pois)
        
        # High priority POI coverage rate
        high_priority_pois = [poi for poi in self.pois if poi.priority >= 4]
        if high_priority_pois:
            high_priority_covered = sum(1 for poi in high_priority_pois if poi.covered)
            task_obs[1] = high_priority_covered / len(high_priority_pois)
        else:
            task_obs[1] = 1.0
        
        # Episode progress
        task_obs[2] = self.current_step / self.max_episode_steps
        
        # Average energy level of active UAVs
        active_uavs = [agent for agent in self.agents.values() 
                      if agent['type'] == 'uav' and agent['active']]
        if active_uavs:
            avg_energy = np.mean([agent['energy'] / self.agent_configs['uav'].energy_capacity 
                                for agent in active_uavs])
            task_obs[3] = avg_energy
        else:
            task_obs[3] = 0.0
        
        # Mission urgency (higher when many high-priority POIs uncovered)
        uncovered_high_priority = sum(1 for poi in self.pois if poi.priority >= 4 and not poi.covered)
        total_high_priority = sum(1 for poi in self.pois if poi.priority >= 4)
        if total_high_priority > 0:
            task_obs[4] = uncovered_high_priority / total_high_priority
        else:
            task_obs[4] = 0.0
        
        return task_obs
    
    def _record_history(self, collisions: int):
        """Record episode history for analysis"""
        # Coverage history
        coverage_status = {i: poi.covered for i, poi in enumerate(self.pois)}
        self.coverage_history.append(coverage_status)
        
        # Energy history
        energy_status = {agent_id: agent['energy'] for agent_id, agent in self.agents.items() 
                        if agent['energy'] is not None}
        self.energy_history.append(energy_status)
        
        # Collision history
        self.collision_history.append(collisions)
        
        # Position history
        position_status = {agent_id: agent['position'].copy() for agent_id, agent in self.agents.items()}
        self.position_history.append(position_status)
    
    def _get_info(self) -> Dict:
        """Get additional environment information"""
        return {
            'step': self.current_step,
            'coverage_rate': sum(1 for poi in self.pois if poi.covered) / len(self.pois),
            'active_agents': sum(1 for agent in self.agents.values() if agent['active']),
            'total_collisions': sum(self.collision_history),
            'pois': self.pois,
            'obstacles': self.obstacles,
            'charging_stations': self.charging_stations,
            'num_pois': len(self.pois),
            'poi_priorities': np.array([poi.priority for poi in self.pois]),
            'area_size': self.area_size
        }
    
    def render(self, mode='human', save_path: Optional[str] = None):
        """Render the environment"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw area boundary
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_aspect('equal')
        
        # Draw obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle.x, obstacle.y), obstacle.radius, 
                              color='red', alpha=0.5, label='Obstacle')
            ax.add_patch(circle)
        
        # Draw charging stations
        for station in self.charging_stations:
            color = 'green' if station.availability else 'orange'
            ax.plot(station.x, station.y, 's', color=color, markersize=8, 
                   label='Charging Station' if station.availability else 'Busy Station')
        
        # Draw POIs
        for poi in self.pois:
            color = 'blue' if poi.covered else 'lightblue'
            size = poi.priority * 2 + 5  # Size based on priority
            ax.plot(poi.x, poi.y, 'o', color=color, markersize=size, 
                   alpha=0.7, label='POI (Covered)' if poi.covered else 'POI (Uncovered)')
        
        # Draw agents
        agent_colors = {'satellite': 'purple', 'uav': 'red', 'ground_station': 'brown'}
        agent_markers = {'satellite': '^', 'uav': 'v', 'ground_station': 's'}
        
        for agent_id, agent in self.agents.items():
            if not agent['active']:
                continue
                
            color = agent_colors[agent['type']]
            marker = agent_markers[agent['type']]
            
            ax.plot(agent['position'][0], agent['position'][1], marker, 
                   color=color, markersize=10, label=f"{agent['type'].title()}")
            
            # Draw coverage radius
            coverage_radius = self.agent_configs[agent['type']].coverage_radius
            circle = plt.Circle(agent['position'], coverage_radius, 
                              color=color, alpha=0.1, linestyle='--')
            ax.add_patch(circle)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.set_title(f"SAGIN Environment - Step {self.current_step}")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if mode == 'human':
            plt.show()
        
        plt.close(fig)
    
    def set_channel_model(self, channel_model):
        """Set 3GPP-compliant channel model"""
        self.channel_model = channel_model
        self.enable_3gpp_channels = True
        
    def set_mac_layer(self, mac_layer):
        """Set MAC layer for protocol simulation"""
        self.mac_layer = mac_layer
        self.enable_mac_protocols = True
        
    def set_qos_manager(self, qos_manager):
        """Set QoS manager for protocol simulation"""
        self.qos_manager = qos_manager
        
    def get_academic_metrics_data(self):
        """Get data for academic metrics evaluation"""
        data = {
            'coverage_events': [],
            'throughput_values': [],
            'latency_values': [],
            'packet_loss_rates': [],
            'energy_consumption': [],
            'spectral_efficiency': [],
            'fairness_indices': [],
            'rewards': []
        }
        
        # Extract coverage events
        for coverage_status in self.coverage_history:
            coverage_rate = sum(coverage_status.values()) / len(coverage_status)
            data['coverage_events'].append(coverage_rate > 0.8)  # 80% coverage threshold
            
        # Extract energy consumption
        for energy_status in self.energy_history:
            if energy_status:
                avg_energy = np.mean(list(energy_status.values()))
                data['energy_consumption'].append(avg_energy)
                
        # Simulated network metrics (would be real in full implementation)
        if hasattr(self, 'mac_layer') and self.mac_layer:
            stats = self.mac_layer.get_statistics()
            data['throughput_values'] = [stats.get('average_throughput_bps', 0)]
            data['packet_loss_rates'] = [stats.get('blocking_probability', 0)]
            
        if hasattr(self, 'qos_manager') and self.qos_manager:
            qos_stats = self.qos_manager.get_statistics()
            data['latency_values'] = [qos_stats['global_metrics'].get('average_delay_ms', 0) / 1000]
            
        # Calculate fairness based on agent rewards (simplified)
        if hasattr(self, 'reward_history') and len(self.reward_history) > 0:
            agent_total_rewards = {}
            for rewards in self.reward_history:
                for agent_id, reward in rewards.items():
                    if agent_id not in agent_total_rewards:
                        agent_total_rewards[agent_id] = 0
                    agent_total_rewards[agent_id] += reward
                    
            if len(agent_total_rewards) > 1:
                reward_values = list(agent_total_rewards.values())
                mean_reward = np.mean(reward_values)
                sum_squared = sum((r - mean_reward)**2 for r in reward_values)
                fairness_index = (mean_reward**2) / (sum_squared / len(reward_values) + 1e-12)
                data['fairness_indices'] = [fairness_index]
                
        return data
    
    def get_episode_data(self):
        """Get episode data for evaluation"""
        # Return academic metrics data for compatibility
        return self.get_academic_metrics_data()