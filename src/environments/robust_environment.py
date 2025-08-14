"""
Robust SAGIN Environment with Advanced Randomization
Enhances robustness through intelligent environment randomization
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import random
from scipy.spatial.distance import cdist
from scipy.stats import truncnorm

from .enhanced_sagin_env import EnhancedSAGINEnvironment, POI, AgentConfig
from ..rewards.multi_objective_rewards import MultiObjectiveRewardSystem, create_multi_objective_reward_system


@dataclass
class EnvironmentalConditions:
    """Environmental conditions that affect agent performance"""
    weather_factor: float = 1.0  # 0.5-1.5, affects visibility and movement
    communication_noise: float = 0.0  # 0-0.3, affects comm range
    wind_speed: float = 0.0  # 0-10 m/s, affects UAV movement
    wind_direction: float = 0.0  # 0-2π radians
    solar_interference: float = 0.0  # 0-0.2, affects satellite communication


@dataclass 
class RandomizationConfig:
    """Configuration for environment randomization"""
    # Agent position randomization
    agent_position_noise: float = 100.0  # Standard deviation in meters
    agent_altitude_variance: float = 20.0  # Altitude variation
    
    # POI randomization
    poi_position_noise: float = 50.0  # POI position variation
    poi_priority_shuffle: bool = True  # Randomly shuffle priorities
    poi_density_variation: float = 0.2  # ±20% POI count variation
    
    # Environmental conditions
    weather_variation: bool = True
    communication_noise_level: float = 0.1
    wind_effects: bool = True
    
    # Obstacle randomization
    dynamic_obstacles: bool = True
    obstacle_count_range: Tuple[int, int] = (2, 8)
    
    # Mission parameters
    mission_time_variation: float = 0.3  # ±30% episode length variation
    emergency_events: bool = True  # Random high-priority events


class RobustSAGINEnvironment(EnhancedSAGINEnvironment):
    """Enhanced environment with comprehensive randomization for robustness"""
    
    def __init__(self, config: Dict):
        # Initialize all attributes first before calling parent init
        self.randomization_config = RandomizationConfig()
        if 'randomization' in config:
            for key, value in config['randomization'].items():
                if hasattr(self.randomization_config, key):
                    setattr(self.randomization_config, key, value)
        
        # Randomization state
        self.environmental_conditions = EnvironmentalConditions()
        self.episode_seed = None
        self.initial_conditions = None
        
        # Advanced statistics for robustness analysis
        self.robustness_metrics = {
            'scenario_difficulty': [],
            'adaptation_performance': [],
            'environmental_variance': []
        }
        
        # Initialize multi-objective reward system
        self.reward_system = create_multi_objective_reward_system(config, config)
        self.previous_coverage = None
        
        # Now initialize base environment (this will call reset)
        super().__init__(config)
        
        print("🎲 Robust SAGIN Environment initialized with advanced randomization")
        print("🎯 Multi-objective reward system integrated")
    
    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[Dict] = None):
        """Enhanced reset with intelligent randomization"""
        
        # Set episode seed for reproducibility if needed
        if seed is not None:
            self.episode_seed = seed
            np.random.seed(seed)
            random.seed(seed)
        else:
            self.episode_seed = np.random.randint(0, 2**32 - 1)
            np.random.seed(self.episode_seed)
        
        # Generate randomized initial conditions
        self.initial_conditions = self._generate_initial_conditions()
        
        # Reset environment with new conditions
        self._reset_with_conditions(self.initial_conditions)
        
        # Get initial observations
        observations = self._get_structured_observations()
        
        # Calculate scenario difficulty for analysis
        scenario_difficulty = self._calculate_scenario_difficulty()
        self.robustness_metrics['scenario_difficulty'].append(scenario_difficulty)
        
        # Reset multi-objective reward system for new episode
        self.reward_system.reset_episode()
        self.previous_coverage = None
        
        info = {
            'episode_seed': self.episode_seed,
            'scenario_difficulty': scenario_difficulty,
            'environmental_conditions': self.environmental_conditions,
            'initial_poi_distribution': [(poi.x, poi.y, poi.priority) for poi in self.pois]
        }
        
        if return_info:
            return observations, info
        return observations
    
    def _generate_initial_conditions(self) -> Dict[str, Any]:
        """Generate intelligent randomized initial conditions"""
        conditions = {}
        
        # 1. Randomize environmental conditions
        if self.randomization_config.weather_variation:
            # Weather factor affects visibility and movement efficiency
            conditions['weather_factor'] = np.clip(
                np.random.normal(1.0, 0.2), 0.5, 1.5
            )
            
            # Communication noise
            conditions['communication_noise'] = np.random.uniform(
                0, self.randomization_config.communication_noise_level
            )
            
            if self.randomization_config.wind_effects:
                conditions['wind_speed'] = np.random.exponential(3.0)  # Average 3 m/s
                conditions['wind_direction'] = np.random.uniform(0, 2 * np.pi)
                conditions['solar_interference'] = np.random.uniform(0, 0.2)
        
        # 2. Generate agent starting positions with strategic constraints
        conditions['agent_positions'] = self._generate_strategic_agent_positions()
        
        # 3. Generate POI distribution with realistic clustering
        conditions['poi_distribution'] = self._generate_realistic_poi_distribution()
        
        # 4. Generate dynamic obstacles (after agent positions are determined)
        if self.randomization_config.dynamic_obstacles:
            conditions['obstacles'] = self._generate_dynamic_obstacles(conditions['agent_positions'])
        
        # 5. Randomize mission parameters
        base_steps = self.max_episode_steps
        variation = self.randomization_config.mission_time_variation
        conditions['episode_length'] = int(base_steps * (1 + np.random.uniform(-variation, variation)))
        
        return conditions
    
    def _generate_strategic_agent_positions(self) -> List[Tuple[float, float, float]]:
        """Generate strategic agent starting positions with intelligent randomization"""
        positions = []
        
        # Define strategic zones
        center = np.array([self.area_size / 2, self.area_size / 2])
        
        # Generate positions for each agent type
        agent_index = 0
        
        # Satellites - high altitude, distributed coverage
        for i in range(self.num_satellites):
            # Base position on outer perimeter for wide coverage
            angle = (i / self.num_satellites) * 2 * np.pi + np.random.uniform(-0.3, 0.3)
            base_radius = self.area_size * 0.4 * np.random.uniform(0.8, 1.2)
            
            x = center[0] + base_radius * np.cos(angle)
            y = center[1] + base_radius * np.sin(angle)
            
            # Add randomization noise
            x += np.random.normal(0, self.randomization_config.agent_position_noise)
            y += np.random.normal(0, self.randomization_config.agent_position_noise)
            
            # Satellite altitude with variation
            base_altitude = 200
            altitude = base_altitude + np.random.normal(0, self.randomization_config.agent_altitude_variance)
            altitude = max(150, min(300, altitude))  # Constrain altitude
            
            # Ensure within bounds
            x = max(100, min(self.area_size - 100, x))
            y = max(100, min(self.area_size - 100, y))
            
            positions.append((x, y, altitude))
            agent_index += 1
        
        # UAVs - medium altitude, tactical positioning
        for i in range(self.num_uavs):
            # Strategic positioning near expected POI clusters
            if i == 0:
                # Center UAV for high-priority POIs
                base_pos = center + np.random.normal(0, 50, 2)
            else:
                # Distributed UAVs for coverage
                angle = (i / self.num_uavs) * 2 * np.pi + np.random.uniform(-0.5, 0.5)
                radius = self.area_size * 0.25 * np.random.uniform(0.7, 1.3)
                base_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            
            x, y = base_pos
            
            # UAV altitude with variation
            base_altitude = 100
            altitude = base_altitude + np.random.normal(0, self.randomization_config.agent_altitude_variance * 1.5)
            altitude = max(50, min(200, altitude))
            
            # Ensure within bounds
            x = max(80, min(self.area_size - 80, x))
            y = max(80, min(self.area_size - 80, y))
            
            positions.append((x, y, altitude))
            agent_index += 1
        
        # Ground stations - strategic fixed positions with variation
        for i in range(self.num_ground_stations):
            # Position near area edges for communication relay
            if i == 0:
                # Corner position
                x = self.area_size * 0.2 + np.random.uniform(-50, 50)
                y = self.area_size * 0.2 + np.random.uniform(-50, 50)
            else:
                # Opposite corner
                x = self.area_size * 0.8 + np.random.uniform(-50, 50)
                y = self.area_size * 0.8 + np.random.uniform(-50, 50)
            
            # Ground station altitude (minimal variation)
            altitude = np.random.uniform(0, 15)
            
            positions.append((x, y, altitude))
            agent_index += 1
        
        return positions
    
    def _generate_realistic_poi_distribution(self) -> List[Tuple[float, float, int]]:
        """Generate realistic POI distribution with clustering and priorities"""
        pois = []
        
        # Adjust POI count with variation
        base_count = self.num_pois
        variation = self.randomization_config.poi_density_variation
        poi_count = int(base_count * (1 + np.random.uniform(-variation, variation)))
        poi_count = max(6, min(20, poi_count))  # Reasonable bounds
        
        center = np.array([self.area_size / 2, self.area_size / 2])
        
        # Generate POI clusters with different priorities
        
        # 1. High-priority cluster (center region)
        high_priority_count = max(1, poi_count // 4)
        for i in range(high_priority_count):
            # Cluster around center with some spread
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.gamma(2, 30)  # Gamma distribution for realistic clustering
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # Add noise
            x += np.random.normal(0, self.randomization_config.poi_position_noise * 0.5)
            y += np.random.normal(0, self.randomization_config.poi_position_noise * 0.5)
            
            # High priority (4 or 5)
            priority = np.random.choice([4, 5], p=[0.4, 0.6])
            
            # Ensure within bounds
            x = max(50, min(self.area_size - 50, x))
            y = max(50, min(self.area_size - 50, y))
            
            pois.append((x, y, priority))
        
        # 2. Medium-priority ring distribution
        medium_priority_count = max(2, poi_count // 3)
        for i in range(medium_priority_count):
            angle = np.random.uniform(0, 2 * np.pi)
            # Ring distribution
            radius = np.random.uniform(150, 300)
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # Add noise
            x += np.random.normal(0, self.randomization_config.poi_position_noise)
            y += np.random.normal(0, self.randomization_config.poi_position_noise)
            
            # Medium priority (2, 3, 4)
            priority = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            
            x = max(50, min(self.area_size - 50, x))
            y = max(50, min(self.area_size - 50, y))
            
            pois.append((x, y, priority))
        
        # 3. Low-priority scattered distribution
        remaining_count = poi_count - high_priority_count - medium_priority_count
        for i in range(remaining_count):
            # Uniform distribution in outer areas
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(300, 450)
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # More noise for scattered POIs
            x += np.random.normal(0, self.randomization_config.poi_position_noise * 1.5)
            y += np.random.normal(0, self.randomization_config.poi_position_noise * 1.5)
            
            # Low priority (1, 2)
            priority = np.random.choice([1, 2], p=[0.7, 0.3])
            
            x = max(50, min(self.area_size - 50, x))
            y = max(50, min(self.area_size - 50, y))
            
            pois.append((x, y, priority))
        
        # Optional priority shuffling for additional randomization
        if self.randomization_config.poi_priority_shuffle:
            priorities = [poi[2] for poi in pois]
            np.random.shuffle(priorities)
            pois = [(poi[0], poi[1], priorities[i]) for i, poi in enumerate(pois)]
        
        return pois
    
    def _generate_dynamic_obstacles(self, agent_positions: List[Tuple[float, float, float]] = None) -> List[Tuple[float, float, float]]:
        """Generate dynamic obstacles with realistic placement"""
        obstacles = []
        
        min_count, max_count = self.randomization_config.obstacle_count_range
        obstacle_count = np.random.randint(min_count, max_count + 1)
        
        for i in range(obstacle_count):
            # Avoid placing obstacles too close to agent start positions
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 50:
                x = np.random.uniform(100, self.area_size - 100)
                y = np.random.uniform(100, self.area_size - 100)
                radius = np.random.uniform(20, 60)
                
                # Check minimum distance from initial agent positions if provided
                if agent_positions:
                    min_distance_to_agents = min([
                        np.linalg.norm([x - pos[0], y - pos[1]])
                        for pos in agent_positions
                    ])
                    
                    if min_distance_to_agents > radius + 100:  # Safe distance
                        valid_position = True
                else:
                    valid_position = True  # No agents to avoid
                
                attempts += 1
            
            if valid_position:
                obstacles.append((x, y, radius))
        
        return obstacles
    
    def _reset_with_conditions(self, conditions: Dict[str, Any]):
        """Reset environment with generated conditions"""
        
        # Set environmental conditions
        self.environmental_conditions = EnvironmentalConditions(
            weather_factor=conditions.get('weather_factor', 1.0),
            communication_noise=conditions.get('communication_noise', 0.0),
            wind_speed=conditions.get('wind_speed', 0.0),
            wind_direction=conditions.get('wind_direction', 0.0),
            solar_interference=conditions.get('solar_interference', 0.0)
        )
        
        # Update communication range based on conditions
        base_comm_range = self.config.get('communication_range', 200)
        noise_factor = 1.0 - self.environmental_conditions.communication_noise
        weather_factor = self.environmental_conditions.weather_factor
        self.effective_comm_range = base_comm_range * noise_factor * weather_factor
        
        # Set episode length
        self.max_episode_steps = conditions.get('episode_length', self.max_episode_steps)
        self.current_step = 0
        
        # Initialize agents with randomized positions
        self.agents = []
        agent_positions = conditions['agent_positions']
        
        agent_index = 0
        # Create satellites
        for i in range(self.num_satellites):
            pos = agent_positions[agent_index]
            agent = self._create_agent('satellite', agent_index, pos)
            self.agents.append(agent)
            agent_index += 1
        
        # Create UAVs
        for i in range(self.num_uavs):
            pos = agent_positions[agent_index]
            agent = self._create_agent('uav', agent_index, pos)
            self.agents.append(agent)
            agent_index += 1
        
        # Create ground stations
        for i in range(self.num_ground_stations):
            pos = agent_positions[agent_index]
            agent = self._create_agent('ground_station', agent_index, pos)
            self.agents.append(agent)
            agent_index += 1
        
        # Initialize POIs with randomized distribution
        self.pois = []
        for i, (x, y, priority) in enumerate(conditions['poi_distribution']):
            poi = POI(x=x, y=y, priority=priority, importance=priority/5.0)
            self.pois.append(poi)
        
        # Initialize obstacles if present
        self.obstacles = []
        if 'obstacles' in conditions:
            for x, y, radius in conditions['obstacles']:
                from .enhanced_sagin_env import Obstacle
                obstacle = Obstacle(x=x, y=y, radius=radius)
                self.obstacles.append(obstacle)
        
        # Reset other environment state
        self.done = False
        self.info = {}
    
    def _create_agent(self, agent_type: str, agent_id: int, position: Tuple[float, float, float]):
        """Create agent with proper initialization"""
        # This would create an agent object with the specified type and position
        # For now, return a simple dict representation
        config = self.agent_configs[agent_type]
        
        agent = {
            'id': agent_id,
            'type': agent_type,
            'x': position[0],
            'y': position[1], 
            'z': position[2],
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'coverage_radius': config.coverage_radius,
            'max_speed': config.max_speed,
            'active': True
        }
        
        # Add energy for UAVs
        if agent_type == 'uav':
            agent['energy'] = config.energy_capacity
            agent['max_energy'] = config.energy_capacity
            agent['energy_consumption_rate'] = config.energy_consumption_rate
        
        return agent
    
    def _calculate_scenario_difficulty(self) -> float:
        """Calculate scenario difficulty for robustness analysis"""
        difficulty_factors = []
        
        # Environmental factors
        weather_difficulty = abs(self.environmental_conditions.weather_factor - 1.0)
        comm_difficulty = self.environmental_conditions.communication_noise
        difficulty_factors.extend([weather_difficulty, comm_difficulty * 5])
        
        # POI distribution difficulty
        poi_positions = np.array([(poi.x, poi.y) for poi in self.pois])
        center = np.array([self.area_size / 2, self.area_size / 2])
        
        # Calculate POI dispersion
        if len(poi_positions) > 1:
            distances_to_center = [np.linalg.norm(pos - center) for pos in poi_positions]
            poi_dispersion = np.std(distances_to_center) / np.mean(distances_to_center)
            difficulty_factors.append(poi_dispersion)
        
        # Agent positioning challenge
        agent_positions = np.array([(agent['x'], agent['y']) for agent in self.agents])
        if len(agent_positions) > 1:
            agent_distances = cdist(agent_positions, agent_positions)
            np.fill_diagonal(agent_distances, np.inf)
            min_agent_distance = np.min(agent_distances)
            # Closer agents = more coordination challenge
            coordination_difficulty = max(0, (300 - min_agent_distance) / 300)
            difficulty_factors.append(coordination_difficulty)
        
        # Obstacle difficulty
        obstacle_difficulty = len(self.obstacles) / 10.0  # Normalize
        difficulty_factors.append(obstacle_difficulty)
        
        return np.mean(difficulty_factors)
    
    def _get_structured_observations(self) -> Dict[str, np.ndarray]:
        """Get structured observations for all agents"""
        observations = {}
        
        for agent_id, agent in enumerate(self.agents):
            obs = self._get_agent_observation(agent)
            observations[f'agent_{agent_id}'] = obs
        
        return observations
    
    def _get_agent_observation(self, agent: Dict) -> Dict[str, np.ndarray]:
        """Get structured observation for a single agent"""
        
        # Self state (normalized)
        self_state = np.array([
            agent['x'] / self.area_size,
            agent['y'] / self.area_size,
            agent['z'] / 300.0,
            agent['vx'] / agent['max_speed'],
            agent['vy'] / agent['max_speed'],
            agent['vz'] / agent['max_speed'] if 'vz' in agent else 0.0,
            agent.get('energy', 1.0) / agent.get('max_energy', 1.0),
            agent['coverage_radius'] / 300.0,
            float(['satellite', 'uav', 'ground_station'].index(agent['type']))
        ]).astype(np.float32)
        
        # Visible POIs (within observation radius)
        observation_radius = 300
        visible_pois = []
        
        for poi in self.pois:
            distance = np.linalg.norm([poi.x - agent['x'], poi.y - agent['y']])
            if distance <= observation_radius:
                relative_x = (poi.x - agent['x']) / observation_radius
                relative_y = (poi.y - agent['y']) / observation_radius
                
                poi_features = np.array([
                    relative_x,
                    relative_y,
                    distance / observation_radius,
                    poi.priority / 5.0,
                    float(poi.covered),
                    poi.coverage_time / 100.0
                ]).astype(np.float32)
                
                visible_pois.append(poi_features)
        
        # Pad or truncate to fixed size
        max_visible_pois = 8
        poi_observation = np.zeros((max_visible_pois, 6), dtype=np.float32)
        for i, poi_features in enumerate(visible_pois[:max_visible_pois]):
            poi_observation[i] = poi_features
        
        # Visible other agents
        visible_agents = []
        for other_agent in self.agents:
            if other_agent['id'] != agent['id']:
                distance = np.linalg.norm([other_agent['x'] - agent['x'], 
                                          other_agent['y'] - agent['y']])
                if distance <= observation_radius:
                    relative_x = (other_agent['x'] - agent['x']) / observation_radius
                    relative_y = (other_agent['y'] - agent['y']) / observation_radius
                    
                    agent_features = np.array([
                        relative_x,
                        relative_y,
                        (other_agent['z'] - agent['z']) / 300.0,
                        distance / observation_radius,
                        other_agent['vx'] / other_agent['max_speed'],
                        other_agent['vy'] / other_agent['max_speed'],
                        float(['satellite', 'uav', 'ground_station'].index(other_agent['type'])),
                        float(distance < self.effective_comm_range)
                    ]).astype(np.float32)
                    
                    visible_agents.append(agent_features)
        
        # Pad or truncate agents
        max_visible_agents = 6
        agent_observation = np.zeros((max_visible_agents, 8), dtype=np.float32)
        for i, agent_features in enumerate(visible_agents[:max_visible_agents]):
            agent_observation[i] = agent_features
        
        # Global information
        total_pois_covered = sum(poi.covered for poi in self.pois)
        coverage_rate = total_pois_covered / len(self.pois) if self.pois else 0
        
        avg_energy = np.mean([agent.get('energy', 1.0) / agent.get('max_energy', 1.0) 
                             for agent in self.agents])
        
        global_info = np.array([
            coverage_rate,
            self.current_step / self.max_episode_steps,
            avg_energy,
            len([a for a in self.agents if a['active']]) / len(self.agents),
            self.environmental_conditions.weather_factor,
            self.environmental_conditions.communication_noise
        ]).astype(np.float32)
        
        return {
            'self_state': self_state,
            'visible_pois': poi_observation.flatten(),
            'visible_agents': agent_observation.flatten(), 
            'global_info': global_info
        }
    
    def get_robustness_analysis(self) -> Dict[str, Any]:
        """Get comprehensive robustness analysis"""
        if not self.robustness_metrics['scenario_difficulty']:
            return {}
        
        analysis = {
            'average_scenario_difficulty': np.mean(self.robustness_metrics['scenario_difficulty']),
            'difficulty_variance': np.var(self.robustness_metrics['scenario_difficulty']),
            'difficulty_range': (
                np.min(self.robustness_metrics['scenario_difficulty']),
                np.max(self.robustness_metrics['scenario_difficulty'])
            ),
            'episodes_analyzed': len(self.robustness_metrics['scenario_difficulty'])
        }
        
        return analysis
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Enhanced step with multi-objective rewards"""
        
        # Store previous coverage state
        current_coverage = [poi.covered for poi in self.pois] if hasattr(self, 'pois') and self.pois else []
        
        # Convert our agent representation for compatibility
        agent_dict = {}
        if hasattr(self, 'agents') and self.agents:
            if isinstance(self.agents, list):
                for i, agent in enumerate(self.agents):
                    agent_dict[i] = agent
            elif isinstance(self.agents, dict):
                agent_dict = self.agents
        
        # Apply actions and update environment (simplified step logic)
        self._apply_actions_to_agents(actions)
        self._update_coverage_status()
        self._update_energy_levels()
        
        # Get observations
        observations = self._get_structured_observations()
        
        # Check if done
        done = self._check_episode_complete()
        dones = {i: done for i in range(len(agent_dict))}
        
        # Compute multi-objective rewards
        individual_rewards, reward_breakdown = self.reward_system.compute_multi_objective_rewards(
            agents=agent_dict,
            pois=self.pois if hasattr(self, 'pois') else [],
            obstacles=getattr(self, 'obstacles', []),
            previous_coverage=self.previous_coverage,
            communication_network=self._get_communication_network_state()
        )
        
        # Update previous coverage for next step
        self.previous_coverage = current_coverage.copy()
        
        # Create info
        info = {
            'reward_breakdown': reward_breakdown,
            'multi_objective_rewards': individual_rewards,
            'reward_scaling_factors': self.reward_system.current_scaling.copy(),
            'coverage_status': current_coverage
        }
        
        return observations, individual_rewards, dones, info
    
    def _apply_actions_to_agents(self, actions: Dict[int, np.ndarray]):
        """Apply actions to agents"""
        if not hasattr(self, 'agents') or not self.agents:
            return
        
        for agent_id, action in actions.items():
            if agent_id < len(self.agents):
                agent = self.agents[agent_id]
                
                # Update velocity based on action
                max_speed = agent.get('max_speed', 5.0)
                agent['vx'] = np.clip(action[0], -1, 1) * max_speed
                agent['vy'] = np.clip(action[1], -1, 1) * max_speed
                
                # Update position
                agent['x'] += agent['vx']
                agent['y'] += agent['vy']
                
                # Keep within bounds
                agent['x'] = np.clip(agent['x'], 0, self.area_size)
                agent['y'] = np.clip(agent['y'], 0, self.area_size)
    
    def _update_coverage_status(self):
        """Update POI coverage status"""
        if not hasattr(self, 'pois') or not self.pois:
            return
        
        for poi in self.pois:
            poi.covered = False
            
            for agent in self.agents:
                if not agent.get('active', True):
                    continue
                
                agent_pos = np.array([agent['x'], agent['y']])
                poi_pos = np.array([poi.x, poi.y])
                distance = np.linalg.norm(agent_pos - poi_pos)
                
                coverage_radius = agent.get('coverage_radius', 50.0)
                if distance <= coverage_radius:
                    poi.covered = True
                    poi.coverage_time += 1
                    break
    
    def _update_energy_levels(self):
        """Update energy levels for UAVs"""
        if not hasattr(self, 'agents'):
            return
        
        for agent in self.agents:
            if agent.get('type') == 'uav' and agent.get('active', True):
                current_energy = agent.get('energy', 1000)
                consumption_rate = agent.get('energy_consumption_rate', 5)
                
                # Energy consumption based on movement
                velocity = np.array([agent.get('vx', 0), agent.get('vy', 0)])
                speed = np.linalg.norm(velocity)
                energy_consumption = consumption_rate * (1 + speed / 10)
                
                agent['energy'] = max(0, current_energy - energy_consumption)
                
                # Deactivate if out of energy
                if agent['energy'] <= 0:
                    agent['active'] = False
    
    def _check_episode_complete(self) -> bool:
        """Check if episode is complete"""
        if not hasattr(self, 'current_step'):
            self.current_step = 0
        
        self.current_step += 1
        
        # Episode length limit
        if self.current_step >= getattr(self, 'max_episode_steps', 200):
            return True
        
        # All POIs covered
        if hasattr(self, 'pois') and self.pois:
            if all(poi.covered for poi in self.pois):
                return True
        
        # All UAVs out of energy
        if hasattr(self, 'agents'):
            active_uavs = [agent for agent in self.agents 
                          if agent.get('type') == 'uav' and agent.get('active', True)]
            if len([a for a in self.agents if a.get('type') == 'uav']) > 0 and len(active_uavs) == 0:
                return True
        
        return False
    
    def update_training_progress(self, episode: int, total_episodes: int):
        """Update training progress for adaptive reward scaling"""
        self.reward_system.update_training_progress(episode, total_episodes)
    
    def _get_communication_network_state(self) -> Dict[str, Any]:
        """Get current communication network state for reward computation"""
        
        if not hasattr(self, 'agents') or not self.agents:
            return {'efficiency': 0.0, 'active_links': [], 'data_transmitted': 0.0}
        
        active_agents = [agent for agent in self.agents if agent.get('active', True)]
        
        if len(active_agents) < 2:
            return {'efficiency': 1.0, 'active_links': [], 'data_transmitted': 0.0}
        
        # Calculate network efficiency based on connectivity
        agent_positions = np.array([(agent['x'], agent['y']) for agent in active_agents])
        distances = cdist(agent_positions, agent_positions)
        np.fill_diagonal(distances, np.inf)
        
        comm_range = self.effective_comm_range
        connected_pairs = np.sum(distances <= comm_range) // 2  # Avoid double counting
        max_pairs = len(active_agents) * (len(active_agents) - 1) // 2
        
        efficiency = connected_pairs / max_pairs if max_pairs > 0 else 1.0
        
        # Simulate active communication links
        active_links = []
        for i, agent in enumerate(active_agents):
            agent_id = agent.get('id', i)
            if distances[i].min() <= comm_range:  # Has at least one connection
                active_links.append(agent_id)
        
        # Simulate data transmission based on network activity
        data_transmitted = efficiency * len(active_links) * 10.0  # Arbitrary units
        
        return {
            'efficiency': efficiency,
            'active_links': active_links,
            'data_transmitted': data_transmitted
        }
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis including robustness and reward metrics"""
        
        robustness_analysis = self.get_robustness_analysis()
        reward_analysis = self.reward_system.get_reward_analysis()
        
        return {
            'robustness_metrics': robustness_analysis,
            'reward_analysis': reward_analysis,
            'environment_stats': {
                'total_episodes': len(self.robustness_metrics['scenario_difficulty']),
                'average_difficulty': np.mean(self.robustness_metrics['scenario_difficulty']) if self.robustness_metrics['scenario_difficulty'] else 0,
                'current_conditions': {
                    'weather_factor': self.environmental_conditions.weather_factor,
                    'communication_noise': self.environmental_conditions.communication_noise,
                    'wind_speed': self.environmental_conditions.wind_speed
                }
            }
        }