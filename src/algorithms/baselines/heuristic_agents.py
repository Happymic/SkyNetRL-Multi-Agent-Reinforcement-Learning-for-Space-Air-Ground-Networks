"""
Heuristic and Random Baseline Agents
Simple baseline methods for comparison with learned policies
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import random


class GreedyHeuristicAgent:
    """Greedy heuristic agent that moves to nearest uncovered POI"""
    
    def __init__(self, agent_id: int, config: Dict):
        """
        Initialize greedy heuristic agent
        
        Args:
            agent_id: Unique agent identifier
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        
        # Determine agent type and capabilities
        self.agent_type = self._determine_agent_type(agent_id, config)
        self.max_speed = self._get_max_speed()
        self.coverage_radius = self._get_coverage_radius()
        
        # Agent state
        self.position = np.array([0.0, 0.0])
        self.energy = config.get('uav_energy_capacity', 1200) if self.agent_type == 'uav' else None
        self.low_energy_threshold = 0.3  # Return to charge when below 30%
        
        # Statistics
        self.total_distance_traveled = 0.0
        self.pois_covered = set()
        self.charging_visits = 0
    
    def _determine_agent_type(self, agent_id: int, config: Dict) -> str:
        """Determine agent type based on ID"""
        num_satellites = config.get('num_satellites', 1)
        num_uavs = config.get('num_uavs', 3)
        
        if agent_id < num_satellites:
            return 'satellite'
        elif agent_id < num_satellites + num_uavs:
            return 'uav'
        else:
            return 'ground_station'
    
    def _get_max_speed(self) -> float:
        """Get maximum speed based on agent type"""
        if self.agent_type == 'satellite':
            return self.config.get('satellite_max_speed', 3)
        elif self.agent_type == 'uav':
            return self.config.get('uav_max_speed', 6)
        else:  # ground_station
            return self.config.get('ground_station_max_speed', 2)
    
    def _get_coverage_radius(self) -> float:
        """Get coverage radius based on agent type"""
        if self.agent_type == 'satellite':
            return self.config.get('satellite_coverage_radius', 250)
        elif self.agent_type == 'uav':
            return self.config.get('uav_coverage_radius', 120)
        else:  # ground_station
            return self.config.get('ground_station_coverage_radius', 80)
    
    def act(self, obs: np.ndarray, env_info: Optional[Dict] = None) -> np.ndarray:
        """
        Generate greedy action based on observation
        
        Args:
            obs: Enhanced observation array [184 dimensions]
            env_info: Optional environment information
        
        Returns:
            action: Velocity command [vx, vy]
        """
        # Parse enhanced observation
        self_obs, spatial_obs, agent_obs, task_obs = self._parse_observation(obs)
        
        # Update internal state
        self.position = self_obs[:2] * self.config.get('area_size', 1000)  # Denormalize position
        if self.agent_type == 'uav':
            self.energy = self_obs[4] * self.config.get('uav_energy_capacity', 1200)  # Denormalize energy
        
        # Check if UAV needs to charge
        if self.agent_type == 'uav' and self.energy is not None:
            if self.energy / self.config.get('uav_energy_capacity', 1200) < self.low_energy_threshold:
                return self._move_to_nearest_charging_station(spatial_obs)
        
        # Find nearest uncovered high-priority POI
        target_poi = self._find_best_target_poi(spatial_obs, task_obs)
        
        if target_poi is not None:
            # Move toward target POI
            direction = target_poi - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Normalize direction and scale by max speed
                velocity = (direction / distance) * self.max_speed
                
                # Normalize to [-1, 1] range for action space
                velocity = np.clip(velocity / self.max_speed, -1, 1)
                
                # Update statistics
                self.total_distance_traveled += np.linalg.norm(velocity * self.max_speed)
                
                return velocity
        
        # No target found, stay in place or random walk
        if np.random.random() < 0.1:  # 10% chance of random exploration
            return np.random.uniform(-0.2, 0.2, 2)
        else:
            return np.array([0.0, 0.0])
    
    def _parse_observation(self, obs: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Parse enhanced observation into components"""
        idx = 0
        
        # Self observation [9]
        self_obs = obs[idx:idx + 9]
        idx += 9
        
        # Spatial observation [80] 
        spatial_obs = obs[idx:idx + 80]
        idx += 80
        
        # Agent observation [90]
        agent_obs = obs[idx:idx + 90]
        idx += 90
        
        # Task observation [5]
        task_obs = obs[idx:idx + 5]
        
        return self_obs, spatial_obs, agent_obs, task_obs
    
    def _find_best_target_poi(self, spatial_obs: np.ndarray, task_obs: np.ndarray) -> Optional[np.ndarray]:
        """Find the best POI to target based on distance and priority"""
        area_size = self.config.get('area_size', 1000)
        
        # Parse spatial observations (first 10 slots are POIs)
        best_poi = None
        best_score = -float('inf')
        
        for i in range(10):  # First 10 objects are POIs
            poi_data = spatial_obs[i*4:(i+1)*4]
            
            if np.sum(poi_data) == 0:  # Empty slot
                continue
            
            poi_x = poi_data[0] * area_size  # Denormalize
            poi_y = poi_data[1] * area_size
            poi_priority = poi_data[2] * 5.0  # Denormalize priority
            poi_covered = poi_data[3]
            
            if poi_covered > 0.5:  # Already covered
                continue
            
            poi_position = np.array([poi_x, poi_y])
            distance = np.linalg.norm(poi_position - self.position)
            
            # Score based on priority and distance (higher priority, closer distance = higher score)
            if distance > 0:
                score = poi_priority / distance
                
                # Bonus for high-priority POIs
                if poi_priority >= 4.0:
                    score *= 2.0
                
                # Consider agent's coverage radius
                if distance <= self.coverage_radius:
                    score *= 1.5  # Bonus for POIs within coverage
                
                if score > best_score:
                    best_score = score
                    best_poi = poi_position
        
        return best_poi
    
    def _move_to_nearest_charging_station(self, spatial_obs: np.ndarray) -> np.ndarray:
        """Move to nearest charging station"""
        area_size = self.config.get('area_size', 1000)
        
        # Parse charging stations (last 5 slots in spatial obs)
        nearest_station = None
        min_distance = float('inf')
        
        for i in range(5):  # Last 5 objects are charging stations
            station_idx = 15 + i  # Charging stations start at index 15
            station_data = spatial_obs[station_idx*4:(station_idx+1)*4]
            
            if np.sum(station_data) == 0:  # Empty slot
                continue
            
            station_x = station_data[0] * area_size
            station_y = station_data[1] * area_size
            station_available = station_data[2]
            
            if station_available < 0.5:  # Station not available
                continue
            
            station_position = np.array([station_x, station_y])
            distance = np.linalg.norm(station_position - self.position)
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = station_position
        
        if nearest_station is not None:
            # Move toward charging station
            direction = nearest_station - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                velocity = (direction / distance) * self.max_speed
                velocity = np.clip(velocity / self.max_speed, -1, 1)
                
                # Update statistics
                if distance <= 30:  # Close to charging station
                    self.charging_visits += 1
                
                return velocity
        
        # No charging station found, stay in place
        return np.array([0.0, 0.0])
    
    def reset(self):
        """Reset agent state"""
        self.position = np.array([0.0, 0.0])
        if self.agent_type == 'uav':
            self.energy = self.config.get('uav_energy_capacity', 1200)
        
        # Reset statistics
        self.total_distance_traveled = 0.0
        self.pois_covered.clear()
        self.charging_visits = 0
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            'agent_type': self.agent_type,
            'total_distance_traveled': self.total_distance_traveled,
            'pois_covered': len(self.pois_covered),
            'charging_visits': self.charging_visits,
            'current_energy': self.energy if self.agent_type == 'uav' else None
        }
    
    def set_eval_mode(self):
        """Set agent to evaluation mode (for consistency with RL agents)"""
        pass  # Heuristic agents don't have train/eval modes
    
    def set_train_mode(self):
        """Set agent to training mode (for consistency with RL agents)"""
        pass  # Heuristic agents don't have train/eval modes


class RandomPolicyAgent:
    """Random policy agent for baseline comparison"""
    
    def __init__(self, agent_id: int, config: Dict):
        """
        Initialize random policy agent
        
        Args:
            agent_id: Unique agent identifier
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        
        # Action bounds
        self.action_bounds = config.get('action_bounds', [-1, 1])
        self.action_std = config.get('random_action_std', 0.5)
        
        # Agent type for statistics
        self.agent_type = self._determine_agent_type(agent_id, config)
        
        # Statistics
        self.total_actions = 0
        self.episode_steps = 0
    
    def _determine_agent_type(self, agent_id: int, config: Dict) -> str:
        """Determine agent type based on ID"""
        num_satellites = config.get('num_satellites', 1)
        num_uavs = config.get('num_uavs', 3)
        
        if agent_id < num_satellites:
            return 'satellite'
        elif agent_id < num_satellites + num_uavs:
            return 'uav'
        else:
            return 'ground_station'
    
    def act(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Generate random action
        
        Args:
            obs: Observation (ignored for random policy)
            add_noise: Whether to add noise (ignored)
        
        Returns:
            action: Random action array
        """
        self.total_actions += 1
        
        # Generate random action in bounds
        action = np.random.uniform(
            self.action_bounds[0], 
            self.action_bounds[1], 
            size=2
        )
        
        # Add some structure - bias toward smaller actions for stability
        if np.random.random() < 0.3:  # 30% chance of larger action
            action = action * self.action_std
        else:
            action = action * (self.action_std * 0.3)
        
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])
    
    def reset(self):
        """Reset agent state"""
        self.episode_steps = 0
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            'agent_type': self.agent_type,
            'total_actions': self.total_actions,
            'episode_steps': self.episode_steps
        }
    
    def set_eval_mode(self):
        """Set agent to evaluation mode (for consistency with RL agents)"""
        pass  # Heuristic agents don't have train/eval modes
    
    def set_train_mode(self):
        """Set agent to training mode (for consistency with RL agents)"""
        pass  # Heuristic agents don't have train/eval modes


class AdaptiveGreedyAgent(GreedyHeuristicAgent):
    """Adaptive greedy agent with learning-like behavior"""
    
    def __init__(self, agent_id: int, config: Dict):
        super().__init__(agent_id, config)
        
        # Adaptive parameters
        self.exploration_rate = config.get('adaptive_exploration_rate', 0.2)
        self.exploration_decay = config.get('adaptive_exploration_decay', 0.995)
        self.min_exploration = config.get('adaptive_min_exploration', 0.05)
        
        # Memory of successful actions
        self.action_memory = []
        self.success_memory = []
        self.memory_size = 100
        
        # Performance tracking
        self.recent_rewards = []
        self.reward_window = 50
    
    def act(self, obs: np.ndarray, env_info: Optional[Dict] = None) -> np.ndarray:
        """
        Generate adaptive greedy action
        
        Args:
            obs: Enhanced observation array
            env_info: Optional environment information
        
        Returns:
            action: Velocity command
        """
        # Exploration vs exploitation decision
        if np.random.random() < self.exploration_rate:
            # Exploration: add noise to greedy action or random action
            if len(self.action_memory) > 10:
                # Use successful action from memory with noise
                success_actions = [self.action_memory[i] for i, success in enumerate(self.success_memory) if success]
                if success_actions:
                    base_action = random.choice(success_actions)
                    noise = np.random.normal(0, 0.2, 2)
                    action = np.clip(base_action + noise, -1, 1)
                    return action
            
            # Random exploration
            return np.random.uniform(-0.5, 0.5, 2)
        else:
            # Exploitation: use greedy strategy
            return super().act(obs, env_info)
    
    def update_memory(self, action: np.ndarray, reward: float):
        """Update action memory based on reward"""
        success = reward > 0  # Simple success criterion
        
        # Add to memory
        self.action_memory.append(action.copy())
        self.success_memory.append(success)
        
        # Maintain memory size
        if len(self.action_memory) > self.memory_size:
            self.action_memory.pop(0)
            self.success_memory.pop(0)
        
        # Track recent performance
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window:
            self.recent_rewards.pop(0)
        
        # Adapt exploration rate based on recent performance
        if len(self.recent_rewards) >= 10:
            avg_recent_reward = np.mean(self.recent_rewards[-10:])
            if avg_recent_reward > np.mean(self.recent_rewards):
                # Performance improving, reduce exploration
                self.exploration_rate *= self.exploration_decay
            else:
                # Performance declining, increase exploration
                self.exploration_rate = min(0.5, self.exploration_rate * 1.01)
        
        self.exploration_rate = max(self.min_exploration, self.exploration_rate)
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        # Don't reset memory - let it persist across episodes for learning
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        base_stats = super().get_statistics()
        base_stats.update({
            'exploration_rate': self.exploration_rate,
            'memory_size': len(self.action_memory),
            'success_rate': np.mean(self.success_memory) if self.success_memory else 0.0,
            'avg_recent_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        })
        return base_stats


class CoordinatedHeuristicAgent(GreedyHeuristicAgent):
    """Heuristic agent with simple coordination"""
    
    def __init__(self, agent_id: int, config: Dict):
        super().__init__(agent_id, config)
        
        # Coordination parameters
        self.coordination_range = config.get('coordination_range', 200)
        self.avoid_overlap_factor = config.get('avoid_overlap_factor', 1.5)
        
        # Shared information (would be updated externally in practice)
        self.other_agents_info = {}
    
    def act(self, obs: np.ndarray, env_info: Optional[Dict] = None) -> np.ndarray:
        """
        Generate coordinated action considering other agents
        
        Args:
            obs: Enhanced observation array
            env_info: Optional environment information
        
        Returns:
            action: Velocity command
        """
        # Parse observation to get other agents' positions
        self_obs, spatial_obs, agent_obs, task_obs = self._parse_observation(obs)
        other_agents_positions = self._parse_other_agents(agent_obs)
        
        # Get base greedy action
        base_action = super().act(obs, env_info)
        
        # Apply coordination adjustment
        coordinated_action = self._apply_coordination(base_action, other_agents_positions, spatial_obs)
        
        return coordinated_action
    
    def _parse_other_agents(self, agent_obs: np.ndarray) -> List[np.ndarray]:
        """Parse other agents' positions from observation"""
        area_size = self.config.get('area_size', 1000)
        positions = []
        
        for i in range(10):  # Up to 10 other agents
            agent_data = agent_obs[i*9:(i+1)*9]
            
            if np.sum(agent_data) == 0:  # Empty slot
                continue
            
            # Relative position in observation, convert to absolute
            rel_pos = agent_data[:2] * self.coordination_range  # Denormalize relative position
            abs_pos = self.position + rel_pos
            positions.append(abs_pos)
        
        return positions
    
    def _apply_coordination(self, base_action: np.ndarray, other_positions: List[np.ndarray], 
                          spatial_obs: np.ndarray) -> np.ndarray:
        """Apply coordination adjustment to base action"""
        if not other_positions:
            return base_action
        
        # Calculate repulsion from nearby agents
        repulsion = np.array([0.0, 0.0])
        
        for other_pos in other_positions:
            distance = np.linalg.norm(other_pos - self.position)
            
            # Apply repulsion if too close
            if 0 < distance < self.coverage_radius * self.avoid_overlap_factor:
                # Repulsion inversely proportional to distance
                repulsion_direction = (self.position - other_pos) / distance
                repulsion_strength = (self.coverage_radius * self.avoid_overlap_factor - distance) / distance
                repulsion += repulsion_direction * repulsion_strength * 0.3
        
        # Combine base action with coordination adjustment
        coordinated_action = base_action + repulsion
        
        # Normalize to action bounds
        coordinated_action = np.clip(coordinated_action, -1, 1)
        
        return coordinated_action