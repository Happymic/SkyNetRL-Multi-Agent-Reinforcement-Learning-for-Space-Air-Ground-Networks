"""
Multi-Objective Reward System for SAGIN Environment
Implements sophisticated reward mechanisms with adaptive scaling and multiple objectives
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
import math


@dataclass
class RewardWeights:
    """Configurable reward weights for different objectives"""
    
    # Primary objectives
    coverage_reward: float = 2.0
    priority_bonus: float = 1.5
    efficiency_reward: float = 1.0
    cooperation_reward: float = 0.8
    
    # Penalties
    energy_penalty: float = 0.3
    collision_penalty: float = 2.0
    redundancy_penalty: float = 0.5
    time_penalty: float = 0.1
    
    # Exploration incentives
    exploration_bonus: float = 0.4
    diversity_bonus: float = 0.3
    
    # Communication rewards
    network_efficiency_reward: float = 0.6
    information_sharing_reward: float = 0.4


@dataclass
class AdaptiveScaling:
    """Adaptive scaling parameters for reward components"""
    
    # Training phase adaptation
    early_phase_threshold: float = 0.3  # 30% of training
    late_phase_threshold: float = 0.8   # 80% of training
    
    # Dynamic scaling factors
    coverage_scaling_range: Tuple[float, float] = (0.8, 2.0)
    exploration_scaling_range: Tuple[float, float] = (2.0, 0.2)
    cooperation_scaling_range: Tuple[float, float] = (0.5, 1.5)
    
    # Performance-based adaptation
    performance_adaptation_rate: float = 0.1
    min_adaptation_episodes: int = 20


class MultiObjectiveRewardSystem:
    """Advanced multi-objective reward system with adaptive scaling"""
    
    def __init__(self, config: Dict, num_agents: int, area_size: float):
        """
        Initialize multi-objective reward system
        
        Args:
            config: Configuration dictionary
            num_agents: Number of agents in environment
            area_size: Size of the operational area
        """
        self.config = config
        self.num_agents = num_agents
        self.area_size = area_size
        
        # Initialize reward weights
        reward_config = config.get('rewards', {})
        self.weights = RewardWeights(**{k: v for k, v in reward_config.items() 
                                      if hasattr(RewardWeights, k)})
        
        # Initialize adaptive scaling
        self.adaptive_scaling = AdaptiveScaling(**config.get('adaptive_scaling', {}))
        
        # Training progress tracking
        self.episode_count = 0
        self.total_episodes = config.get('num_episodes', 1000)
        self.training_progress = 0.0
        
        # Performance history for adaptation
        self.performance_history = {
            'coverage_rates': [],
            'cooperation_scores': [],
            'efficiency_scores': [],
            'exploration_scores': []
        }
        
        # Current scaling factors
        self.current_scaling = {
            'coverage': 1.0,
            'exploration': 1.0,
            'cooperation': 1.0,
            'efficiency': 1.0
        }
        
        # Visited positions for exploration tracking
        self.visited_positions = {}
        self.exploration_grid_size = 50  # Grid cells for tracking exploration
        self.exploration_grid = np.zeros((self.exploration_grid_size, self.exploration_grid_size))
        
        print("🎯 Multi-Objective Reward System initialized")
        print(f"   📊 Tracking {len(self.performance_history)} performance metrics")
        print(f"   🎛️  Adaptive scaling enabled with {self.adaptive_scaling.min_adaptation_episodes} episode warmup")
    
    def update_training_progress(self, episode: int, total_episodes: int):
        """Update training progress for adaptive scaling"""
        self.episode_count = episode
        self.total_episodes = total_episodes
        self.training_progress = episode / total_episodes
        
        # Update adaptive scaling factors
        self._update_scaling_factors()
    
    def _update_scaling_factors(self):
        """Update adaptive scaling factors based on training progress and performance"""
        
        progress = self.training_progress
        
        # Training phase-based scaling
        if progress < self.adaptive_scaling.early_phase_threshold:
            # Early phase: emphasize exploration
            exploration_factor = self._interpolate(
                self.adaptive_scaling.exploration_scaling_range, 
                progress / self.adaptive_scaling.early_phase_threshold
            )
            coverage_factor = self._interpolate(
                self.adaptive_scaling.coverage_scaling_range,
                progress / self.adaptive_scaling.early_phase_threshold
            )
            cooperation_factor = self.adaptive_scaling.cooperation_scaling_range[0]
            
        elif progress > self.adaptive_scaling.late_phase_threshold:
            # Late phase: emphasize performance and cooperation
            late_progress = (progress - self.adaptive_scaling.late_phase_threshold) / \
                          (1.0 - self.adaptive_scaling.late_phase_threshold)
            
            exploration_factor = self.adaptive_scaling.exploration_scaling_range[1]
            coverage_factor = self.adaptive_scaling.coverage_scaling_range[1]
            cooperation_factor = self._interpolate(
                self.adaptive_scaling.cooperation_scaling_range,
                late_progress
            )
            
        else:
            # Middle phase: balanced approach
            mid_progress = (progress - self.adaptive_scaling.early_phase_threshold) / \
                         (self.adaptive_scaling.late_phase_threshold - self.adaptive_scaling.early_phase_threshold)
            
            exploration_factor = self._interpolate(
                (self.adaptive_scaling.exploration_scaling_range[0], 
                 self.adaptive_scaling.exploration_scaling_range[1]), 
                mid_progress
            )
            coverage_factor = self._interpolate(
                self.adaptive_scaling.coverage_scaling_range,
                mid_progress
            )
            cooperation_factor = self._interpolate(
                self.adaptive_scaling.cooperation_scaling_range,
                mid_progress
            )
        
        # Performance-based adaptation
        if len(self.performance_history['coverage_rates']) >= self.adaptive_scaling.min_adaptation_episodes:
            recent_performance = np.mean(self.performance_history['coverage_rates'][-10:])
            historical_performance = np.mean(self.performance_history['coverage_rates'][:-10])
            
            if historical_performance > 0:
                performance_ratio = recent_performance / historical_performance
                adaptation = (performance_ratio - 1.0) * self.adaptive_scaling.performance_adaptation_rate
                
                # Adapt coverage scaling based on recent performance
                coverage_factor *= (1.0 + adaptation)
                coverage_factor = np.clip(coverage_factor, 0.5, 3.0)
        
        # Update current scaling factors
        self.current_scaling.update({
            'exploration': exploration_factor,
            'coverage': coverage_factor,
            'cooperation': cooperation_factor,
            'efficiency': 1.0  # Keep efficiency scaling stable
        })
    
    def _interpolate(self, range_tuple: Tuple[float, float], t: float) -> float:
        """Linear interpolation between two values"""
        t = np.clip(t, 0.0, 1.0)
        return range_tuple[0] + t * (range_tuple[1] - range_tuple[0])
    
    def compute_multi_objective_rewards(
        self, 
        agents: Dict, 
        pois: List, 
        obstacles: List,
        previous_coverage: Optional[List[bool]] = None,
        communication_network: Optional[Dict] = None
    ) -> Tuple[Dict[int, float], Dict[str, float]]:
        """
        Compute comprehensive multi-objective rewards
        
        Args:
            agents: Dictionary of agent states
            pois: List of POI objects
            obstacles: List of obstacle objects
            previous_coverage: Previous episode's coverage for comparison
            communication_network: Current communication network state
            
        Returns:
            Tuple of (individual rewards, reward breakdown)
        """
        
        # Initialize reward components
        reward_components = {
            'coverage': 0.0,
            'priority': 0.0,
            'efficiency': 0.0,
            'cooperation': 0.0,
            'exploration': 0.0,
            'energy_penalty': 0.0,
            'collision_penalty': 0.0,
            'communication': 0.0,
            'diversity': 0.0
        }
        
        individual_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
        
        # 1. Coverage Rewards with Priority Weighting
        coverage_rewards, coverage_metrics = self._compute_coverage_rewards(agents, pois, previous_coverage)
        reward_components['coverage'] = coverage_metrics['total_coverage_reward']
        reward_components['priority'] = coverage_metrics['priority_bonus']
        
        # 2. Efficiency Rewards
        efficiency_rewards, efficiency_score = self._compute_efficiency_rewards(agents, pois)
        reward_components['efficiency'] = efficiency_score
        
        # 3. Cooperation Rewards
        cooperation_rewards, cooperation_score = self._compute_cooperation_rewards(agents, pois)
        reward_components['cooperation'] = cooperation_score
        
        # 4. Exploration Rewards
        exploration_rewards, exploration_score = self._compute_exploration_rewards(agents)
        reward_components['exploration'] = exploration_score
        
        # 5. Communication Rewards
        comm_rewards, comm_score = self._compute_communication_rewards(agents, communication_network)
        reward_components['communication'] = comm_score
        
        # 6. Diversity Rewards
        diversity_rewards, diversity_score = self._compute_diversity_rewards(agents, pois)
        reward_components['diversity'] = diversity_score
        
        # 7. Penalties
        energy_penalties = self._compute_energy_penalties(agents)
        collision_penalties = self._compute_collision_penalties(agents, obstacles)
        
        reward_components['energy_penalty'] = sum(energy_penalties.values())
        reward_components['collision_penalty'] = sum(collision_penalties.values())
        
        # Combine all rewards with adaptive scaling
        for agent_id in agents.keys():
            individual_rewards[agent_id] = (
                coverage_rewards.get(agent_id, 0.0) * self.current_scaling['coverage'] +
                efficiency_rewards.get(agent_id, 0.0) * self.current_scaling['efficiency'] +
                cooperation_rewards.get(agent_id, 0.0) * self.current_scaling['cooperation'] +
                exploration_rewards.get(agent_id, 0.0) * self.current_scaling['exploration'] +
                comm_rewards.get(agent_id, 0.0) +
                diversity_rewards.get(agent_id, 0.0) -
                energy_penalties.get(agent_id, 0.0) -
                collision_penalties.get(agent_id, 0.0)
            )
        
        # Update performance history
        self._update_performance_history(coverage_metrics, cooperation_score, efficiency_score, exploration_score)
        
        return individual_rewards, reward_components
    
    def _compute_coverage_rewards(
        self, 
        agents: Dict, 
        pois: List, 
        previous_coverage: Optional[List[bool]]
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        """Compute coverage-based rewards with priority weighting"""
        
        coverage_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
        
        total_coverage_reward = 0.0
        priority_bonus = 0.0
        new_coverage_count = 0
        high_priority_coverage = 0
        
        current_coverage = [poi.covered for poi in pois]
        
        for i, poi in enumerate(pois):
            if poi.covered:
                # Base coverage reward
                base_reward = self.weights.coverage_reward * poi.importance
                
                # Priority multiplier
                priority_multiplier = 1.0 + (poi.priority - 1) * 0.2  # Priority 1-5 → multiplier 1.0-1.8
                coverage_reward = base_reward * priority_multiplier
                
                # New coverage bonus
                if previous_coverage is None or not previous_coverage[i]:
                    coverage_reward *= 1.5  # 50% bonus for new coverage
                    new_coverage_count += 1
                
                # High priority bonus
                if poi.priority >= 4:
                    high_priority_coverage += 1
                    priority_bonus += self.weights.priority_bonus * poi.priority
                
                # Distribute reward to covering agents
                covering_agents = self._find_covering_agents(poi, agents)
                if covering_agents:
                    reward_per_agent = coverage_reward / len(covering_agents)
                    for agent_id in covering_agents:
                        coverage_rewards[agent_id] += reward_per_agent
                
                total_coverage_reward += coverage_reward
        
        # Team coverage bonuses
        coverage_rate = sum(current_coverage) / len(pois) if pois else 0
        if coverage_rate >= 0.9:  # 90% coverage bonus
            team_bonus = 5.0
            for agent_id in coverage_rewards.keys():
                coverage_rewards[agent_id] += team_bonus / len(agents)
        
        metrics = {
            'total_coverage_reward': total_coverage_reward,
            'priority_bonus': priority_bonus,
            'new_coverage_count': new_coverage_count,
            'high_priority_coverage': high_priority_coverage,
            'coverage_rate': coverage_rate
        }
        
        return coverage_rewards, metrics
    
    def _compute_efficiency_rewards(self, agents: Dict, pois: List) -> Tuple[Dict[int, float], float]:
        """Compute efficiency-based rewards"""
        
        efficiency_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
        total_efficiency = 0.0
        
        for agent_id, agent in agents.items():
            if not agent.get('active', True):
                continue
            
            agent_pos = np.array([agent['x'], agent['y']])
            
            # Distance efficiency: reward for being close to uncovered POIs
            uncovered_pois = [poi for poi in pois if not poi.covered]
            if uncovered_pois:
                poi_positions = np.array([[poi.x, poi.y] for poi in uncovered_pois])
                distances = np.linalg.norm(poi_positions - agent_pos, axis=1)
                
                # Weighted by POI priority
                priorities = np.array([poi.priority for poi in uncovered_pois])
                weighted_distances = distances / (priorities + 1)  # Closer to high-priority = higher reward
                
                # Inverse distance reward (closer = better)
                min_distance = np.min(weighted_distances)
                distance_efficiency = np.exp(-min_distance / 100.0)  # Exponential decay
                
                efficiency_rewards[agent_id] += self.weights.efficiency_reward * distance_efficiency
                total_efficiency += distance_efficiency
            
            # Movement efficiency: penalize excessive movement
            velocity = np.array([agent.get('vx', 0), agent.get('vy', 0)])
            speed = np.linalg.norm(velocity)
            max_speed = agent.get('max_speed', 10.0)
            
            if speed > max_speed * 0.8:  # Moving too fast penalty
                efficiency_rewards[agent_id] -= 0.1
            elif speed < max_speed * 0.1:  # Standing still penalty (unless at POI)
                # Check if at POI
                at_poi = any(np.linalg.norm(agent_pos - np.array([poi.x, poi.y])) <= 
                           agent.get('coverage_radius', 50) for poi in pois)
                if not at_poi:
                    efficiency_rewards[agent_id] -= 0.05
        
        return efficiency_rewards, total_efficiency
    
    def _compute_cooperation_rewards(self, agents: Dict, pois: List) -> Tuple[Dict[int, float], float]:
        """Compute cooperation-based rewards"""
        
        cooperation_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
        total_cooperation = 0.0
        
        active_agents = {aid: agent for aid, agent in agents.items() if agent.get('active', True)}
        
        if len(active_agents) < 2:
            return cooperation_rewards, 0.0
        
        # Task distribution efficiency
        poi_assignments = {}
        for poi_idx, poi in enumerate(pois):
            covering_agents = self._find_covering_agents(poi, active_agents)
            if covering_agents:
                poi_assignments[poi_idx] = covering_agents
        
        # Reward for balanced workload distribution
        agent_workloads = {agent_id: 0 for agent_id in active_agents.keys()}
        for covering_agents in poi_assignments.values():
            for agent_id in covering_agents:
                agent_workloads[agent_id] += 1
        
        if agent_workloads:
            workload_values = list(agent_workloads.values())
            workload_balance = 1.0 - (np.std(workload_values) / (np.mean(workload_values) + 1e-6))
            
            balance_reward = self.weights.cooperation_reward * workload_balance
            for agent_id in active_agents.keys():
                cooperation_rewards[agent_id] += balance_reward / len(active_agents)
            
            total_cooperation += balance_reward
        
        # Communication-based cooperation
        agent_positions = np.array([[agent['x'], agent['y']] for agent in active_agents.values()])
        agent_ids = list(active_agents.keys())
        
        distances = cdist(agent_positions, agent_positions)
        np.fill_diagonal(distances, np.inf)
        
        # Reward for maintaining communication networks
        comm_range = 200.0  # Communication range
        connected_pairs = 0
        
        for i, agent_id_1 in enumerate(agent_ids):
            for j, agent_id_2 in enumerate(agent_ids[i+1:], i+1):
                if distances[i, j] <= comm_range:
                    connected_pairs += 1
                    # Both agents get cooperation reward for staying connected
                    conn_reward = self.weights.cooperation_reward * 0.1
                    cooperation_rewards[agent_id_1] += conn_reward
                    cooperation_rewards[agent_id_2] += conn_reward
                    total_cooperation += conn_reward * 2
        
        # Network connectivity bonus
        max_connections = len(active_agents) * (len(active_agents) - 1) // 2
        if max_connections > 0:
            connectivity = connected_pairs / max_connections
            if connectivity > 0.7:  # Well-connected network bonus
                network_bonus = self.weights.cooperation_reward * connectivity
                for agent_id in active_agents.keys():
                    cooperation_rewards[agent_id] += network_bonus / len(active_agents)
                total_cooperation += network_bonus
        
        return cooperation_rewards, total_cooperation
    
    def _compute_exploration_rewards(self, agents: Dict) -> Tuple[Dict[int, float], float]:
        """Compute exploration-based rewards"""
        
        exploration_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
        total_exploration = 0.0
        
        for agent_id, agent in agents.items():
            if not agent.get('active', True):
                continue
            
            agent_pos = np.array([agent['x'], agent['y']])
            
            # Convert position to grid coordinates
            grid_x = int(np.clip(agent_pos[0] / self.area_size * self.exploration_grid_size, 
                                0, self.exploration_grid_size - 1))
            grid_y = int(np.clip(agent_pos[1] / self.area_size * self.exploration_grid_size, 
                                0, self.exploration_grid_size - 1))
            
            # Exploration reward for visiting new areas
            if self.exploration_grid[grid_x, grid_y] == 0:
                exploration_reward = self.weights.exploration_bonus
                exploration_rewards[agent_id] += exploration_reward
                total_exploration += exploration_reward
                
                # Mark as visited
                self.exploration_grid[grid_x, grid_y] = 1
            else:
                # Diminishing returns for revisiting areas
                visit_count = self.exploration_grid[grid_x, grid_y]
                revisit_penalty = max(0, self.weights.exploration_bonus * (1.0 / (1 + visit_count)))
                exploration_rewards[agent_id] += revisit_penalty
                total_exploration += revisit_penalty
                
                # Increment visit count
                self.exploration_grid[grid_x, grid_y] += 1
        
        return exploration_rewards, total_exploration
    
    def _compute_communication_rewards(
        self, 
        agents: Dict, 
        communication_network: Optional[Dict]
    ) -> Tuple[Dict[int, float], float]:
        """Compute communication efficiency rewards"""
        
        comm_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
        total_comm_score = 0.0
        
        if not communication_network:
            return comm_rewards, 0.0
        
        # Network efficiency reward
        network_efficiency = communication_network.get('efficiency', 0.0)
        if network_efficiency > 0.8:  # High efficiency bonus
            efficiency_reward = self.weights.network_efficiency_reward * network_efficiency
            
            active_agents = [aid for aid, agent in agents.items() if agent.get('active', True)]
            for agent_id in active_agents:
                comm_rewards[agent_id] += efficiency_reward / len(active_agents)
            
            total_comm_score += efficiency_reward
        
        # Information sharing rewards
        data_shared = communication_network.get('data_transmitted', 0.0)
        if data_shared > 0:
            sharing_reward = self.weights.information_sharing_reward * min(data_shared / 100.0, 1.0)
            
            # Reward agents involved in communication
            active_links = communication_network.get('active_links', [])
            if active_links:
                for agent_id in active_links:
                    if agent_id in agents:
                        comm_rewards[agent_id] += sharing_reward / len(active_links)
                
                total_comm_score += sharing_reward
        
        return comm_rewards, total_comm_score
    
    def _compute_diversity_rewards(self, agents: Dict, pois: List) -> Tuple[Dict[int, float], float]:
        """Compute diversity and specialization rewards"""
        
        diversity_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
        total_diversity = 0.0
        
        # Type diversity bonus
        agent_types = {}
        for agent_id, agent in agents.items():
            agent_type = agent.get('type', 'unknown')
            if agent_type not in agent_types:
                agent_types[agent_type] = []
            agent_types[agent_type].append(agent_id)
        
        # Reward for utilizing different agent types effectively
        if len(agent_types) > 1:
            type_diversity = len(agent_types) / 3.0  # Assume max 3 types
            diversity_bonus = self.weights.diversity_bonus * type_diversity
            
            for agent_id in agents.keys():
                diversity_rewards[agent_id] += diversity_bonus / len(agents)
            
            total_diversity += diversity_bonus
        
        # Role specialization rewards
        for poi in pois:
            if poi.covered:
                covering_agents = self._find_covering_agents(poi, agents)
                if covering_agents:
                    # High priority POIs covered by appropriate agents
                    if poi.priority >= 4:
                        for agent_id in covering_agents:
                            agent = agents[agent_id]
                            # Satellites good for high-priority distant POIs
                            if agent.get('type') == 'satellite':
                                diversity_rewards[agent_id] += self.weights.diversity_bonus * 0.5
                                total_diversity += self.weights.diversity_bonus * 0.5
        
        return diversity_rewards, total_diversity
    
    def _compute_energy_penalties(self, agents: Dict) -> Dict[int, float]:
        """Compute energy-related penalties"""
        
        energy_penalties = {agent_id: 0.0 for agent_id in agents.keys()}
        
        for agent_id, agent in agents.items():
            energy = agent.get('energy')
            if energy is not None:  # Only for agents with limited energy (UAVs)
                max_energy = agent.get('max_energy', agent.get('energy_capacity', 1000))
                energy_ratio = energy / max_energy
                
                # Progressive penalty as energy gets low
                if energy_ratio < 0.3:
                    penalty = self.weights.energy_penalty * (0.3 - energy_ratio) ** 2
                    energy_penalties[agent_id] = penalty
                
                # Critical energy penalty
                if energy_ratio < 0.1:
                    energy_penalties[agent_id] += self.weights.energy_penalty * 2.0
        
        return energy_penalties
    
    def _compute_collision_penalties(self, agents: Dict, obstacles: List) -> Dict[int, float]:
        """Compute collision penalties"""
        
        collision_penalties = {agent_id: 0.0 for agent_id in agents.keys()}
        
        active_agents = {aid: agent for aid, agent in agents.items() if agent.get('active', True)}
        
        # Agent-agent collision penalties
        agent_positions = {}
        for agent_id, agent in active_agents.items():
            agent_positions[agent_id] = np.array([agent['x'], agent['y']])
        
        collision_threshold = 25.0  # Minimum safe distance
        
        for agent_id_1, pos_1 in agent_positions.items():
            for agent_id_2, pos_2 in agent_positions.items():
                if agent_id_1 >= agent_id_2:  # Avoid double counting
                    continue
                
                distance = np.linalg.norm(pos_1 - pos_2)
                if distance < collision_threshold:
                    penalty = self.weights.collision_penalty * (collision_threshold - distance) / collision_threshold
                    collision_penalties[agent_id_1] += penalty
                    collision_penalties[agent_id_2] += penalty
        
        # Agent-obstacle collision penalties
        for agent_id, agent in active_agents.items():
            agent_pos = np.array([agent['x'], agent['y']])
            
            for obstacle in obstacles:
                obstacle_pos = np.array([obstacle.x, obstacle.y])
                distance = np.linalg.norm(agent_pos - obstacle_pos)
                safety_margin = obstacle.radius + 10.0  # Safety buffer
                
                if distance < safety_margin:
                    penalty = self.weights.collision_penalty * (safety_margin - distance) / safety_margin
                    collision_penalties[agent_id] += penalty
        
        return collision_penalties
    
    def _find_covering_agents(self, poi, agents: Dict) -> List[int]:
        """Find agents that are covering a specific POI"""
        covering_agents = []
        poi_pos = np.array([poi.x, poi.y])
        
        for agent_id, agent in agents.items():
            if not agent.get('active', True):
                continue
                
            agent_pos = np.array([agent['x'], agent['y']])
            coverage_radius = agent.get('coverage_radius', 50.0)
            
            distance = np.linalg.norm(agent_pos - poi_pos)
            if distance <= coverage_radius:
                covering_agents.append(agent_id)
        
        return covering_agents
    
    def _update_performance_history(
        self, 
        coverage_metrics: Dict, 
        cooperation_score: float, 
        efficiency_score: float, 
        exploration_score: float
    ):
        """Update performance history for adaptive scaling"""
        
        self.performance_history['coverage_rates'].append(coverage_metrics.get('coverage_rate', 0.0))
        self.performance_history['cooperation_scores'].append(cooperation_score)
        self.performance_history['efficiency_scores'].append(efficiency_score)
        self.performance_history['exploration_scores'].append(exploration_score)
        
        # Keep only recent history
        max_history = 100
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
    
    def get_reward_analysis(self) -> Dict[str, Any]:
        """Get comprehensive reward analysis"""
        if not any(self.performance_history.values()):
            return {}
        
        analysis = {
            'current_scaling_factors': self.current_scaling.copy(),
            'training_progress': self.training_progress,
            'performance_trends': {},
            'reward_statistics': {}
        }
        
        # Calculate performance trends
        for metric_name, values in self.performance_history.items():
            if len(values) >= 10:
                recent_avg = np.mean(values[-10:])
                historical_avg = np.mean(values[:-10]) if len(values) > 10 else recent_avg
                
                analysis['performance_trends'][metric_name] = {
                    'recent_average': recent_avg,
                    'historical_average': historical_avg,
                    'improvement': recent_avg - historical_avg,
                    'trend': 'improving' if recent_avg > historical_avg else 'declining'
                }
        
        # Reward statistics
        analysis['reward_statistics'] = {
            'episode_count': self.episode_count,
            'exploration_coverage': np.sum(self.exploration_grid > 0) / (self.exploration_grid_size ** 2),
            'total_grid_visits': np.sum(self.exploration_grid)
        }
        
        return analysis
    
    def reset_episode(self):
        """Reset per-episode state"""
        # Keep exploration grid and performance history across episodes
        # Only reset if needed for specific episode-based metrics
        pass


def create_multi_objective_reward_system(config: Dict, env_config: Dict) -> MultiObjectiveRewardSystem:
    """Factory function to create multi-objective reward system"""
    
    num_agents = (env_config.get('num_satellites', 2) + 
                  env_config.get('num_uavs', 3) + 
                  env_config.get('num_ground_stations', 2))
    area_size = env_config.get('area_size', 1000)
    
    return MultiObjectiveRewardSystem(config, num_agents, area_size)