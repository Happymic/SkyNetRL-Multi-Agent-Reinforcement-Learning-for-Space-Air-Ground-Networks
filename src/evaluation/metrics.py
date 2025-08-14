"""
Comprehensive Evaluation Metrics System for SAGIN Coverage Optimization
Implements all metrics mentioned in the research paper
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class EpisodeData:
    """Data structure for episode information"""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    coverage_status: List[Dict]
    energy_levels: List[Dict]
    collisions: List[int]
    agent_positions: List[Dict]
    poi_priorities: List[np.ndarray]
    timestamps: List[float]


class ComprehensiveEvaluator:
    """Comprehensive evaluation system for SAGIN environments"""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator with environment configuration
        
        Args:
            config: Configuration dictionary containing environment parameters
        """
        self.config = config
        self.num_agents = config['num_agents']
        self.num_satellites = config.get('num_satellites', 1)
        self.num_uavs = config.get('num_uavs', 3)
        self.num_ground_stations = config.get('num_ground_stations', 2)
        self.area_size = config.get('area_size', 1000)
        self.uav_max_energy = config.get('uav_max_energy', 1200)
        
        # Metrics storage
        self.episode_metrics = []
        self.running_metrics = defaultdict(list)
        
        # Performance thresholds
        self.coverage_threshold = config.get('coverage_threshold', 0.8)  # 80% coverage target
        self.energy_efficiency_baseline = config.get('energy_efficiency_baseline', 0.5)
        
    def evaluate_episode(self, episode_data: EpisodeData, env_info: Dict) -> Dict:
        """
        Evaluate a complete episode using all metrics from the paper
        
        Args:
            episode_data: Episode data structure
            env_info: Environment information (POIs, obstacles, etc.)
            
        Returns:
            metrics: Dictionary of all computed metrics
        """
        metrics = {}
        
        # 1. Coverage Rate (Primary Metric)
        metrics['coverage_rate'] = self._compute_coverage_rate(episode_data, env_info)
        
        # 2. Energy Efficiency
        metrics['energy_efficiency'] = self._compute_energy_efficiency(episode_data)
        
        # 3. Task Completion Time
        metrics['completion_time'] = self._compute_completion_time(episode_data)
        
        # 4. Collision Rate
        metrics['collision_rate'] = self._compute_collision_rate(episode_data)
        
        # 5. Cooperation Index
        metrics['cooperation_index'] = self._compute_cooperation_index(episode_data)
        
        # 6. Task Priority Fulfillment
        metrics['priority_fulfillment'] = self._compute_priority_fulfillment(episode_data, env_info)
        
        # 7. Spatial Coverage Distribution
        metrics['spatial_distribution'] = self._compute_spatial_distribution(episode_data, env_info)
        
        # 8. Temporal Efficiency
        metrics['temporal_efficiency'] = self._compute_temporal_efficiency(episode_data)
        
        # 9. Energy Utilization per Agent Type
        metrics['agent_energy_utilization'] = self._compute_agent_energy_utilization(episode_data)
        
        # 10. Coverage Persistence
        metrics['coverage_persistence'] = self._compute_coverage_persistence(episode_data)
        
        # Store metrics
        self.episode_metrics.append(metrics)
        
        return metrics
    
    def _compute_coverage_rate(self, episode_data: EpisodeData, env_info: Dict) -> Dict:
        """Compute coverage rate metrics"""
        coverage_history = episode_data.coverage_status
        poi_priorities = env_info.get('poi_priorities', np.ones(env_info['num_pois']))
        
        # Overall coverage rate
        final_coverage_status = coverage_history[-1] if coverage_history else {}
        total_pois = env_info['num_pois']
        covered_pois = sum(1 for covered in final_coverage_status.values() if covered)
        overall_coverage_rate = covered_pois / total_pois if total_pois > 0 else 0.0
        
        # Weighted coverage rate (by priority)
        weighted_covered = sum(poi_priorities[poi_id] for poi_id, covered in final_coverage_status.items() if covered)
        total_priority_weight = sum(poi_priorities)
        weighted_coverage_rate = weighted_covered / total_priority_weight if total_priority_weight > 0 else 0.0
        
        # Coverage over time
        coverage_over_time = []
        for coverage_status in coverage_history:
            step_coverage = sum(1 for covered in coverage_status.values() if covered) / total_pois
            coverage_over_time.append(step_coverage)
        
        # Average coverage throughout episode
        avg_coverage_rate = np.mean(coverage_over_time) if coverage_over_time else 0.0
        
        return {
            'final_coverage_rate': overall_coverage_rate,
            'weighted_coverage_rate': weighted_coverage_rate,
            'average_coverage_rate': avg_coverage_rate,
            'coverage_trajectory': coverage_over_time,
            'coverage_improvement_rate': self._compute_coverage_improvement_rate(coverage_over_time)
        }
    
    def _compute_energy_efficiency(self, episode_data: EpisodeData) -> Dict:
        """Compute energy efficiency metrics"""
        energy_history = episode_data.energy_levels
        coverage_history = episode_data.coverage_status
        
        if not energy_history or not coverage_history:
            return {'energy_efficiency': 0.0}
        
        # Total energy consumed by UAVs
        total_energy_consumed = 0.0
        uav_count = 0
        
        initial_energy = {}
        final_energy = {}
        
        for agent_id, energy in energy_history[0].items():
            if self._is_uav(agent_id):
                initial_energy[agent_id] = energy
                uav_count += 1
        
        for agent_id, energy in energy_history[-1].items():
            if self._is_uav(agent_id):
                final_energy[agent_id] = energy
        
        for agent_id in initial_energy:
            if agent_id in final_energy:
                consumed = initial_energy[agent_id] - final_energy[agent_id]
                total_energy_consumed += max(0, consumed)
        
        # Coverage achieved
        final_coverage = len([c for c in coverage_history[-1].values() if c])
        
        # Energy efficiency = Coverage per unit energy
        energy_efficiency = final_coverage / (total_energy_consumed + 1e-9)
        
        # Normalized energy efficiency (0-1 scale)
        max_possible_coverage = len(coverage_history[-1]) if coverage_history[-1] else 1
        max_energy = uav_count * self.uav_max_energy
        theoretical_max_efficiency = max_possible_coverage / max_energy
        normalized_efficiency = energy_efficiency / (theoretical_max_efficiency + 1e-9)
        
        return {
            'energy_efficiency': energy_efficiency,
            'normalized_energy_efficiency': min(1.0, normalized_efficiency),
            'total_energy_consumed': total_energy_consumed,
            'energy_per_coverage': total_energy_consumed / (final_coverage + 1e-9),
            'uav_energy_utilization': self._compute_uav_energy_utilization(energy_history)
        }
    
    def _compute_completion_time(self, episode_data: EpisodeData) -> Dict:
        """Compute task completion time metrics"""
        coverage_history = episode_data.coverage_status
        
        if not coverage_history:
            return {'completion_time': float('inf')}
        
        # Find when coverage threshold is first reached
        total_pois = len(coverage_history[0]) if coverage_history[0] else 1
        target_coverage = self.coverage_threshold * total_pois
        
        completion_step = None
        for step, coverage_status in enumerate(coverage_history):
            covered_count = sum(1 for covered in coverage_status.values() if covered)
            if covered_count >= target_coverage:
                completion_step = step
                break
        
        completion_time = completion_step if completion_step is not None else len(coverage_history)
        
        # Time to different coverage levels
        coverage_milestones = {
            '25%': 0.25 * total_pois,
            '50%': 0.50 * total_pois,
            '75%': 0.75 * total_pois,
            '90%': 0.90 * total_pois
        }
        
        milestone_times = {}
        for milestone, target in coverage_milestones.items():
            milestone_time = None
            for step, coverage_status in enumerate(coverage_history):
                covered_count = sum(1 for covered in coverage_status.values() if covered)
                if covered_count >= target:
                    milestone_time = step
                    break
            milestone_times[f'time_to_{milestone}_coverage'] = milestone_time or len(coverage_history)
        
        return {
            'completion_time': completion_time,
            'normalized_completion_time': completion_time / len(coverage_history),
            **milestone_times
        }
    
    def _compute_collision_rate(self, episode_data: EpisodeData) -> Dict:
        """Compute collision-related metrics"""
        collisions = episode_data.collisions
        total_steps = len(collisions)
        
        if total_steps == 0:
            return {'collision_rate': 0.0}
        
        # Collision rate
        total_collisions = sum(collisions)
        collision_rate = total_collisions / (total_steps * self.num_agents)
        
        # Collision density over time
        collision_density = [c / self.num_agents for c in collisions]
        
        # Peak collision periods
        collision_variance = np.var(collision_density) if collision_density else 0.0
        
        return {
            'collision_rate': collision_rate,
            'total_collisions': total_collisions,
            'collision_density_variance': collision_variance,
            'max_simultaneous_collisions': max(collisions) if collisions else 0,
            'collision_free_periods': self._compute_collision_free_periods(collisions)
        }
    
    def _compute_cooperation_index(self, episode_data: EpisodeData) -> Dict:
        """Compute cooperation index metrics"""
        agent_positions = episode_data.agent_positions
        coverage_history = episode_data.coverage_status
        
        if not agent_positions or not coverage_history:
            return {'cooperation_index': 0.0}
        
        cooperation_scores = []
        
        for step in range(len(agent_positions)):
            positions = agent_positions[step]
            coverage = coverage_history[step]
            
            # Measure spatial coordination
            spatial_coordination = self._compute_spatial_coordination(positions)
            
            # Measure coverage coordination
            coverage_coordination = self._compute_coverage_coordination(positions, coverage)
            
            # Combined cooperation score
            step_cooperation = 0.6 * spatial_coordination + 0.4 * coverage_coordination
            cooperation_scores.append(step_cooperation)
        
        cooperation_index = np.mean(cooperation_scores) if cooperation_scores else 0.0
        
        return {
            'cooperation_index': cooperation_index,
            'spatial_coordination': np.mean([self._compute_spatial_coordination(pos) for pos in agent_positions]),
            'temporal_cooperation_variance': np.var(cooperation_scores) if cooperation_scores else 0.0,
            'peak_cooperation': max(cooperation_scores) if cooperation_scores else 0.0
        }
    
    def _compute_priority_fulfillment(self, episode_data: EpisodeData, env_info: Dict) -> Dict:
        """Compute task priority fulfillment metrics"""
        coverage_history = episode_data.coverage_status
        poi_priorities = env_info.get('poi_priorities', np.ones(env_info.get('num_pois', 1)))
        
        if not coverage_history:
            return {'priority_fulfillment': 0.0}
        
        final_coverage = coverage_history[-1]
        
        # High priority POI coverage (priority >= 4)
        high_priority_pois = [i for i, priority in enumerate(poi_priorities) if priority >= 4]
        high_priority_covered = sum(1 for poi_id in high_priority_pois if final_coverage.get(poi_id, False))
        high_priority_rate = high_priority_covered / len(high_priority_pois) if high_priority_pois else 1.0
        
        # Medium priority POI coverage (priority 2-3)
        medium_priority_pois = [i for i, priority in enumerate(poi_priorities) if 2 <= priority < 4]
        medium_priority_covered = sum(1 for poi_id in medium_priority_pois if final_coverage.get(poi_id, False))
        medium_priority_rate = medium_priority_covered / len(medium_priority_pois) if medium_priority_pois else 1.0
        
        # Priority-weighted score
        priority_weighted_score = sum(
            poi_priorities[poi_id] for poi_id, covered in final_coverage.items() if covered
        ) / sum(poi_priorities)
        
        return {
            'priority_fulfillment': priority_weighted_score,
            'high_priority_coverage_rate': high_priority_rate,
            'medium_priority_coverage_rate': medium_priority_rate,
            'priority_balance_score': self._compute_priority_balance_score(final_coverage, poi_priorities)
        }
    
    def _compute_spatial_distribution(self, episode_data: EpisodeData, env_info: Dict) -> Dict:
        """Compute spatial coverage distribution metrics"""
        agent_positions = episode_data.agent_positions
        coverage_history = episode_data.coverage_status
        
        if not agent_positions:
            return {'spatial_distribution': 0.0}
        
        # Compute coverage uniformity
        final_positions = agent_positions[-1]
        coverage_uniformity = self._compute_coverage_uniformity(final_positions, env_info)
        
        # Compute area utilization
        area_utilization = self._compute_area_utilization(agent_positions, env_info)
        
        return {
            'spatial_distribution': coverage_uniformity,
            'coverage_uniformity': coverage_uniformity,
            'area_utilization': area_utilization,
            'coverage_density_variance': self._compute_coverage_density_variance(agent_positions, env_info)
        }
    
    def _compute_temporal_efficiency(self, episode_data: EpisodeData) -> Dict:
        """Compute temporal efficiency metrics"""
        coverage_history = episode_data.coverage_status
        timestamps = episode_data.timestamps
        
        if len(coverage_history) < 2:
            return {'temporal_efficiency': 0.0}
        
        # Coverage gain per unit time
        initial_coverage = sum(1 for covered in coverage_history[0].values() if covered)
        final_coverage = sum(1 for covered in coverage_history[-1].values() if covered)
        coverage_gain = final_coverage - initial_coverage
        
        episode_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else len(coverage_history)
        temporal_efficiency = coverage_gain / (episode_duration + 1e-9)
        
        # Coverage velocity over time
        coverage_velocities = []
        for i in range(1, len(coverage_history)):
            prev_coverage = sum(1 for covered in coverage_history[i-1].values() if covered)
            curr_coverage = sum(1 for covered in coverage_history[i].values() if covered)
            velocity = curr_coverage - prev_coverage
            coverage_velocities.append(velocity)
        
        return {
            'temporal_efficiency': temporal_efficiency,
            'average_coverage_velocity': np.mean(coverage_velocities) if coverage_velocities else 0.0,
            'coverage_acceleration': np.std(coverage_velocities) if len(coverage_velocities) > 1 else 0.0
        }
    
    def _compute_agent_energy_utilization(self, episode_data: EpisodeData) -> Dict:
        """Compute energy utilization per agent type"""
        energy_history = episode_data.energy_levels
        
        if not energy_history:
            return {}
        
        initial_energy = energy_history[0]
        final_energy = energy_history[-1]
        
        utilization_by_type = {
            'satellite': [],
            'uav': [],
            'ground_station': []
        }
        
        for agent_id in initial_energy:
            agent_type = self._get_agent_type(agent_id)
            if agent_type == 'uav':  # Only UAVs have energy constraints
                initial = initial_energy[agent_id]
                final = final_energy.get(agent_id, initial)
                utilization = (initial - final) / initial if initial > 0 else 0.0
                utilization_by_type['uav'].append(utilization)
        
        # Compute statistics for each type
        result = {}
        for agent_type, utilizations in utilization_by_type.items():
            if utilizations:
                result[f'{agent_type}_avg_utilization'] = np.mean(utilizations)
                result[f'{agent_type}_utilization_std'] = np.std(utilizations)
                result[f'{agent_type}_max_utilization'] = np.max(utilizations)
            else:
                result[f'{agent_type}_avg_utilization'] = 0.0
                result[f'{agent_type}_utilization_std'] = 0.0
                result[f'{agent_type}_max_utilization'] = 0.0
        
        return result
    
    def _compute_coverage_persistence(self, episode_data: EpisodeData) -> Dict:
        """Compute coverage persistence metrics"""
        coverage_history = episode_data.coverage_status
        
        if len(coverage_history) < 2:
            return {'coverage_persistence': 1.0}
        
        # Track how long each POI remains covered
        poi_coverage_durations = defaultdict(list)
        poi_current_duration = defaultdict(int)
        
        for step, coverage_status in enumerate(coverage_history):
            for poi_id, is_covered in coverage_status.items():
                if is_covered:
                    poi_current_duration[poi_id] += 1
                else:
                    if poi_current_duration[poi_id] > 0:
                        poi_coverage_durations[poi_id].append(poi_current_duration[poi_id])
                        poi_current_duration[poi_id] = 0
        
        # Add final durations
        for poi_id, duration in poi_current_duration.items():
            if duration > 0:
                poi_coverage_durations[poi_id].append(duration)
        
        # Compute persistence metrics
        all_durations = [d for durations in poi_coverage_durations.values() for d in durations]
        avg_persistence = np.mean(all_durations) if all_durations else 0.0
        
        # Coverage stability (how often coverage changes)
        coverage_changes = 0
        for step in range(1, len(coverage_history)):
            prev_coverage = coverage_history[step-1]
            curr_coverage = coverage_history[step]
            for poi_id in prev_coverage:
                if prev_coverage.get(poi_id, False) != curr_coverage.get(poi_id, False):
                    coverage_changes += 1
        
        coverage_stability = 1.0 - (coverage_changes / (len(coverage_history) * len(coverage_history[0])))
        
        return {
            'coverage_persistence': avg_persistence / len(coverage_history),
            'coverage_stability': coverage_stability,
            'average_coverage_duration': avg_persistence
        }
    
    # Helper methods
    def _is_uav(self, agent_id: int) -> bool:
        """Check if agent is a UAV"""
        return self.num_satellites <= agent_id < (self.num_satellites + self.num_uavs)
    
    def _get_agent_type(self, agent_id: int) -> str:
        """Get agent type from ID"""
        if agent_id < self.num_satellites:
            return 'satellite'
        elif agent_id < self.num_satellites + self.num_uavs:
            return 'uav'
        else:
            return 'ground_station'
    
    def _compute_coverage_improvement_rate(self, coverage_trajectory: List[float]) -> float:
        """Compute rate of coverage improvement"""
        if len(coverage_trajectory) < 2:
            return 0.0
        
        improvements = [max(0, coverage_trajectory[i] - coverage_trajectory[i-1])
                       for i in range(1, len(coverage_trajectory))]
        return np.mean(improvements)
    
    def _compute_uav_energy_utilization(self, energy_history: List[Dict]) -> Dict:
        """Compute UAV-specific energy utilization metrics"""
        if not energy_history:
            return {}
        
        uav_energy_profiles = defaultdict(list)
        
        for energy_status in energy_history:
            for agent_id, energy in energy_status.items():
                if self._is_uav(agent_id):
                    uav_energy_profiles[agent_id].append(energy)
        
        # Compute energy efficiency metrics
        energy_metrics = {}
        for agent_id, energy_profile in uav_energy_profiles.items():
            if len(energy_profile) > 1:
                initial_energy = energy_profile[0]
                final_energy = energy_profile[-1]
                energy_consumption_rate = (initial_energy - final_energy) / len(energy_profile)
                energy_metrics[f'uav_{agent_id}_consumption_rate'] = energy_consumption_rate
                
                # Energy usage smoothness (lower is better)
                energy_changes = [abs(energy_profile[i] - energy_profile[i-1])
                                for i in range(1, len(energy_profile))]
                energy_metrics[f'uav_{agent_id}_usage_smoothness'] = np.std(energy_changes)
        
        return energy_metrics
    
    def _compute_collision_free_periods(self, collisions: List[int]) -> int:
        """Compute number of collision-free periods"""
        if not collisions:
            return 0
        
        collision_free_periods = 0
        in_collision_free_period = collisions[0] == 0
        
        for collision_count in collisions[1:]:
            if collision_count == 0 and not in_collision_free_period:
                collision_free_periods += 1
                in_collision_free_period = True
            elif collision_count > 0:
                in_collision_free_period = False
        
        return collision_free_periods
    
    def _compute_spatial_coordination(self, positions: Dict) -> float:
        """Compute spatial coordination score"""
        if len(positions) < 2:
            return 1.0
        
        # Compute pairwise distances
        agent_ids = list(positions.keys())
        distances = []
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                pos1 = positions[agent_ids[i]]
                pos2 = positions[agent_ids[j]]
                if isinstance(pos1, (list, tuple)) and isinstance(pos2, (list, tuple)):
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Optimal distance should be around coverage radius
        optimal_distance = 200  # Average coverage radius
        distance_deviations = [abs(d - optimal_distance) / optimal_distance for d in distances]
        
        # Coordination score (lower deviation = higher coordination)
        coordination_score = 1.0 / (1.0 + np.mean(distance_deviations))
        
        return coordination_score
    
    def _compute_coverage_coordination(self, positions: Dict, coverage_status: Dict) -> float:
        """Compute coverage-based coordination score"""
        # This is a simplified metric - in practice, you'd want more sophisticated coverage overlap analysis
        covered_pois = sum(1 for covered in coverage_status.values() if covered)
        total_pois = len(coverage_status)
        
        return covered_pois / total_pois if total_pois > 0 else 0.0
    
    def _compute_priority_balance_score(self, coverage_status: Dict, poi_priorities: np.ndarray) -> float:
        """Compute how well coverage balances different priority levels"""
        if len(poi_priorities) == 0:
            return 1.0
        
        # Group POIs by priority level
        priority_groups = defaultdict(list)
        for poi_id, priority in enumerate(poi_priorities):
            priority_groups[int(priority)].append(poi_id)
        
        # Compute coverage rate for each priority level
        priority_coverage_rates = {}
        for priority, poi_ids in priority_groups.items():
            covered = sum(1 for poi_id in poi_ids if coverage_status.get(poi_id, False))
            priority_coverage_rates[priority] = covered / len(poi_ids)
        
        # Balance score: higher priority levels should have higher coverage rates
        balance_score = 0.0
        total_weight = 0.0
        
        for priority, coverage_rate in priority_coverage_rates.items():
            weight = priority  # Higher priority gets higher weight
            balance_score += weight * coverage_rate
            total_weight += weight
        
        return balance_score / total_weight if total_weight > 0 else 0.0
    
    def _compute_coverage_uniformity(self, positions: Dict, env_info: Dict) -> float:
        """Compute spatial uniformity of coverage"""
        # Simplified uniformity metric
        area_size = env_info.get('area_size', self.area_size)
        
        if not positions:
            return 0.0
        
        # Divide area into grid and check coverage distribution
        grid_size = 10
        cell_size = area_size // grid_size
        coverage_grid = np.zeros((grid_size, grid_size))
        
        for agent_id, pos in positions.items():
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
                grid_x = int(min(x // cell_size, grid_size - 1))
                grid_y = int(min(y // cell_size, grid_size - 1))
                coverage_grid[grid_x, grid_y] = 1
        
        # Uniformity = 1 - coefficient of variation
        coverage_variance = np.var(coverage_grid.flatten())
        coverage_mean = np.mean(coverage_grid.flatten())
        
        if coverage_mean == 0:
            return 0.0
        
        coefficient_of_variation = np.sqrt(coverage_variance) / coverage_mean
        uniformity = 1.0 / (1.0 + coefficient_of_variation)
        
        return uniformity
    
    def _compute_area_utilization(self, agent_positions: List[Dict], env_info: Dict) -> float:
        """Compute how well agents utilize the available area"""
        if not agent_positions:
            return 0.0
        
        area_size = env_info.get('area_size', self.area_size)
        
        # Track all positions visited throughout episode
        visited_cells = set()
        cell_size = area_size // 20  # 20x20 grid
        
        for positions in agent_positions:
            for agent_id, pos in positions.items():
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    x, y = pos[0], pos[1]
                    cell_x = int(min(x // cell_size, 19))
                    cell_y = int(min(y // cell_size, 19))
                    visited_cells.add((cell_x, cell_y))
        
        # Area utilization = visited cells / total cells
        total_cells = 20 * 20
        area_utilization = len(visited_cells) / total_cells
        
        return area_utilization
    
    def _compute_coverage_density_variance(self, agent_positions: List[Dict], env_info: Dict) -> float:
        """Compute variance in coverage density across the area"""
        if not agent_positions:
            return 0.0
        
        area_size = env_info.get('area_size', self.area_size)
        
        # Create density map
        grid_size = 15
        cell_size = area_size // grid_size
        density_grid = np.zeros((grid_size, grid_size))
        
        # Accumulate agent presence over time
        for positions in agent_positions:
            for agent_id, pos in positions.items():
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    x, y = pos[0], pos[1]
                    grid_x = int(min(x // cell_size, grid_size - 1))
                    grid_y = int(min(y // cell_size, grid_size - 1))
                    density_grid[grid_x, grid_y] += 1
        
        # Normalize by episode length
        density_grid = density_grid / len(agent_positions)
        
        # Return normalized variance
        density_variance = np.var(density_grid.flatten())
        max_possible_variance = np.var([0, self.num_agents])  # Maximum possible variance
        
        return density_variance / max_possible_variance if max_possible_variance > 0 else 0.0
    
    def get_aggregate_metrics(self, num_episodes: int = None) -> Dict:
        """Get aggregate metrics across multiple episodes"""
        if not self.episode_metrics:
            return {}
        
        episodes_to_analyze = self.episode_metrics[-num_episodes:] if num_episodes else self.episode_metrics
        
        aggregate = {}
        
        # For each metric, compute statistics
        all_metrics = defaultdict(list)
        
        for episode_metrics in episodes_to_analyze:
            for metric_category, metric_values in episode_metrics.items():
                if isinstance(metric_values, dict):
                    for sub_metric, value in metric_values.items():
                        if isinstance(value, (int, float)):
                            all_metrics[f"{metric_category}_{sub_metric}"].append(value)
                elif isinstance(metric_values, (int, float)):
                    all_metrics[metric_category].append(metric_values)
        
        # Compute statistics for each metric
        for metric_name, values in all_metrics.items():
            if values:
                aggregate[f"{metric_name}_mean"] = np.mean(values)
                aggregate[f"{metric_name}_std"] = np.std(values)
                aggregate[f"{metric_name}_min"] = np.min(values)
                aggregate[f"{metric_name}_max"] = np.max(values)
        
        return aggregate
    
    def reset(self):
        """Reset all stored metrics"""
        self.episode_metrics = []
        self.running_metrics = defaultdict(list)