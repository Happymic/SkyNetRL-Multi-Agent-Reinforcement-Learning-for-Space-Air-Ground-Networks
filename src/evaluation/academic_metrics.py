"""
Academic Performance Metrics for SAGIN Networks
Implements standardized metrics for research evaluation and comparison
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NetworkPerformanceMetrics:
    """Container for network performance metrics"""
    # Coverage metrics
    coverage_probability: float = 0.0
    coverage_efficiency: float = 0.0
    handover_success_rate: float = 0.0
    
    # Throughput metrics
    aggregate_throughput: float = 0.0
    per_user_throughput: float = 0.0
    spectral_efficiency: float = 0.0
    
    # Delay metrics
    average_delay: float = 0.0
    delay_jitter: float = 0.0
    delay_violation_rate: float = 0.0
    
    # Reliability metrics
    packet_delivery_ratio: float = 0.0
    reliability_availability: float = 0.0
    error_rate: float = 0.0
    
    # Energy metrics
    energy_efficiency: float = 0.0
    power_consumption: float = 0.0
    battery_lifetime: float = 0.0
    
    # Fairness metrics
    jain_fairness_index: float = 0.0
    proportional_fairness: float = 0.0
    max_min_fairness: float = 0.0
    
    # Network efficiency
    resource_utilization: float = 0.0
    blocking_probability: float = 0.0
    network_overhead: float = 0.0


@dataclass
class ConvergenceMetrics:
    """Convergence analysis metrics"""
    convergence_episode: int = -1
    convergence_time: float = 0.0
    convergence_rate: float = 0.0
    stability_measure: float = 0.0
    oscillation_frequency: float = 0.0


@dataclass
class ScalabilityMetrics:
    """Scalability analysis metrics"""
    computational_complexity: float = 0.0
    memory_usage: float = 0.0
    communication_overhead: float = 0.0
    scalability_factor: float = 0.0


class StandardizedMetrics:
    """
    Implements standardized metrics for academic evaluation of SAGIN networks
    Based on ITU-R recommendations and 3GPP specifications
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.epsilon = 1e-12  # Small value to avoid division by zero
        
        # Standard thresholds (can be configured)
        self.coverage_threshold_db = -100  # dBm
        self.delay_threshold_ms = 100  # ms
        self.reliability_threshold = 0.99
        self.spectral_efficiency_reference = 1.0  # bps/Hz
        
    def compute_all_metrics(self, data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute all standardized metrics from simulation data
        
        Args:
            data: Dictionary containing simulation data arrays
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Coverage metrics
        if 'coverage_events' in data and data['coverage_events']:
            metrics.update(self._compute_coverage_metrics(data['coverage_events']))
        
        # Throughput metrics
        if 'throughput_values' in data and data['throughput_values']:
            metrics.update(self._compute_throughput_metrics(data['throughput_values']))
        
        # Delay metrics
        if 'latency_values' in data and data['latency_values']:
            metrics.update(self._compute_delay_metrics(data['latency_values']))
        
        # Reliability metrics
        if 'packet_loss_rates' in data and data['packet_loss_rates']:
            metrics.update(self._compute_reliability_metrics(data['packet_loss_rates']))
        
        # Energy metrics
        if 'energy_consumption' in data and data['energy_consumption']:
            metrics.update(self._compute_energy_metrics(data['energy_consumption']))
        
        # Fairness metrics
        if 'throughput_values' in data and data['throughput_values']:
            metrics.update(self._compute_fairness_metrics(data['throughput_values']))
        
        # Spectral efficiency
        if 'spectral_efficiency' in data and data['spectral_efficiency']:
            metrics.update(self._compute_spectral_efficiency_metrics(data['spectral_efficiency']))
        
        # Network efficiency
        metrics.update(self._compute_network_efficiency_metrics(data))
        
        return metrics
    
    def _compute_coverage_metrics(self, coverage_events: List[bool]) -> Dict[str, float]:
        """Compute coverage-related metrics"""
        coverage_array = np.array(coverage_events, dtype=float)
        
        # Coverage probability (ITU-R standard)
        coverage_probability = np.mean(coverage_array)
        
        # Coverage efficiency (weighted by quality)
        coverage_efficiency = coverage_probability  # Simplified
        
        # Handover success rate (from coverage transitions)
        handover_events = []
        for i in range(1, len(coverage_array)):
            if coverage_array[i-1] != coverage_array[i]:
                handover_events.append(coverage_array[i])
        
        handover_success_rate = np.mean(handover_events) if handover_events else 1.0
        
        return {
            'coverage_probability': coverage_probability,
            'coverage_efficiency': coverage_efficiency,
            'handover_success_rate': handover_success_rate
        }
    
    def _compute_throughput_metrics(self, throughput_values: List[float]) -> Dict[str, float]:
        """Compute throughput-related metrics"""
        throughput_array = np.array(throughput_values)
        
        # Aggregate throughput
        aggregate_throughput = np.sum(throughput_array)
        
        # Per-user average throughput
        per_user_throughput = np.mean(throughput_array)
        
        # Throughput standard deviation (stability)
        throughput_std = np.std(throughput_array)
        
        # Throughput efficiency (compared to theoretical maximum)
        # Assuming theoretical max is 100 Mbps per user
        theoretical_max = 100e6  # bps
        throughput_efficiency = per_user_throughput / theoretical_max
        
        return {
            'aggregate_throughput_bps': aggregate_throughput,
            'per_user_throughput_bps': per_user_throughput,
            'throughput_std_bps': throughput_std,
            'throughput_efficiency': min(throughput_efficiency, 1.0)
        }
    
    def _compute_delay_metrics(self, latency_values: List[float]) -> Dict[str, float]:
        """Compute delay-related metrics"""
        delay_array = np.array(latency_values) * 1000  # Convert to ms
        
        # Average delay
        average_delay = np.mean(delay_array)
        
        # Delay jitter (standard deviation)
        delay_jitter = np.std(delay_array)
        
        # 95th percentile delay
        delay_95th_percentile = np.percentile(delay_array, 95)
        
        # Delay violation rate (delays > threshold)
        delay_violations = np.sum(delay_array > self.delay_threshold_ms)
        delay_violation_rate = delay_violations / len(delay_array)
        
        # Delay efficiency
        delay_efficiency = 1.0 - min(average_delay / self.delay_threshold_ms, 1.0)
        
        return {
            'average_delay_ms': average_delay,
            'delay_jitter_ms': delay_jitter,
            'delay_95th_percentile_ms': delay_95th_percentile,
            'delay_violation_rate': delay_violation_rate,
            'delay_efficiency': delay_efficiency
        }
    
    def _compute_reliability_metrics(self, packet_loss_rates: List[float]) -> Dict[str, float]:
        """Compute reliability-related metrics"""
        loss_array = np.array(packet_loss_rates)
        
        # Packet delivery ratio
        packet_delivery_ratio = 1.0 - np.mean(loss_array)
        
        # Reliability (99.9% target)
        reliability_availability = packet_delivery_ratio
        
        # Error rate
        error_rate = np.mean(loss_array)
        
        # Reliability efficiency
        reliability_efficiency = min(reliability_availability / self.reliability_threshold, 1.0)
        
        # MTBF (Mean Time Between Failures) - simplified
        failure_events = np.sum(loss_array > 0.01)  # 1% loss threshold
        mtbf = len(loss_array) / (failure_events + self.epsilon)
        
        return {
            'packet_delivery_ratio': packet_delivery_ratio,
            'reliability_availability': reliability_availability,
            'error_rate': error_rate,
            'reliability_efficiency': reliability_efficiency,
            'mean_time_between_failures': mtbf
        }
    
    def _compute_energy_metrics(self, energy_consumption: List[float]) -> Dict[str, float]:
        """Compute energy-related metrics"""
        energy_array = np.array(energy_consumption)
        
        # Average power consumption
        power_consumption = np.mean(energy_array)
        
        # Energy efficiency (bits per joule)
        # Simplified: assuming 1 Mbps per unit energy
        energy_efficiency = 1e6 / (power_consumption + self.epsilon)
        
        # Battery lifetime estimation (hours)
        # Assuming 1000 Wh battery capacity
        battery_capacity_wh = 1000
        battery_lifetime = battery_capacity_wh / (power_consumption + self.epsilon)
        
        # Energy consumption variability
        energy_std = np.std(energy_array)
        energy_coefficient_variation = energy_std / (power_consumption + self.epsilon)
        
        return {
            'power_consumption_w': power_consumption,
            'energy_efficiency_bits_per_joule': energy_efficiency,
            'battery_lifetime_hours': battery_lifetime,
            'energy_coefficient_variation': energy_coefficient_variation
        }
    
    def _compute_fairness_metrics(self, throughput_values: List[float]) -> Dict[str, float]:
        """Compute fairness-related metrics"""
        throughput_array = np.array(throughput_values)
        n = len(throughput_array)
        
        if n <= 1:
            return {
                'jain_fairness_index': 1.0,
                'proportional_fairness': 1.0,
                'max_min_fairness': 1.0
            }
        
        # Jain's Fairness Index
        sum_throughput = np.sum(throughput_array)
        sum_squared_throughput = np.sum(throughput_array ** 2)
        jain_fairness = (sum_throughput ** 2) / (n * sum_squared_throughput + self.epsilon)
        
        # Proportional Fairness (log utility)
        log_throughput = np.log(throughput_array + self.epsilon)
        proportional_fairness = np.sum(log_throughput)
        
        # Max-Min Fairness (based on minimum throughput)
        min_throughput = np.min(throughput_array)
        max_throughput = np.max(throughput_array)
        max_min_fairness = min_throughput / (max_throughput + self.epsilon)
        
        # Gini coefficient (inequality measure)
        sorted_throughput = np.sort(throughput_array)
        cumsum = np.cumsum(sorted_throughput)
        gini_coefficient = (2 * np.sum((np.arange(1, n + 1) * sorted_throughput))) / (n * sum_throughput + self.epsilon) - (n + 1) / n
        
        return {
            'jain_fairness_index': jain_fairness,
            'proportional_fairness': proportional_fairness,
            'max_min_fairness': max_min_fairness,
            'gini_coefficient': gini_coefficient
        }
    
    def _compute_spectral_efficiency_metrics(self, spectral_efficiency: List[float]) -> Dict[str, float]:
        """Compute spectral efficiency metrics"""
        se_array = np.array(spectral_efficiency)
        
        # Average spectral efficiency
        average_se = np.mean(se_array)
        
        # Peak spectral efficiency
        peak_se = np.max(se_array)
        
        # Spectral efficiency percentiles
        se_5th = np.percentile(se_array, 5)
        se_95th = np.percentile(se_array, 95)
        
        # Spectral efficiency normalized to reference
        se_efficiency = average_se / self.spectral_efficiency_reference
        
        return {
            'average_spectral_efficiency_bps_hz': average_se,
            'peak_spectral_efficiency_bps_hz': peak_se,
            'spectral_efficiency_5th_percentile': se_5th,
            'spectral_efficiency_95th_percentile': se_95th,
            'spectral_efficiency_normalized': se_efficiency
        }
    
    def _compute_network_efficiency_metrics(self, data: Dict[str, List[float]]) -> Dict[str, float]:
        """Compute network efficiency metrics"""
        metrics = {}
        
        # Resource utilization
        if 'resource_utilization' in data:
            metrics['resource_utilization'] = np.mean(data['resource_utilization'])
        
        # Blocking probability
        if 'blocking_events' in data:
            blocking_array = np.array(data['blocking_events'])
            metrics['blocking_probability'] = np.mean(blocking_array)
        
        # Network overhead (simplified)
        if 'overhead_bytes' in data and 'data_bytes' in data:
            overhead = np.sum(data['overhead_bytes'])
            data_volume = np.sum(data['data_bytes'])
            metrics['network_overhead'] = overhead / (data_volume + self.epsilon)
        
        # Load balancing efficiency
        if 'load_distribution' in data:
            load_array = np.array(data['load_distribution'])
            load_std = np.std(load_array)
            load_mean = np.mean(load_array)
            metrics['load_balancing_efficiency'] = 1.0 - (load_std / (load_mean + self.epsilon))
        
        return metrics
    
    def compute_convergence_metrics(self, reward_history: List[float], 
                                  window_size: int = 100) -> ConvergenceMetrics:
        """
        Compute convergence analysis metrics
        
        Args:
            reward_history: Episode reward history
            window_size: Window size for convergence detection
            
        Returns:
            ConvergenceMetrics object
        """
        reward_array = np.array(reward_history)
        
        # Find convergence episode
        convergence_episode = self._detect_convergence_episode(reward_array, window_size)
        
        # Convergence rate (slope of improvement)
        if convergence_episode > 0:
            convergence_rewards = reward_array[:convergence_episode]
            if len(convergence_rewards) > 1:
                x = np.arange(len(convergence_rewards))
                slope, _, _, _, _ = stats.linregress(x, convergence_rewards)
                convergence_rate = slope
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        # Stability measure (variance after convergence)
        if convergence_episode > 0 and convergence_episode < len(reward_array):
            post_convergence = reward_array[convergence_episode:]
            stability_measure = 1.0 / (1.0 + np.var(post_convergence))
        else:
            stability_measure = 0.0
        
        # Oscillation frequency
        oscillation_frequency = self._compute_oscillation_frequency(reward_array)
        
        return ConvergenceMetrics(
            convergence_episode=convergence_episode,
            convergence_time=convergence_episode,  # Simplified
            convergence_rate=convergence_rate,
            stability_measure=stability_measure,
            oscillation_frequency=oscillation_frequency
        )
    
    def compute_scalability_metrics(self, num_agents_list: List[int],
                                   computation_times: List[float],
                                   memory_usage: List[float],
                                   communication_overhead: List[float]) -> ScalabilityMetrics:
        """
        Compute scalability analysis metrics
        
        Args:
            num_agents_list: List of agent counts
            computation_times: Corresponding computation times
            memory_usage: Corresponding memory usage
            communication_overhead: Corresponding communication overhead
            
        Returns:
            ScalabilityMetrics object
        """
        agents_array = np.array(num_agents_list)
        comp_array = np.array(computation_times)
        mem_array = np.array(memory_usage)
        comm_array = np.array(communication_overhead)
        
        # Computational complexity analysis
        if len(agents_array) > 1:
            # Fit power law: computation_time = a * num_agents^b
            log_agents = np.log(agents_array + self.epsilon)
            log_comp = np.log(comp_array + self.epsilon)
            slope, _, _, _, _ = stats.linregress(log_agents, log_comp)
            computational_complexity = slope
        else:
            computational_complexity = 1.0
        
        # Memory scaling
        if len(agents_array) > 1:
            log_mem = np.log(mem_array + self.epsilon)
            slope, _, _, _, _ = stats.linregress(log_agents, log_mem)
            memory_scaling = slope
        else:
            memory_scaling = 1.0
        
        # Communication overhead scaling
        if len(agents_array) > 1:
            log_comm = np.log(comm_array + self.epsilon)
            slope, _, _, _, _ = stats.linregress(log_agents, log_comm)
            communication_scaling = slope
        else:
            communication_scaling = 1.0
        
        # Overall scalability factor (lower is better)
        scalability_factor = (computational_complexity + memory_scaling + communication_scaling) / 3.0
        
        return ScalabilityMetrics(
            computational_complexity=computational_complexity,
            memory_usage=np.mean(mem_array),
            communication_overhead=np.mean(comm_array),
            scalability_factor=scalability_factor
        )
    
    def _detect_convergence_episode(self, reward_array: np.ndarray, 
                                   window_size: int = 100) -> int:
        """Detect convergence episode using moving average stability"""
        if len(reward_array) < window_size * 2:
            return -1
        
        # Compute moving averages
        moving_avg = np.convolve(reward_array, np.ones(window_size)/window_size, mode='valid')
        
        # Find where variance becomes stable
        for i in range(window_size, len(moving_avg)):
            recent_window = moving_avg[i-window_size:i]
            variance = np.var(recent_window)
            
            # Check if variance is below threshold
            if variance < 0.01 * np.var(moving_avg):
                return i
        
        return -1
    
    def _compute_oscillation_frequency(self, reward_array: np.ndarray) -> float:
        """Compute oscillation frequency in reward signal"""
        if len(reward_array) < 10:
            return 0.0
        
        # Simple peak counting
        peaks = 0
        for i in range(1, len(reward_array) - 1):
            if (reward_array[i] > reward_array[i-1] and 
                reward_array[i] > reward_array[i+1]):
                peaks += 1
        
        return peaks / len(reward_array)
    
    def generate_performance_report(self, metrics: Dict[str, float]) -> str:
        """Generate formatted performance report"""
        report = "# Network Performance Report\n\n"
        
        # Coverage metrics
        report += "## Coverage Metrics\n"
        coverage_metrics = {k: v for k, v in metrics.items() if 'coverage' in k or 'handover' in k}
        for key, value in coverage_metrics.items():
            report += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "\n"
        
        # Throughput metrics
        report += "## Throughput Metrics\n"
        throughput_metrics = {k: v for k, v in metrics.items() if 'throughput' in k or 'spectral' in k}
        for key, value in throughput_metrics.items():
            if 'bps' in key:
                report += f"- {key.replace('_', ' ').title()}: {value/1e6:.2f} Mbps\n"
            else:
                report += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "\n"
        
        # Delay metrics
        report += "## Delay Metrics\n"
        delay_metrics = {k: v for k, v in metrics.items() if 'delay' in k or 'latency' in k}
        for key, value in delay_metrics.items():
            if 'ms' in key:
                report += f"- {key.replace('_', ' ').title()}: {value:.2f} ms\n"
            else:
                report += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "\n"
        
        # Reliability metrics
        report += "## Reliability Metrics\n"
        reliability_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['reliability', 'error', 'packet', 'delivery'])}
        for key, value in reliability_metrics.items():
            report += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "\n"
        
        # Energy metrics
        report += "## Energy Metrics\n"
        energy_metrics = {k: v for k, v in metrics.items() if 'energy' in k or 'power' in k or 'battery' in k}
        for key, value in energy_metrics.items():
            if 'hours' in key:
                report += f"- {key.replace('_', ' ').title()}: {value:.1f} hours\n"
            elif 'w' in key:
                report += f"- {key.replace('_', ' ').title()}: {value:.2f} W\n"
            else:
                report += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "\n"
        
        # Fairness metrics
        report += "## Fairness Metrics\n"
        fairness_metrics = {k: v for k, v in metrics.items() if 'fairness' in k or 'gini' in k}
        for key, value in fairness_metrics.items():
            report += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
        
        return report