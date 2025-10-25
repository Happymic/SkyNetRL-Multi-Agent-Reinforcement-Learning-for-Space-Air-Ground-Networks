"""
MAC Layer Protocol Simulation for SAGIN Networks
Implements realistic resource allocation and medium access control
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict


class ResourceType(Enum):
    """Types of radio resources"""
    TIME_SLOT = "time_slot"
    FREQUENCY_BLOCK = "frequency_block"
    SPATIAL_STREAM = "spatial_stream"
    CODE_SEQUENCE = "code_sequence"


class PriorityClass(Enum):
    """Traffic priority classes"""
    EMERGENCY = 0
    REAL_TIME = 1
    HIGH_PRIORITY = 2
    BEST_EFFORT = 3


@dataclass
class ResourceAllocation:
    """Resource allocation result"""
    user_id: int
    resource_blocks: List[int]
    power_allocation: float
    modulation_coding_scheme: int
    transmission_duration: float
    achievable_rate: float


@dataclass
class UserRequest:
    """User transmission request"""
    user_id: int
    data_size_bits: int
    priority_class: PriorityClass
    delay_tolerance_ms: float
    arrival_time: float
    qos_requirements: Dict[str, float]


@dataclass
class ChannelState:
    """Channel state information"""
    user_id: int
    channel_quality_indicator: int
    signal_to_noise_ratio: float
    path_loss: float
    interference_power: float
    doppler_shift: float


class MACLayer:
    """
    MAC Layer implementation for SAGIN networks
    Supports multiple access schemes and resource allocation algorithms
    """
    
    def __init__(self, config: Dict):
        """Initialize MAC layer with configuration"""
        self.config = config
        
        # System parameters
        self.num_resource_blocks = config.get('num_resource_blocks', 50)
        self.resource_block_bandwidth = config.get('resource_block_bandwidth', 180e3)  # Hz
        self.symbol_duration = config.get('symbol_duration', 66.7e-6)  # seconds
        self.frame_duration = config.get('frame_duration', 1e-3)  # 1ms frame
        self.max_users_per_frame = config.get('max_users_per_frame', 100)
        
        # Power control parameters
        self.max_transmit_power = config.get('max_transmit_power', 23)  # dBm
        self.min_transmit_power = config.get('min_transmit_power', -40)  # dBm
        self.power_control_step = config.get('power_control_step', 1)  # dB
        
        # QoS parameters
        self.priority_weights = {
            PriorityClass.EMERGENCY: 1.0,
            PriorityClass.REAL_TIME: 0.8,
            PriorityClass.HIGH_PRIORITY: 0.6,
            PriorityClass.BEST_EFFORT: 0.4
        }
        
        # Scheduling algorithm
        self.scheduling_algorithm = config.get('scheduling_algorithm', 'proportional_fair')
        
        # Resource allocation state
        self.current_frame = 0
        self.user_histories = defaultdict(list)  # For fair scheduling
        self.resource_usage_history = []
        
        # Statistics
        self.allocation_stats = {
            'total_allocated_blocks': 0,
            'successful_allocations': 0,
            'blocked_requests': 0,
            'average_delay': 0.0,
            'throughput_per_user': defaultdict(float)
        }
    
    def allocate_uplink_resources(self, user_requests: List[UserRequest], 
                                  channel_states: List[ChannelState]) -> List[ResourceAllocation]:
        """
        Allocate uplink resources for user requests
        
        Args:
            user_requests: List of user transmission requests
            channel_states: Current channel state information
            
        Returns:
            List of resource allocations
        """
        # Sort requests by priority and delay tolerance
        sorted_requests = self._prioritize_requests(user_requests)
        
        # Create channel state mapping
        channel_map = {cs.user_id: cs for cs in channel_states}
        
        # Available resources
        available_blocks = list(range(self.num_resource_blocks))
        allocations = []
        
        for request in sorted_requests:
            if request.user_id not in channel_map:
                continue
                
            channel_state = channel_map[request.user_id]
            
            # Determine required resources
            required_blocks = self._calculate_required_blocks(request, channel_state)
            
            if len(available_blocks) >= required_blocks:
                # Allocate resources
                allocated_blocks = available_blocks[:required_blocks]
                available_blocks = available_blocks[required_blocks:]
                
                # Power allocation
                power_allocation = self._optimize_power_allocation(
                    request, channel_state, allocated_blocks
                )
                
                # Determine MCS
                mcs = self._select_modulation_coding_scheme(channel_state)
                
                # Calculate achievable rate
                achievable_rate = self._calculate_achievable_rate(
                    allocated_blocks, channel_state, power_allocation, mcs
                )
                
                # Create allocation
                allocation = ResourceAllocation(
                    user_id=request.user_id,
                    resource_blocks=allocated_blocks,
                    power_allocation=power_allocation,
                    modulation_coding_scheme=mcs,
                    transmission_duration=self._calculate_transmission_duration(
                        request.data_size_bits, achievable_rate
                    ),
                    achievable_rate=achievable_rate
                )
                
                allocations.append(allocation)
                self.allocation_stats['successful_allocations'] += 1
            else:
                # Request blocked
                self.allocation_stats['blocked_requests'] += 1
        
        # Update statistics
        self._update_allocation_statistics(allocations)
        
        return allocations
    
    def allocate_downlink_resources(self, base_station_id: int, user_requests: List[UserRequest],
                                    channel_states: List[ChannelState]) -> List[ResourceAllocation]:
        """
        Allocate downlink resources from base station to users
        
        Args:
            base_station_id: ID of the transmitting base station
            user_requests: List of user requests
            channel_states: Channel state information
            
        Returns:
            List of resource allocations
        """
        if self.scheduling_algorithm == 'round_robin':
            return self._round_robin_scheduling(user_requests, channel_states)
        elif self.scheduling_algorithm == 'proportional_fair':
            return self._proportional_fair_scheduling(user_requests, channel_states)
        elif self.scheduling_algorithm == 'max_rate':
            return self._max_rate_scheduling(user_requests, channel_states)
        else:
            return self._priority_based_scheduling(user_requests, channel_states)
    
    def _prioritize_requests(self, user_requests: List[UserRequest]) -> List[UserRequest]:
        """Prioritize user requests based on QoS requirements"""
        def priority_key(request):
            # Primary: Priority class
            priority_score = self.priority_weights[request.priority_class]
            
            # Secondary: Delay urgency
            delay_urgency = 1.0 / (request.delay_tolerance_ms + 1e-6)
            
            # Tertiary: Data size (smaller first for fairness)
            size_factor = 1.0 / (request.data_size_bits + 1e-6)
            
            return (priority_score, delay_urgency, size_factor)
        
        return sorted(user_requests, key=priority_key, reverse=True)
    
    def _calculate_required_blocks(self, request: UserRequest, channel_state: ChannelState) -> int:
        """Calculate number of resource blocks required for request"""
        # Base spectral efficiency from CQI
        spectral_efficiency = self._cqi_to_spectral_efficiency(channel_state.channel_quality_indicator)
        
        # Achievable rate per resource block
        rate_per_block = self.resource_block_bandwidth * spectral_efficiency
        
        # Required blocks (with some margin)
        required_rate = request.data_size_bits / (request.delay_tolerance_ms * 1e-3)
        required_blocks = math.ceil(required_rate / rate_per_block)
        
        # Limit to available blocks
        return min(required_blocks, self.num_resource_blocks)
    
    def _optimize_power_allocation(self, request: UserRequest, channel_state: ChannelState,
                                   allocated_blocks: List[int]) -> float:
        """Optimize power allocation for given resource allocation"""
        # Water-filling algorithm for power allocation
        num_blocks = len(allocated_blocks)
        
        if num_blocks == 0:
            return 0.0
        
        # Noise plus interference power per block
        noise_power_per_block = self._calculate_noise_power_per_block()
        interference_power = channel_state.interference_power
        
        # Channel gain (linear scale)
        channel_gain_linear = 10**(-channel_state.path_loss / 10)
        
        # Water-filling level
        total_power_linear = 10**(self.max_transmit_power / 10) * 1e-3  # Convert to Watts
        water_level = (total_power_linear + num_blocks * (noise_power_per_block + interference_power)) / (num_blocks * channel_gain_linear)
        
        # Power per block
        power_per_block = max(0, water_level - (noise_power_per_block + interference_power) / channel_gain_linear)
        
        # Convert back to dBm
        if power_per_block > 0:
            power_dbm = 10 * math.log10(power_per_block * 1000)
            return min(max(power_dbm, self.min_transmit_power), self.max_transmit_power)
        else:
            return self.min_transmit_power
    
    def _select_modulation_coding_scheme(self, channel_state: ChannelState) -> int:
        """Select appropriate MCS based on channel conditions"""
        cqi = channel_state.channel_quality_indicator
        
        # CQI to MCS mapping (simplified)
        if cqi >= 12:
            return 15  # 64QAM, high code rate
        elif cqi >= 9:
            return 12  # 16QAM, medium code rate
        elif cqi >= 6:
            return 8   # QPSK, medium code rate
        elif cqi >= 3:
            return 4   # QPSK, low code rate
        else:
            return 0   # BPSK, very low code rate
    
    def _calculate_achievable_rate(self, allocated_blocks: List[int], channel_state: ChannelState,
                                   power_allocation: float, mcs: int) -> float:
        """Calculate achievable data rate for allocation"""
        num_blocks = len(allocated_blocks)
        
        if num_blocks == 0:
            return 0.0
        
        # Spectral efficiency from MCS
        spectral_efficiency = self._mcs_to_spectral_efficiency(mcs)
        
        # Total bandwidth
        total_bandwidth = num_blocks * self.resource_block_bandwidth
        
        # SINR calculation
        signal_power = 10**(power_allocation / 10) * 1e-3  # Watts
        noise_power = self._calculate_noise_power_per_block() * num_blocks
        interference_power = channel_state.interference_power
        channel_gain = 10**(-channel_state.path_loss / 10)
        
        received_signal_power = signal_power * channel_gain
        sinr_linear = received_signal_power / (noise_power + interference_power)
        
        # Shannon capacity with spectral efficiency limit
        shannon_rate = total_bandwidth * math.log2(1 + sinr_linear)
        practical_rate = total_bandwidth * spectral_efficiency
        
        return min(shannon_rate, practical_rate)
    
    def _calculate_transmission_duration(self, data_size_bits: int, achievable_rate: float) -> float:
        """Calculate required transmission duration"""
        if achievable_rate <= 0:
            return float('inf')
        
        return data_size_bits / achievable_rate
    
    def _round_robin_scheduling(self, user_requests: List[UserRequest],
                                channel_states: List[ChannelState]) -> List[ResourceAllocation]:
        """Round-robin scheduling algorithm"""
        # Simple round-robin implementation
        allocations = []
        blocks_per_user = self.num_resource_blocks // len(user_requests) if user_requests else 0
        
        for i, request in enumerate(user_requests):
            start_block = i * blocks_per_user
            end_block = min((i + 1) * blocks_per_user, self.num_resource_blocks)
            allocated_blocks = list(range(start_block, end_block))
            
            if allocated_blocks:
                channel_state = next((cs for cs in channel_states if cs.user_id == request.user_id), None)
                if channel_state:
                    power_allocation = self.max_transmit_power / len(allocated_blocks)
                    mcs = self._select_modulation_coding_scheme(channel_state)
                    achievable_rate = self._calculate_achievable_rate(
                        allocated_blocks, channel_state, power_allocation, mcs
                    )
                    
                    allocation = ResourceAllocation(
                        user_id=request.user_id,
                        resource_blocks=allocated_blocks,
                        power_allocation=power_allocation,
                        modulation_coding_scheme=mcs,
                        transmission_duration=self._calculate_transmission_duration(
                            request.data_size_bits, achievable_rate
                        ),
                        achievable_rate=achievable_rate
                    )
                    allocations.append(allocation)
        
        return allocations
    
    def _proportional_fair_scheduling(self, user_requests: List[UserRequest],
                                      channel_states: List[ChannelState]) -> List[ResourceAllocation]:
        """Proportional fair scheduling algorithm"""
        allocations = []
        
        # Calculate proportional fair metrics
        user_metrics = []
        for request in user_requests:
            channel_state = next((cs for cs in channel_states if cs.user_id == request.user_id), None)
            if channel_state:
                current_rate = self._estimate_achievable_rate(channel_state)
                average_rate = self._get_average_rate(request.user_id)
                
                if average_rate > 0:
                    metric = current_rate / average_rate
                else:
                    metric = current_rate
                
                user_metrics.append((metric, request, channel_state))
        
        # Sort by proportional fair metric
        user_metrics.sort(key=lambda x: x[0], reverse=True)
        
        # Allocate resources greedily
        available_blocks = list(range(self.num_resource_blocks))
        
        for metric, request, channel_state in user_metrics:
            required_blocks = self._calculate_required_blocks(request, channel_state)
            
            if len(available_blocks) >= required_blocks:
                allocated_blocks = available_blocks[:required_blocks]
                available_blocks = available_blocks[required_blocks:]
                
                power_allocation = self._optimize_power_allocation(
                    request, channel_state, allocated_blocks
                )
                mcs = self._select_modulation_coding_scheme(channel_state)
                achievable_rate = self._calculate_achievable_rate(
                    allocated_blocks, channel_state, power_allocation, mcs
                )
                
                allocation = ResourceAllocation(
                    user_id=request.user_id,
                    resource_blocks=allocated_blocks,
                    power_allocation=power_allocation,
                    modulation_coding_scheme=mcs,
                    transmission_duration=self._calculate_transmission_duration(
                        request.data_size_bits, achievable_rate
                    ),
                    achievable_rate=achievable_rate
                )
                allocations.append(allocation)
                
                # Update user history
                self.user_histories[request.user_id].append(achievable_rate)
        
        return allocations
    
    def _max_rate_scheduling(self, user_requests: List[UserRequest],
                             channel_states: List[ChannelState]) -> List[ResourceAllocation]:
        """Maximum rate scheduling algorithm"""
        # Always select user with best channel conditions
        allocations = []
        
        # Calculate rates for all users
        user_rates = []
        for request in user_requests:
            channel_state = next((cs for cs in channel_states if cs.user_id == request.user_id), None)
            if channel_state:
                rate = self._estimate_achievable_rate(channel_state)
                user_rates.append((rate, request, channel_state))
        
        # Sort by rate (highest first)
        user_rates.sort(key=lambda x: x[0], reverse=True)
        
        # Allocate all resources to best users
        available_blocks = list(range(self.num_resource_blocks))
        
        for rate, request, channel_state in user_rates:
            if not available_blocks:
                break
                
            required_blocks = min(
                self._calculate_required_blocks(request, channel_state),
                len(available_blocks)
            )
            
            allocated_blocks = available_blocks[:required_blocks]
            available_blocks = available_blocks[required_blocks:]
            
            if allocated_blocks:
                power_allocation = self._optimize_power_allocation(
                    request, channel_state, allocated_blocks
                )
                mcs = self._select_modulation_coding_scheme(channel_state)
                achievable_rate = self._calculate_achievable_rate(
                    allocated_blocks, channel_state, power_allocation, mcs
                )
                
                allocation = ResourceAllocation(
                    user_id=request.user_id,
                    resource_blocks=allocated_blocks,
                    power_allocation=power_allocation,
                    modulation_coding_scheme=mcs,
                    transmission_duration=self._calculate_transmission_duration(
                        request.data_size_bits, achievable_rate
                    ),
                    achievable_rate=achievable_rate
                )
                allocations.append(allocation)
        
        return allocations
    
    def _priority_based_scheduling(self, user_requests: List[UserRequest],
                                   channel_states: List[ChannelState]) -> List[ResourceAllocation]:
        """Priority-based scheduling algorithm"""
        return self.allocate_uplink_resources(user_requests, channel_states)
    
    def _cqi_to_spectral_efficiency(self, cqi: int) -> float:
        """Convert CQI to spectral efficiency (bits/s/Hz)"""
        cqi_table = [0.0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 
                     1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 
                     5.1152, 5.5547]
        
        return cqi_table[min(cqi, 15)]
    
    def _mcs_to_spectral_efficiency(self, mcs: int) -> float:
        """Convert MCS to spectral efficiency"""
        mcs_table = [0.1172, 0.1523, 0.2344, 0.3008, 0.3770, 0.4785, 0.6016,
                     0.7402, 0.8770, 1.0273, 1.1758, 1.3262, 1.4766, 1.6953,
                     1.9141, 2.1602, 2.4063, 2.5664, 2.7305, 3.0293, 3.3223,
                     3.6094, 3.9023, 4.2129, 4.5234, 4.8164, 5.1152, 5.4141,
                     5.5547]
        
        return mcs_table[min(mcs, len(mcs_table) - 1)]
    
    def _calculate_noise_power_per_block(self) -> float:
        """Calculate thermal noise power per resource block"""
        boltzmann_constant = 1.380649e-23  # J/K
        temperature = 290  # K
        noise_power = boltzmann_constant * temperature * self.resource_block_bandwidth
        return noise_power  # Watts
    
    def _estimate_achievable_rate(self, channel_state: ChannelState) -> float:
        """Estimate achievable rate for channel state"""
        spectral_efficiency = self._cqi_to_spectral_efficiency(channel_state.channel_quality_indicator)
        return self.resource_block_bandwidth * spectral_efficiency
    
    def _get_average_rate(self, user_id: int) -> float:
        """Get historical average rate for user"""
        if user_id in self.user_histories and self.user_histories[user_id]:
            return np.mean(self.user_histories[user_id][-10:])  # Last 10 values
        else:
            return 1.0  # Default value for new users
    
    def _update_allocation_statistics(self, allocations: List[ResourceAllocation]):
        """Update MAC layer statistics"""
        self.allocation_stats['total_allocated_blocks'] += sum(
            len(alloc.resource_blocks) for alloc in allocations
        )
        
        for allocation in allocations:
            self.allocation_stats['throughput_per_user'][allocation.user_id] += allocation.achievable_rate
        
        self.current_frame += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MAC layer performance statistics"""
        total_requests = (self.allocation_stats['successful_allocations'] + 
                         self.allocation_stats['blocked_requests'])
        
        blocking_probability = (self.allocation_stats['blocked_requests'] / total_requests 
                               if total_requests > 0 else 0.0)
        
        average_throughput = (np.mean(list(self.allocation_stats['throughput_per_user'].values()))
                             if self.allocation_stats['throughput_per_user'] else 0.0)
        
        resource_utilization = (self.allocation_stats['total_allocated_blocks'] / 
                               (self.current_frame * self.num_resource_blocks)
                               if self.current_frame > 0 else 0.0)
        
        return {
            'blocking_probability': blocking_probability,
            'resource_utilization': resource_utilization,
            'average_throughput_bps': average_throughput,
            'total_allocations': self.allocation_stats['successful_allocations'],
            'total_blocked': self.allocation_stats['blocked_requests'],
            'frames_processed': self.current_frame
        }
    
    def reset_statistics(self):
        """Reset MAC layer statistics"""
        self.allocation_stats = {
            'total_allocated_blocks': 0,
            'successful_allocations': 0,
            'blocked_requests': 0,
            'average_delay': 0.0,
            'throughput_per_user': defaultdict(float)
        }
        self.current_frame = 0
        self.user_histories.clear()