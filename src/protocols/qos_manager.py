"""
Quality of Service (QoS) Manager for SAGIN Networks
Implements traffic classification, admission control, and QoS enforcement
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time


class ServiceClass(Enum):
    """5G NR service classes"""
    EMBB = "enhanced_mobile_broadband"
    URLLC = "ultra_reliable_low_latency"
    MMTC = "massive_machine_type"
    BROADCAST = "broadcast_multicast"


class QoSFlowType(Enum):
    """QoS flow types"""
    GBR = "guaranteed_bit_rate"
    NON_GBR = "non_guaranteed_bit_rate"
    DELAY_CRITICAL = "delay_critical"


@dataclass
class QoSRequirements:
    """QoS requirements for a flow"""
    service_class: ServiceClass
    flow_type: QoSFlowType
    guaranteed_bit_rate: float = 0.0  # bps
    maximum_bit_rate: float = float('inf')  # bps
    packet_delay_budget: float = 100.0  # ms
    packet_error_rate: float = 1e-3
    priority_level: int = 5  # 1 (highest) to 15 (lowest)
    averaging_window: float = 2000.0  # ms
    maximum_data_burst: float = 0.0  # bytes


@dataclass
class TrafficFlow:
    """Represents a traffic flow"""
    flow_id: int
    source_id: int
    destination_id: int
    qos_requirements: QoSRequirements
    arrival_time: float
    packets: List['Packet'] = field(default_factory=list)
    allocated_resources: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Packet:
    """Network packet"""
    packet_id: int
    flow_id: int
    size_bytes: int
    arrival_time: float
    deadline: float
    priority: int
    service_class: ServiceClass
    transmission_attempts: int = 0
    delivered: bool = False
    delivery_time: Optional[float] = None


@dataclass
class QoSMetrics:
    """QoS performance metrics"""
    throughput: float = 0.0
    delay: float = 0.0
    jitter: float = 0.0
    packet_loss_rate: float = 0.0
    reliability: float = 1.0
    resource_utilization: float = 0.0


class AdmissionController:
    """Admission control for QoS flows"""
    
    def __init__(self, config: Dict):
        self.max_flows = config.get('max_flows', 1000)
        self.resource_budget = config.get('resource_budget', 1.0)
        self.current_resource_usage = 0.0
        self.active_flows = {}
        
    def admit_flow(self, flow: TrafficFlow, estimated_resources: float) -> bool:
        """
        Decide whether to admit a new flow
        
        Args:
            flow: Traffic flow to consider
            estimated_resources: Estimated resource requirement
            
        Returns:
            True if flow should be admitted
        """
        # Check capacity constraints
        if len(self.active_flows) >= self.max_flows:
            return False
        
        # Check resource availability
        if self.current_resource_usage + estimated_resources > self.resource_budget:
            # Try preemption for lower priority flows
            return self._try_preemption(flow, estimated_resources)
        
        return True
    
    def _try_preemption(self, new_flow: TrafficFlow, required_resources: float) -> bool:
        """Try to preempt lower priority flows"""
        # Find flows with lower priority
        preemptable_flows = [
            (fid, flow) for fid, flow in self.active_flows.items()
            if flow.qos_requirements.priority_level > new_flow.qos_requirements.priority_level
        ]
        
        # Sort by priority (lowest first for preemption)
        preemptable_flows.sort(key=lambda x: x[1].qos_requirements.priority_level, reverse=True)
        
        freed_resources = 0.0
        for flow_id, flow in preemptable_flows:
            flow_resources = flow.allocated_resources.get('estimated_requirement', 0.0)
            freed_resources += flow_resources
            
            if freed_resources >= required_resources:
                # Preempt flows
                for fid, _ in preemptable_flows:
                    if fid in self.active_flows:
                        del self.active_flows[fid]
                        self.current_resource_usage -= flow_resources
                        if freed_resources >= required_resources:
                            break
                return True
        
        return False


class TrafficShaper:
    """Traffic shaping and policing"""
    
    def __init__(self, config: Dict):
        self.token_bucket_size = config.get('token_bucket_size', 10000)  # bytes
        self.token_rate = config.get('token_rate', 1e6)  # bytes/s
        self.buckets = defaultdict(lambda: self.token_bucket_size)
        self.last_update = defaultdict(float)
        
    def shape_traffic(self, flow: TrafficFlow, current_time: float) -> List[Packet]:
        """
        Apply traffic shaping to flow
        
        Args:
            flow: Traffic flow to shape
            current_time: Current simulation time
            
        Returns:
            List of packets that can be transmitted
        """
        flow_id = flow.flow_id
        
        # Update token bucket
        time_elapsed = current_time - self.last_update[flow_id]
        tokens_to_add = time_elapsed * self.token_rate
        self.buckets[flow_id] = min(
            self.token_bucket_size,
            self.buckets[flow_id] + tokens_to_add
        )
        self.last_update[flow_id] = current_time
        
        # Select packets that can be transmitted
        transmittable_packets = []
        available_tokens = self.buckets[flow_id]
        
        for packet in flow.packets:
            if packet.delivered:
                continue
                
            if available_tokens >= packet.size_bytes:
                transmittable_packets.append(packet)
                available_tokens -= packet.size_bytes
            else:
                break  # Not enough tokens for this packet
        
        # Update bucket
        self.buckets[flow_id] = available_tokens
        
        return transmittable_packets


class QueueManager:
    """Queue management for different service classes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.queues = {
            ServiceClass.URLLC: deque(),
            ServiceClass.EMBB: deque(),
            ServiceClass.MMTC: deque(),
            ServiceClass.BROADCAST: deque()
        }
        
        # Queue limits
        self.queue_limits = {
            ServiceClass.URLLC: config.get('urllc_queue_limit', 100),
            ServiceClass.EMBB: config.get('embb_queue_limit', 1000),
            ServiceClass.MMTC: config.get('mmtc_queue_limit', 5000),
            ServiceClass.BROADCAST: config.get('broadcast_queue_limit', 500)
        }
        
        # Drop policies
        self.drop_policies = {
            ServiceClass.URLLC: 'tail_drop',
            ServiceClass.EMBB: 'red',
            ServiceClass.MMTC: 'tail_drop',
            ServiceClass.BROADCAST: 'tail_drop'
        }
    
    def enqueue_packet(self, packet: Packet, current_time: float) -> bool:
        """
        Enqueue packet with appropriate policy
        
        Args:
            packet: Packet to enqueue
            current_time: Current simulation time
            
        Returns:
            True if packet was successfully enqueued
        """
        service_class = packet.service_class
        queue = self.queues[service_class]
        limit = self.queue_limits[service_class]
        
        # Check deadline for URLLC packets
        if service_class == ServiceClass.URLLC:
            if current_time > packet.deadline:
                return False  # Packet already expired
        
        # Apply drop policy
        if len(queue) >= limit:
            if self.drop_policies[service_class] == 'tail_drop':
                return False  # Drop incoming packet
            elif self.drop_policies[service_class] == 'red':
                # Random Early Detection
                drop_probability = self._calculate_red_drop_probability(queue, limit)
                if np.random.random() < drop_probability:
                    return False
                else:
                    # Drop oldest packet
                    if queue:
                        queue.popleft()
        
        queue.append(packet)
        return True
    
    def dequeue_packets(self, max_packets: int, current_time: float) -> List[Packet]:
        """
        Dequeue packets based on priority scheduling
        
        Args:
            max_packets: Maximum number of packets to dequeue
            current_time: Current simulation time
            
        Returns:
            List of dequeued packets
        """
        dequeued_packets = []
        
        # Strict priority: URLLC > eMBB > mMTC > Broadcast
        for service_class in [ServiceClass.URLLC, ServiceClass.EMBB, 
                             ServiceClass.MMTC, ServiceClass.BROADCAST]:
            queue = self.queues[service_class]
            
            while queue and len(dequeued_packets) < max_packets:
                packet = queue.popleft()
                
                # Check if packet is still valid
                if service_class == ServiceClass.URLLC and current_time > packet.deadline:
                    continue  # Skip expired packet
                
                dequeued_packets.append(packet)
        
        return dequeued_packets
    
    def _calculate_red_drop_probability(self, queue: deque, limit: int) -> float:
        """Calculate RED drop probability"""
        queue_length = len(queue)
        min_threshold = limit * 0.3
        max_threshold = limit * 0.8
        max_drop_prob = 0.1
        
        if queue_length <= min_threshold:
            return 0.0
        elif queue_length >= max_threshold:
            return 1.0
        else:
            return max_drop_prob * (queue_length - min_threshold) / (max_threshold - min_threshold)


class QoSManager:
    """
    Main QoS Manager for SAGIN networks
    Coordinates admission control, traffic shaping, and queue management
    """
    
    def __init__(self, config: Dict):
        """Initialize QoS Manager"""
        self.config = config
        
        # Components
        self.admission_controller = AdmissionController(config.get('admission_control', {}))
        self.traffic_shaper = TrafficShaper(config.get('traffic_shaping', {}))
        self.queue_manager = QueueManager(config.get('queue_management', {}))
        
        # State
        self.active_flows = {}
        self.flow_metrics = defaultdict(lambda: QoSMetrics())
        self.global_metrics = QoSMetrics()
        
        # Monitoring
        self.monitoring_window = config.get('monitoring_window', 1000.0)  # ms
        self.packet_history = deque(maxlen=10000)
        
        # Performance tracking
        self.performance_history = {
            'throughput': deque(maxlen=1000),
            'delay': deque(maxlen=1000),
            'packet_loss': deque(maxlen=1000)
        }
    
    def create_flow(self, source_id: int, destination_id: int, 
                    qos_requirements: QoSRequirements, current_time: float) -> Optional[int]:
        """
        Create a new QoS flow
        
        Args:
            source_id: Source node ID
            destination_id: Destination node ID
            qos_requirements: QoS requirements for the flow
            current_time: Current simulation time
            
        Returns:
            Flow ID if admitted, None otherwise
        """
        flow_id = len(self.active_flows)
        
        # Create flow
        flow = TrafficFlow(
            flow_id=flow_id,
            source_id=source_id,
            destination_id=destination_id,
            qos_requirements=qos_requirements,
            arrival_time=current_time
        )
        
        # Estimate resource requirements
        estimated_resources = self._estimate_resource_requirements(qos_requirements)
        
        # Admission control
        if self.admission_controller.admit_flow(flow, estimated_resources):
            self.active_flows[flow_id] = flow
            flow.allocated_resources['estimated_requirement'] = estimated_resources
            return flow_id
        else:
            return None
    
    def add_packet_to_flow(self, flow_id: int, packet_size: int, 
                          current_time: float, deadline_offset: float = None) -> bool:
        """
        Add packet to existing flow
        
        Args:
            flow_id: Flow identifier
            packet_size: Packet size in bytes
            current_time: Current simulation time
            deadline_offset: Deadline offset from current time (ms)
            
        Returns:
            True if packet was accepted
        """
        if flow_id not in self.active_flows:
            return False
        
        flow = self.active_flows[flow_id]
        qos_req = flow.qos_requirements
        
        # Calculate deadline
        if deadline_offset is not None:
            deadline = current_time + deadline_offset * 1e-3
        else:
            deadline = current_time + qos_req.packet_delay_budget * 1e-3
        
        # Create packet
        packet = Packet(
            packet_id=len(self.packet_history),
            flow_id=flow_id,
            size_bytes=packet_size,
            arrival_time=current_time,
            deadline=deadline,
            priority=qos_req.priority_level,
            service_class=qos_req.service_class
        )
        
        # Add to flow
        flow.packets.append(packet)
        self.packet_history.append(packet)
        
        # Queue packet
        return self.queue_manager.enqueue_packet(packet, current_time)
    
    def process_transmission_opportunities(self, max_packets: int, 
                                         current_time: float) -> List[Packet]:
        """
        Process transmission opportunities and return packets to transmit
        
        Args:
            max_packets: Maximum number of packets to transmit
            current_time: Current simulation time
            
        Returns:
            List of packets selected for transmission
        """
        # Dequeue packets from queues
        candidate_packets = self.queue_manager.dequeue_packets(max_packets, current_time)
        
        # Apply traffic shaping per flow
        transmittable_packets = []
        packets_by_flow = defaultdict(list)
        
        # Group packets by flow
        for packet in candidate_packets:
            packets_by_flow[packet.flow_id].append(packet)
        
        # Apply shaping per flow
        for flow_id, packets in packets_by_flow.items():
            if flow_id in self.active_flows:
                flow = self.active_flows[flow_id]
                
                # Temporarily add packets to flow for shaping
                original_packets = flow.packets[:]
                flow.packets = packets
                
                shaped_packets = self.traffic_shaper.shape_traffic(flow, current_time)
                transmittable_packets.extend(shaped_packets)
                
                # Restore original packets
                flow.packets = original_packets
        
        return transmittable_packets[:max_packets]
    
    def record_packet_delivery(self, packet: Packet, current_time: float, 
                              successful: bool = True):
        """
        Record packet delivery outcome
        
        Args:
            packet: Delivered packet
            current_time: Current simulation time
            successful: Whether delivery was successful
        """
        packet.delivered = True
        packet.delivery_time = current_time
        
        # Update flow metrics
        if packet.flow_id in self.active_flows:
            flow = self.active_flows[packet.flow_id]
            flow_metrics = self.flow_metrics[packet.flow_id]
            
            if successful:
                # Calculate delay
                delay = current_time - packet.arrival_time
                flow_metrics.delay = delay
                
                # Update throughput (simplified)
                flow_metrics.throughput += packet.size_bytes * 8  # bits
                
                # Check QoS satisfaction
                qos_req = flow.qos_requirements
                if delay <= qos_req.packet_delay_budget * 1e-3:
                    flow_metrics.reliability += 1.0
                else:
                    flow_metrics.packet_loss_rate += 1.0
            else:
                flow_metrics.packet_loss_rate += 1.0
        
        # Update global metrics
        self._update_global_metrics(current_time)
    
    def get_flow_metrics(self, flow_id: int) -> QoSMetrics:
        """Get performance metrics for specific flow"""
        return self.flow_metrics.get(flow_id, QoSMetrics())
    
    def get_global_metrics(self) -> QoSMetrics:
        """Get global QoS performance metrics"""
        return self.global_metrics
    
    def cleanup_completed_flows(self, current_time: float):
        """Remove completed flows and update statistics"""
        completed_flows = []
        
        for flow_id, flow in self.active_flows.items():
            # Check if flow has been inactive
            if flow.packets and all(p.delivered for p in flow.packets):
                last_activity = max(p.delivery_time or p.arrival_time for p in flow.packets)
                if current_time - last_activity > self.monitoring_window * 1e-3:
                    completed_flows.append(flow_id)
        
        # Remove completed flows
        for flow_id in completed_flows:
            del self.active_flows[flow_id]
            # Keep metrics for analysis
    
    def _estimate_resource_requirements(self, qos_requirements: QoSRequirements) -> float:
        """Estimate resource requirements for QoS flow"""
        # Simplified estimation based on guaranteed bit rate
        base_requirement = qos_requirements.guaranteed_bit_rate / 1e6  # Normalize to Mbps
        
        # Priority factor
        priority_factor = (16 - qos_requirements.priority_level) / 15.0
        
        # Service class factor
        service_factors = {
            ServiceClass.URLLC: 2.0,
            ServiceClass.EMBB: 1.0,
            ServiceClass.MMTC: 0.5,
            ServiceClass.BROADCAST: 0.8
        }
        
        service_factor = service_factors.get(qos_requirements.service_class, 1.0)
        
        return base_requirement * priority_factor * service_factor
    
    def _update_global_metrics(self, current_time: float):
        """Update global performance metrics"""
        # Aggregate metrics from all flows
        total_throughput = sum(metrics.throughput for metrics in self.flow_metrics.values())
        total_packets = len(self.packet_history)
        
        if total_packets > 0:
            delivered_packets = sum(1 for p in self.packet_history if p.delivered)
            lost_packets = total_packets - delivered_packets
            
            self.global_metrics.throughput = total_throughput
            self.global_metrics.packet_loss_rate = lost_packets / total_packets
            
            # Calculate average delay for delivered packets
            delivered_delays = [
                p.delivery_time - p.arrival_time 
                for p in self.packet_history 
                if p.delivered and p.delivery_time
            ]
            
            if delivered_delays:
                self.global_metrics.delay = np.mean(delivered_delays)
                self.global_metrics.jitter = np.std(delivered_delays)
        
        # Update performance history
        self.performance_history['throughput'].append(self.global_metrics.throughput)
        self.performance_history['delay'].append(self.global_metrics.delay)
        self.performance_history['packet_loss'].append(self.global_metrics.packet_loss_rate)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive QoS statistics"""
        stats = {
            'active_flows': len(self.active_flows),
            'total_packets': len(self.packet_history),
            'global_metrics': {
                'throughput_bps': self.global_metrics.throughput,
                'average_delay_ms': self.global_metrics.delay * 1000,
                'jitter_ms': self.global_metrics.jitter * 1000,
                'packet_loss_rate': self.global_metrics.packet_loss_rate,
                'reliability': self.global_metrics.reliability
            },
            'queue_occupancy': {
                service.name: len(queue) 
                for service, queue in self.queue_manager.queues.items()
            },
            'flow_distribution': {
                service.name: sum(1 for flow in self.active_flows.values() 
                                if flow.qos_requirements.service_class == service)
                for service in ServiceClass
            }
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset all QoS statistics"""
        self.flow_metrics.clear()
        self.global_metrics = QoSMetrics()
        self.packet_history.clear()
        self.performance_history = {
            'throughput': deque(maxlen=1000),
            'delay': deque(maxlen=1000),
            'packet_loss': deque(maxlen=1000)
        }