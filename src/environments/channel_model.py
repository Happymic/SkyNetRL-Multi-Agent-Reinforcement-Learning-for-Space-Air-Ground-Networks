"""
3GPP-Compliant Channel Model for SAGIN Networks
Implementation based on 3GPP TR 38.811 specifications
"""

import numpy as np
import math
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class ChannelType(Enum):
    """Channel types for different link scenarios"""
    SATELLITE_TO_GROUND = "sat_to_ground"
    SATELLITE_TO_UAV = "sat_to_uav"
    UAV_TO_GROUND = "uav_to_ground"
    GROUND_TO_GROUND = "ground_to_ground"


@dataclass
class ChannelParameters:
    """Channel model parameters"""
    frequency_ghz: float = 2.0
    bandwidth_mhz: float = 20.0
    noise_figure_db: float = 9.0
    thermal_noise_dbm: float = -174.0
    shadowing_std_db: float = 4.0
    multipath_components: int = 6
    doppler_max_hz: float = 100.0


@dataclass
class LinkBudget:
    """Link budget calculation results"""
    path_loss_db: float
    shadowing_db: float
    fading_db: float
    snr_db: float
    capacity_bps: float
    ber: float


class ThreeGPPChannelModel:
    """
    3GPP-compliant channel model for SAGIN networks
    Implements realistic path loss, fading, and interference models
    """
    
    def __init__(self, config: Dict):
        """Initialize channel model with configuration"""
        self.params = ChannelParameters(
            frequency_ghz=config.get('frequency_ghz', 2.0),
            bandwidth_mhz=config.get('bandwidth_mhz', 20.0),
            noise_figure_db=config.get('noise_figure_db', 9.0),
            thermal_noise_dbm=config.get('thermal_noise_dbm', -174.0),
            shadowing_std_db=config.get('shadowing_std_db', 4.0),
            multipath_components=config.get('multipath_components', 6),
            doppler_max_hz=config.get('doppler_max_hz', 100.0)
        )
        
        # Cache for fading realizations
        self.fading_cache = {}
        self.cache_size = 1000
        
        # Constants
        self.SPEED_OF_LIGHT = 299792458  # m/s
        self.BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
        self.TEMPERATURE_K = 290  # K
        
    def compute_path_loss_3gpp(self, distance_3d_m: float, height_tx_m: float, 
                               height_rx_m: float, channel_type: ChannelType) -> float:
        """
        Compute 3GPP-compliant path loss
        
        Args:
            distance_3d_m: 3D distance in meters
            height_tx_m: Transmitter height in meters
            height_rx_m: Receiver height in meters
            channel_type: Type of communication link
            
        Returns:
            Path loss in dB
        """
        if channel_type == ChannelType.SATELLITE_TO_GROUND:
            return self._satellite_to_ground_path_loss(distance_3d_m, height_tx_m, height_rx_m)
        elif channel_type == ChannelType.SATELLITE_TO_UAV:
            return self._satellite_to_uav_path_loss(distance_3d_m, height_tx_m, height_rx_m)
        elif channel_type == ChannelType.UAV_TO_GROUND:
            return self._uav_to_ground_path_loss(distance_3d_m, height_tx_m, height_rx_m)
        else:
            return self._ground_to_ground_path_loss(distance_3d_m)
    
    def _satellite_to_ground_path_loss(self, distance_3d_m: float, 
                                       height_sat_m: float, height_ground_m: float) -> float:
        """3GPP TR 38.811 satellite-to-ground path loss"""
        # Free space path loss
        fspl_db = 32.45 + 20 * math.log10(distance_3d_m / 1000) + 20 * math.log10(self.params.frequency_ghz)
        
        # Additional losses for satellite links
        atmospheric_loss_db = 0.5  # Simplified atmospheric absorption
        polarization_loss_db = 0.3
        
        # Elevation angle effect
        elevation_angle_rad = math.asin(height_sat_m / distance_3d_m)
        elevation_factor = max(0, math.sin(elevation_angle_rad))
        elevation_loss_db = -10 * math.log10(elevation_factor + 0.1)
        
        total_loss = fspl_db + atmospheric_loss_db + polarization_loss_db + elevation_loss_db
        return total_loss
    
    def _satellite_to_uav_path_loss(self, distance_3d_m: float, 
                                    height_sat_m: float, height_uav_m: float) -> float:
        """Satellite-to-UAV path loss (reduced atmospheric effects)"""
        fspl_db = 32.45 + 20 * math.log10(distance_3d_m / 1000) + 20 * math.log10(self.params.frequency_ghz)
        
        # Reduced atmospheric loss for high-altitude UAVs
        atmospheric_loss_db = 0.2 * (1 - height_uav_m / 20000)  # Altitude-dependent
        
        return fspl_db + atmospheric_loss_db
    
    def _uav_to_ground_path_loss(self, distance_3d_m: float, 
                                 height_uav_m: float, height_ground_m: float) -> float:
        """UAV-to-ground path loss with air-to-ground propagation model"""
        # 3GPP air-to-ground channel model
        distance_2d_m = math.sqrt(distance_3d_m**2 - (height_uav_m - height_ground_m)**2)
        
        # Probability of LoS
        if distance_2d_m == 0:
            prob_los = 1.0
        else:
            elevation_angle_deg = math.degrees(math.atan((height_uav_m - height_ground_m) / distance_2d_m))
            prob_los = 1 / (1 + 20 * math.exp(-0.5 * (elevation_angle_deg - 20)))
        
        # LoS path loss
        pl_los_db = 20 * math.log10(distance_3d_m) + 20 * math.log10(self.params.frequency_ghz) + 32.45
        
        # NLoS path loss
        pl_nlos_db = pl_los_db + 20  # Additional 20 dB for NLoS
        
        # Combined path loss
        path_loss_db = prob_los * pl_los_db + (1 - prob_los) * pl_nlos_db
        
        return path_loss_db
    
    def _ground_to_ground_path_loss(self, distance_3d_m: float) -> float:
        """Ground-to-ground path loss (urban/suburban model)"""
        # Okumura-Hata model for urban environment
        if distance_3d_m < 1000:  # Near field
            path_loss_db = 32.45 + 20 * math.log10(distance_3d_m) + 20 * math.log10(self.params.frequency_ghz)
        else:  # Far field
            path_loss_db = 46.3 + 33.9 * math.log10(self.params.frequency_ghz) - \
                          13.82 * math.log10(30) + (44.9 - 6.55 * math.log10(30)) * \
                          math.log10(distance_3d_m / 1000)
        
        return path_loss_db
    
    def compute_shadowing(self, seed: Optional[int] = None) -> float:
        """
        Compute log-normal shadowing
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Shadowing loss in dB
        """
        if seed is not None:
            np.random.seed(seed)
        
        return np.random.normal(0, self.params.shadowing_std_db)
    
    def compute_small_scale_fading(self, channel_type: ChannelType, 
                                   doppler_shift_hz: float = 0.0,
                                   seed: Optional[int] = None) -> float:
        """
        Compute small-scale fading (Rician/Rayleigh)
        
        Args:
            channel_type: Type of communication link
            doppler_shift_hz: Doppler shift in Hz
            seed: Random seed for reproducibility
            
        Returns:
            Fading loss in dB
        """
        if seed is not None:
            np.random.seed(seed)
        
        if channel_type in [ChannelType.SATELLITE_TO_GROUND, ChannelType.SATELLITE_TO_UAV]:
            # Rician fading for satellite links (strong LoS component)
            k_factor_db = 10  # Strong LoS
            return self._rician_fading(k_factor_db)
        else:
            # Rayleigh fading for terrestrial links
            return self._rayleigh_fading()
    
    def _rician_fading(self, k_factor_db: float) -> float:
        """Generate Rician fading realization"""
        k_linear = 10**(k_factor_db / 10)
        
        # LoS component
        los_component = math.sqrt(k_linear / (k_linear + 1))
        
        # Multipath components
        i_component = np.random.normal(0, 1/math.sqrt(2*(k_linear + 1)))
        q_component = np.random.normal(0, 1/math.sqrt(2*(k_linear + 1)))
        
        # Total amplitude
        amplitude = abs(los_component + i_component + 1j * q_component)
        
        # Convert to dB
        fading_db = 20 * math.log10(amplitude)
        
        return fading_db
    
    def _rayleigh_fading(self) -> float:
        """Generate Rayleigh fading realization"""
        i_component = np.random.normal(0, 1/math.sqrt(2))
        q_component = np.random.normal(0, 1/math.sqrt(2))
        
        amplitude = math.sqrt(i_component**2 + q_component**2)
        fading_db = 20 * math.log10(amplitude)
        
        return fading_db
    
    def compute_doppler_shift(self, velocity_ms: float, elevation_angle_rad: float) -> float:
        """
        Compute Doppler shift
        
        Args:
            velocity_ms: Relative velocity in m/s
            elevation_angle_rad: Elevation angle in radians
            
        Returns:
            Doppler shift in Hz
        """
        doppler_hz = velocity_ms * self.params.frequency_ghz * 1e9 / self.SPEED_OF_LIGHT * math.cos(elevation_angle_rad)
        return doppler_hz
    
    def compute_thermal_noise_power(self) -> float:
        """
        Compute thermal noise power
        
        Returns:
            Noise power in dBm
        """
        # Thermal noise power = kTB + NF
        noise_power_w = self.BOLTZMANN_CONSTANT * self.TEMPERATURE_K * self.params.bandwidth_mhz * 1e6
        noise_power_dbm = 10 * math.log10(noise_power_w * 1000)
        
        # Add noise figure
        total_noise_dbm = noise_power_dbm + self.params.noise_figure_db
        
        return total_noise_dbm
    
    def compute_link_budget(self, tx_power_dbm: float, tx_gain_db: float, rx_gain_db: float,
                           distance_3d_m: float, height_tx_m: float, height_rx_m: float,
                           channel_type: ChannelType, velocity_ms: float = 0.0) -> LinkBudget:
        """
        Compute complete link budget
        
        Args:
            tx_power_dbm: Transmit power in dBm
            tx_gain_db: Transmit antenna gain in dB
            rx_gain_db: Receive antenna gain in dB
            distance_3d_m: 3D distance in meters
            height_tx_m: Transmitter height in meters
            height_rx_m: Receiver height in meters
            channel_type: Type of communication link
            velocity_ms: Relative velocity in m/s
            
        Returns:
            LinkBudget object with all calculated values
        """
        # Path loss
        path_loss_db = self.compute_path_loss_3gpp(distance_3d_m, height_tx_m, height_rx_m, channel_type)
        
        # Shadowing
        shadowing_db = self.compute_shadowing()
        
        # Small-scale fading
        elevation_angle_rad = math.asin((height_tx_m - height_rx_m) / distance_3d_m) if distance_3d_m > 0 else 0
        doppler_shift_hz = self.compute_doppler_shift(velocity_ms, elevation_angle_rad)
        fading_db = self.compute_small_scale_fading(channel_type, doppler_shift_hz)
        
        # Noise power
        noise_power_dbm = self.compute_thermal_noise_power()
        
        # Received power
        rx_power_dbm = tx_power_dbm + tx_gain_db + rx_gain_db - path_loss_db - shadowing_db + fading_db
        
        # SNR
        snr_db = rx_power_dbm - noise_power_dbm
        
        # Channel capacity (Shannon)
        snr_linear = 10**(snr_db / 10)
        capacity_bps = self.params.bandwidth_mhz * 1e6 * math.log2(1 + snr_linear)
        
        # BER estimation (QPSK)
        ber = 0.5 * math.erfc(math.sqrt(snr_linear))
        
        return LinkBudget(
            path_loss_db=path_loss_db,
            shadowing_db=shadowing_db,
            fading_db=fading_db,
            snr_db=snr_db,
            capacity_bps=capacity_bps,
            ber=ber
        )
    
    def compute_interference_power(self, interferer_powers: List[float], 
                                   interferer_distances: List[float],
                                   interferer_channel_types: List[ChannelType]) -> float:
        """
        Compute total interference power from multiple interferers
        
        Args:
            interferer_powers: List of interferer transmit powers in dBm
            interferer_distances: List of distances to interferers in meters
            interferer_channel_types: List of channel types for each interferer
            
        Returns:
            Total interference power in dBm
        """
        total_interference_linear = 0
        
        for power_dbm, distance_m, channel_type in zip(interferer_powers, interferer_distances, interferer_channel_types):
            # Simplified interference calculation
            path_loss_db = self.compute_path_loss_3gpp(distance_m, 1000, 10, channel_type)
            received_power_dbm = power_dbm - path_loss_db
            received_power_linear = 10**(received_power_dbm / 10)
            total_interference_linear += received_power_linear
        
        if total_interference_linear > 0:
            return 10 * math.log10(total_interference_linear)
        else:
            return -float('inf')
    
    def get_channel_quality_indicator(self, snr_db: float) -> int:
        """
        Convert SNR to CQI (Channel Quality Indicator)
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            CQI value (0-15)
        """
        # Simplified CQI mapping
        if snr_db < -6.5:
            return 0
        elif snr_db < -4.5:
            return 1
        elif snr_db < -2.5:
            return 2
        elif snr_db < -0.5:
            return 3
        elif snr_db < 1.5:
            return 4
        elif snr_db < 3.5:
            return 5
        elif snr_db < 5.5:
            return 6
        elif snr_db < 7.5:
            return 7
        elif snr_db < 9.5:
            return 8
        elif snr_db < 11.5:
            return 9
        elif snr_db < 13.5:
            return 10
        elif snr_db < 15.5:
            return 11
        elif snr_db < 17.5:
            return 12
        elif snr_db < 19.5:
            return 13
        elif snr_db < 21.5:
            return 14
        else:
            return 15