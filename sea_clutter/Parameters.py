from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Dataclass parameter blocks
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class RadarParams:
    prf: float = 5000.0            # Pulse-repetition frequency [Hz]
    n_pulses: int = 128            # Pulses per coherent processing interval (CPI)
    n_ranges: int = 128            # Range bins
    range_resolution: float = 1    # [m]
    carrier_wavelength: float = 0.03  # [m] – unused but kept for completeness

@dataclass
class ClutterParams:
    mean_power_db: float = -20.0    # Average clutter power per cell [dB]
    shape_param: float = 0.01        # Gamma shape κ; smaller → heavier tail
    ar_coeff: float = 0.98          # Slow-time AR(1) coefficient
    bragg_offset_hz: Optional[float] = 25.0  # Bragg Doppler [Hz]
    bragg_width_hz: float = 2.0     # Bragg peak width [Hz]
    bragg_power_rel: float = 5.0    # Bragg peak height over background [dB]
    wave_speed_mps: float = 2.0     # Wave propagation speed toward radar [m/s]


@dataclass
class Target:
    rng_idx: int
    doppler_hz: float
    power: float = 10.0             # Linear power for the central cell

@dataclass
class SequenceParams:
    n_frames: int = 25            # Frames to simulate
    frame_rate_hz: float = 2.0      # Frames per second

class TargetType(Enum):
    FISHING_VESSEL = "fishing"      # Slow, meandering, affected by waves
    CARGO_SHIP = "cargo"           # Fast, steady course, less affected by waves  
    PATROL_BOAT = "patrol"         # Variable speed, frequent course changes
    SMALL_CRAFT = "small"          # Very affected by waves, erratic movement
    SAILBOAT = "sailboat"          # Wind-dependent, gentle movements
    SPEEDBOAT = "speedboat"        # Fast, agile, sudden speed changes
    FIXED = "fixed"                # No specific target, used for clutter only

@dataclass
class RealisticTarget:
    rng_idx: int
    doppler_hz: float
    power: float = 10.0
    target_type: TargetType = TargetType.CARGO_SHIP
    
    # Movement parameters
    base_velocity_mps: float = 5.0      # Base radial velocity [m/s]
    velocity_noise_std: float = 0.5     # Standard deviation of velocity noise
    max_velocity_mps: float = 20.0      # Maximum possible velocity
    min_velocity_mps: float = -20.0     # Minimum possible velocity (negative = approaching)
    
    # Internal state for realistic movement
    current_velocity_mps: float = 0.0   # Current radial velocity
    velocity_trend: float = 0.0         # Persistent velocity trend


def get_clutter_params_for_sea_state(state: int) -> ClutterParams:
    """
    Map WMO sea states (1, 3, 5, 7, 9) to realistic model parameters.
    
    Sea state descriptions:
    1 - Calm (0-0.1m waves)
    3 - Slight (0.5-1.25m waves) 
    5 - Moderate (2.5-4m waves)
    7 - Rough (4-6m waves)
    9 - Very rough (7-9m waves)
    """
    configs = {
        1: {  # Calm seas
            'mean_power_db': -35.0,      # Very low clutter power
            'shape_param': 0.8,          # More Gaussian-like (less spiky)
            'ar_coeff': 0.998,           # Very stable, slow decorrelation
            'bragg_offset_hz': 15.0,     # Weak Bragg lines
            'bragg_width_hz': 1.0,       # Narrow Bragg peaks
            'bragg_power_rel': 3.0,      # Weak Bragg enhancement
            'wave_speed_mps': 1.0        # Slow wave movement
        },
        3: {  # Slight seas
            'mean_power_db': -28.0,      # Low clutter power
            'shape_param': 0.5,          # Moderate non-Gaussianity
            'ar_coeff': 0.995,           # High correlation
            'bragg_offset_hz': 20.0,     # Moderate Bragg lines
            'bragg_width_hz': 1.5,       # Moderate Bragg width
            'bragg_power_rel': 4.0,      # Moderate Bragg enhancement
            'wave_speed_mps': 2.0        # Moderate wave movement
        },
        5: {  # Moderate seas
            'mean_power_db': 10.0,      # Moderate clutter power
            'shape_param': 0.1,          # Significant non-Gaussianity
            'ar_coeff': 0.95,           # Moderate correlation
            'bragg_offset_hz': 25.0,     # Strong Bragg lines
            'bragg_width_hz': 2.0,       # Wider Bragg peaks
            'bragg_power_rel': 5.0,      # Strong Bragg enhancement
            'wave_speed_mps': 3.0        # Moderate-fast wave movement
        },
        7: {  # Rough seas
            'mean_power_db': -18.0,      # High clutter power
            'shape_param': 0.15,         # Strong non-Gaussianity (spiky)
            'ar_coeff': 0.975,           # Lower correlation (faster decorrelation)
            'bragg_offset_hz': 30.0,     # Very strong Bragg lines
            'bragg_width_hz': 3.0,       # Broad Bragg peaks
            'bragg_power_rel': 6.0,      # Very strong Bragg enhancement
            'wave_speed_mps': 4.5        # Fast wave movement
        },
        9: {  # Very rough seas
            'mean_power_db': -15.0,      # Very high clutter power
            'shape_param': 0.05,         # Extreme non-Gaussianity (very spiky)
            'ar_coeff': 0.950,           # Low correlation (rapid decorrelation)
            'bragg_offset_hz': 35.0,     # Extreme Bragg lines
            'bragg_width_hz': 4.0,       # Very broad Bragg peaks
            'bragg_power_rel': 8.0,      # Extreme Bragg enhancement
            'wave_speed_mps': 6.0        # Very fast wave movement
        },
    }
    
    if state not in configs:
        raise ValueError(f"Unsupported sea state {state}; choose from {list(configs.keys())}")
    
    params = configs[state]
    return ClutterParams(**params)

def create_realistic_target(target_type: TargetType, initial_range_idx: int, rp: RadarParams) -> RealisticTarget:
    """Create a target with realistic parameters based on vessel type."""
    
    # Standardized movement parameters for all targets
    standard_velocity_noise_std = 0.5
    
    # Base configurations for different vessel types
    if target_type == TargetType.FISHING_VESSEL:
        base_velocity = np.random.uniform(-8, 8)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.01, 0.08),
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=12.0,
            min_velocity_mps=-12.0,
            current_velocity_mps=base_velocity
        )
    
    elif target_type == TargetType.CARGO_SHIP:
        base_velocity = np.random.uniform(-15, 15)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.08, 0.15),
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=20.0,
            min_velocity_mps=-20.0,
            current_velocity_mps=base_velocity
        )
    
    elif target_type == TargetType.PATROL_BOAT:
        base_velocity = np.random.uniform(-18, 18)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.04, 0.10),
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=25.0,
            min_velocity_mps=-25.0,
            current_velocity_mps=base_velocity
        )
    
    elif target_type == TargetType.SMALL_CRAFT:
        base_velocity = np.random.uniform(-10, 10)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.01, 0.05),
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=15.0,
            min_velocity_mps=-15.0,
            current_velocity_mps=base_velocity
        )
    
    elif target_type == TargetType.SAILBOAT:
        base_velocity = np.random.uniform(-6, 6)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.015, 0.06),
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=12.0,
            min_velocity_mps=-12.0,
            current_velocity_mps=base_velocity
        )
    
    elif target_type == TargetType.SPEEDBOAT:
        base_velocity = np.random.uniform(-25, 25)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.03, 0.08),
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=35.0,
            min_velocity_mps=-35.0,
            current_velocity_mps=base_velocity
        )
    
    else:
        # Default case
        base_velocity = np.random.uniform(-20, 20)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=60, #np.random.uniform(0.01,0.08),
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            current_velocity_mps=base_velocity
        )