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
    n_ranges: int = 128           # Range bins
    range_resolution: float = 1  # [m]
    carrier_wavelength: float = 0.03  # [m] – unused but kept for completeness

@dataclass
class ClutterParams:
    mean_power_db: float = -20.0    # Average clutter power per cell [dB]
    shape_param: float = 0.01        # Gamma shape κ; smaller → heavier tail
    ar_coeff: float = 0.98          # Slow-time AR(1) coefficient
    bragg_offset_hz: Optional[float] = 25.0  # Bragg Doppler [Hz]
    bragg_width_hz: float = 2.0     # Bragg peak width [Hz]
    bragg_power_rel: float = 5.0    # Bragg peak height over background [dB]

@dataclass
class Target:
    rng_idx: int
    doppler_hz: float
    power: float = 10.0             # Linear power for the central cell

@dataclass
class SequenceParams:
    n_frames: int = 25            # Frames to simulate
    frame_rate_hz: float = 2.0      # Frames per second
    wave_speed_mps: float = 3.0     # Wave propagation speed toward radar [m/s]

class TargetType(Enum):
    FISHING_VESSEL = "fishing"      # Slow, meandering, affected by waves
    CARGO_SHIP = "cargo"           # Fast, steady course, less affected by waves  
    PATROL_BOAT = "patrol"         # Variable speed, frequent course changes
    SMALL_CRAFT = "small"          # Very affected by waves, erratic movement
    SAILBOAT = "sailboat"          # Wind-dependent, gentle movements
    SPEEDBOAT = "speedboat"        # Fast, agile, sudden speed changes

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
    """Map WMO sea states (1, ..., 9) to model parameters."""
    configs = {
        1: {'mean_power_db': -30.0, 'shape_param': 0.5, 'ar_coeff': 0.995},
        3: {'mean_power_db': -25.0, 'shape_param': 0.4, 'ar_coeff': 0.992},
        5: {'mean_power_db': -20.0, 'shape_param': 0.3, 'ar_coeff': 0.990},
        7: {'mean_power_db': -15.0, 'shape_param': 0.2, 'ar_coeff': 0.960},
        9: {'mean_power_db': -10.0, 'shape_param': 0.1, 'ar_coeff': 0.950},
    }
    if state not in configs:
        raise ValueError(f"Unsupported sea state {state}; choose from {list(configs)}")
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
            power=np.random.uniform(0.02, 0.08),
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
        base_velocity = np.random.uniform(-10, 10)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=0.05,
            target_type=target_type,
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            current_velocity_mps=base_velocity
        )