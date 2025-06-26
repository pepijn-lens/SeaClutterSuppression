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
    mean_power_db: float = 15.0        # Increase from 0.0 → much stronger clutter
    shape_param: float = 0.3           # Increase from 0.01 → more spiky/impulsive
    ar_coeff: float = 0.85             # Decrease from 0.98 → faster decorrelation
    bragg_offset_hz: Optional[float] = 45.0  # Increase from 25.0 → stronger Bragg lines
    bragg_width_hz: float = 4.0        # Increase from 2.0 → broader Bragg peaks
    bragg_power_rel: float = 8.0       # Increase from 5.0 → stronger Bragg enhancement
    wave_speed_mps: float = 6.0        # Increase from 2.0 → faster wave motion


@dataclass
class Target:
    rng_idx: int = 30
    doppler_hz: float = 2000
    power: float = 20.0             # Linear power for the central cell
    size: int = 1                   # Target size in pixels (range bins)

@dataclass
class SequenceParams:
    n_frames: int = 6             # Frames to simulate
    frame_rate_hz: float = 2      # Frames per second

class TargetType(Enum):
    CARGO_SHIP = "cargo"           # Fast, steady course, less affected by waves   
    SPEEDBOAT = "speedboat"        # Fast, agile, sudden speed changes

@dataclass
class RealisticTarget:
    rng_idx: int
    doppler_hz: float
    power: float = 20.0
    target_type: TargetType = TargetType.SPEEDBOAT
    size: int = 1                       # Target size in pixels (range bins)
    
    # Movement parameters
    base_velocity_mps: float = 5.0      # Base radial velocity [m/s]
    velocity_noise_std: float = 0.5     # Standard deviation of velocity noise
    max_velocity_mps: float = 36.0      # Maximum possible velocity
    min_velocity_mps: float = -36.0     # Minimum possible velocity (negative = approaching)
    
    # Internal state for realistic movement
    current_velocity_mps: float = 0.0   # Current radial velocity
    velocity_trend: float = 0.0         # Persistent velocity trend
