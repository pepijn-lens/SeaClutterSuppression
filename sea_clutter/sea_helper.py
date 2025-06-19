from __future__ import annotations

from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .parameters import RadarParams, RealisticTarget, ClutterParams, TargetType
from .physics import db


# ────────────────────────────────────────────────────────────────────────────────
# Animation helper
# ────────────────────────────────────────────────────────────────────────────────
def animate_sequence(
    rdm_list: List[np.ndarray],
    rp: RadarParams,
    *,
    interval_ms: int = 1000,
    save_path: Optional[str] = None,
):
    fd = np.fft.fftshift(np.fft.fftfreq(rp.n_pulses, d=1.0 / rp.prf))
    # Convert Doppler frequency to velocity (m/s)
    velocity = fd * rp.carrier_wavelength / 2.0
    rng = np.arange(rp.n_ranges) * rp.range_resolution
    imgs_db = [db(np.abs(rd) ** 2) for rd in rdm_list]
    # print(f"Average noise power in dB: {np.mean([np.mean(img[:, 0]) for img in imgs_db])}")
    vmax = np.max(imgs_db); vmin = np.min(imgs_db)
    
    # Single plot if no masks provided
    fig, ax = plt.subplots(figsize=(6, 5))
    im1 = ax.imshow(imgs_db[0], aspect="auto", cmap="viridis",
                    extent=[velocity[0], velocity[-1], rng[-1], rng[0]],
                    vmin=vmin, vmax=vmax, 
                    interpolation="nearest")
    ax.set_xlabel("Radial Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    title_text = ax.set_title("Frame 0")
    plt.tight_layout()
    
    def update(frame_idx):
        im1.set_data(imgs_db[frame_idx])
        title_text.set_text(f"Frame {frame_idx}")
        return im1, title_text
    
    ani = animation.FuncAnimation(fig, update, frames=len(imgs_db),
                                  interval=interval_ms, blit=False, repeat=True)
    if save_path is not None:
        writer = "pillow" if save_path.endswith(".gif") else None
        ani.save(save_path, dpi=80, writer=writer)
    plt.show()
    return ani


def update_realistic_target_velocity(tgt: RealisticTarget, rp: RadarParams):
    """Update target velocity and Doppler frequency for a target.

    The previous implementation used a hard coded carrier wavelength of
    0.03 m when converting the updated velocity to Doppler frequency.
    This produced incorrect Doppler values whenever the radar parameters
    specified a different wavelength.  The function now requires the radar
    parameters so that the correct wavelength is used.
    """
    
    # Standard velocity noise - same for all targets and sea states
    velocity_noise_std = 0.1  # Fixed standard deviation for all targets
    velocity_noise = np.random.normal(0, velocity_noise_std)
    
    # Update velocity trend (persistent component) - also standardized
    trend_drift_rate = 0.1  # Fixed drift rate for all targets
    trend_noise = np.random.normal(0, trend_drift_rate)
    tgt.velocity_trend += trend_noise
    
    # Prevent trend from getting too extreme
    tgt.velocity_trend = np.clip(tgt.velocity_trend, -3.0, 3.0)
    
    # Combine velocity components - simplified, consistent approach
    tgt.current_velocity_mps += velocity_noise + tgt.velocity_trend * 0.1
    
    # Apply velocity limits (these can still be target-specific for realism)
    tgt.current_velocity_mps = np.clip(
        tgt.current_velocity_mps, 
        tgt.min_velocity_mps, 
        tgt.max_velocity_mps
    )
    
    # Update Doppler frequency based on new velocity and radar wavelength
    tgt.doppler_hz = 2.0 * tgt.current_velocity_mps / rp.carrier_wavelength

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
            'mean_power_db': -5.0,      # Very low clutter power
            'shape_param': 0.8,          # More Gaussian-like (less spiky)
            'ar_coeff': 0.998,           # Very stable, slow decorrelation
            'bragg_offset_hz': 15.0,     # Weak Bragg lines
            'bragg_width_hz': 1.0,       # Narrow Bragg peaks
            'bragg_power_rel': 3.0,      # Weak Bragg enhancement
            'wave_speed_mps': 1.0        # Slow wave movement
        },
        3: {  # Slight seas
            'mean_power_db': -1.0,      # Low clutter power
            'shape_param': 0.5,          # Moderate non-Gaussianity
            'ar_coeff': 0.995,           # High correlation
            'bragg_offset_hz': 20.0,     # Moderate Bragg lines
            'bragg_width_hz': 1.5,       # Moderate Bragg width
            'bragg_power_rel': 4.0,      # Moderate Bragg enhancement
            'wave_speed_mps': 2.0        # Moderate wave movement
        },
        5: {  # Moderate seas
            'mean_power_db': -12.0,      # Moderate clutter power
            'shape_param': 0.3,          # Significant non-Gaussianity
            'ar_coeff': 0.9,           # Moderate correlation
            'bragg_offset_hz': None,     # Strong Bragg lines
            'bragg_width_hz': 2.0,       # Wider Bragg peaks
            'bragg_power_rel': 4.0,      # Strong Bragg enhancement
            'wave_speed_mps': 4.0        # Moderate-fast wave movement
        },
        7: {  # Rough seas
            'mean_power_db': 7,      # High clutter power
            'shape_param': 0.30,         # Strong non-Gaussianity (spiky)
            'ar_coeff': 0.975,           # Lower correlation (faster decorrelation)
            'bragg_offset_hz': 40.0,     # Very strong Bragg lines
            'bragg_width_hz': 2.5,       # Broad Bragg peaks
            'bragg_power_rel': 4.0,      # Very strong Bragg enhancement
            'wave_speed_mps': 4.5        # Fast wave movement
        },
        9: {  # Very rough seas
            'mean_power_db': 11,      # Very high clutter power
            'shape_param': 0.25,         # Extreme non-Gaussianity (very spiky)
            'ar_coeff': 0.95,           # Low correlation (rapid decorrelation)
            'bragg_offset_hz': 50.0,     # Extreme Bragg lines
            'bragg_width_hz': 3.0,       # Very broad Bragg peaks
            'bragg_power_rel': 5.0,      # Extreme Bragg enhancement
            'wave_speed_mps': 5.0        # Very fast wave movement
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
    if target_type == TargetType.CARGO_SHIP:
        base_velocity = np.random.uniform(-15, 15)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.08, 0.15),
            target_type=target_type,
            size=3,  # Cargo ships are larger - 3 pixels in range
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=20.0,
            min_velocity_mps=-20.0,
            current_velocity_mps=base_velocity
        )

    elif target_type == TargetType.SPEEDBOAT:
        base_velocity = np.random.uniform(-25, 25)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=np.random.uniform(0.03, 0.08),
            target_type=target_type,
            size=1,  # Speedboats are small - 1 pixel in range
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=35.0,
            min_velocity_mps=-35.0,
            current_velocity_mps=base_velocity
        )
    
    elif target_type == TargetType.FIXED:
        # Default case
        base_velocity = np.random.uniform(-20, 20)
        return RealisticTarget(
            rng_idx=initial_range_idx,
            doppler_hz=2.0 * base_velocity / rp.carrier_wavelength,
            power=20,
            target_type=target_type,
            size=1,  # Default size for fixed targets
            base_velocity_mps=base_velocity,
            velocity_noise_std=standard_velocity_noise_std,    # Standardized
            max_velocity_mps=36.0,
            min_velocity_mps=-36.0,
            current_velocity_mps=base_velocity
        )