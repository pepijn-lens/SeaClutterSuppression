from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from parameters import RadarParams, RealisticTarget
from physics import db


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


def update_realistic_target_velocity(tgt: RealisticTarget):
    """Update target velocity with consistent randomness across all targets and states."""
    
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
    
    # Update Doppler frequency based on new velocity
    tgt.doppler_hz = 2.0 * tgt.current_velocity_mps / 0.03  # Using default wavelength
