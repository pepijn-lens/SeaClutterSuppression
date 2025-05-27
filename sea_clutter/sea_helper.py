from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Parameters import RadarParams, RealisticTarget
from rd_map import db


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
    vmax = max(map(np.max, imgs_db)); vmin = vmax - 50.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(imgs_db[0], aspect="auto", cmap="viridis",
                   extent=[velocity[0], velocity[-1], rng[-1], rng[0]],
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Radial Velocity (m/s)")  # Updated label
    ax.set_ylabel("Range (m)")
    title_text = ax.set_title("Frame 0"); plt.tight_layout()
    def update(frame_idx):
        im.set_data(imgs_db[frame_idx])
        title_text.set_text(f"Frame {frame_idx}")
        return im, title_text
    ani = animation.FuncAnimation(fig, update, frames=len(imgs_db),
                                  interval=interval_ms, blit=False, repeat=True)
    if save_path is not None:
        writer = "pillow" if save_path.endswith(".gif") else None
        ani.save(save_path, dpi=80, writer=writer)
    plt.show(); return ani

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

def create_target_mask(
    targets: List[RealisticTarget], 
    rp: RadarParams, 
    mask_radius_range: int = 2,
    mask_radius_doppler: int = 3
) -> np.ndarray:
    """Create a binary mask where targets are present."""
    mask = np.zeros((rp.n_ranges, rp.n_pulses), dtype=bool)
    
    # Convert Doppler frequencies to bin indices
    fd = np.fft.fftshift(np.fft.fftfreq(rp.n_pulses, d=1.0 / rp.prf))
    
    for tgt in targets:
        # Find closest Doppler bin
        doppler_bin = np.argmin(np.abs(fd - tgt.doppler_hz))
        
        # Create mask region around target
        range_start = max(0, tgt.rng_idx - mask_radius_range)
        range_end = min(rp.n_ranges, tgt.rng_idx + mask_radius_range + 1)
        doppler_start = max(0, doppler_bin - mask_radius_doppler)
        doppler_end = min(rp.n_pulses, doppler_bin + mask_radius_doppler + 1)
        
        mask[range_start:range_end, doppler_start:doppler_end] = True
    
    return mask

def animate_targets_with_masks(
    rdm_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    target_tracks: Dict[str, List],
    rp: RadarParams,
    *,
    interval_ms: int = 1000,
    save_path: Optional[str] = None,
):
    """Animate sequence showing both full RDM and target-only mask."""
    fd = np.fft.fftshift(np.fft.fftfreq(rp.n_pulses, d=1.0 / rp.prf))
    velocity = fd * rp.carrier_wavelength / 2.0
    rng = np.arange(rp.n_ranges) * rp.range_resolution
    
    # Prepare data for visualization
    imgs_db = [db(np.abs(rd) ** 2) for rd in rdm_list]
    masked_imgs = [img * mask for img, mask in zip(imgs_db, mask_list)]
    
    vmax = max(map(np.max, imgs_db))
    vmin = vmax - 50.0
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Full RDM
    im1 = ax1.imshow(imgs_db[0], aspect="auto", cmap="viridis",
                     extent=[velocity[0], velocity[-1], rng[-1], rng[0]],
                     vmin=vmin, vmax=vmax, interpolation="nearest")
    ax1.set_xlabel("Radial Velocity (m/s)")
    ax1.set_ylabel("Range (m)")
    ax1.set_title("Full Range-Doppler Map")
    
    # Target mask only
    im2 = ax2.imshow(masked_imgs[0], aspect="auto", cmap="viridis",
                     extent=[velocity[0], velocity[-1], rng[-1], rng[0]],
                     vmin=vmin, vmax=vmax, interpolation="nearest")
    ax2.set_xlabel("Radial Velocity (m/s)")
    ax2.set_ylabel("Range (m)")
    ax2.set_title("Targets Only (Masked)")
    
    plt.tight_layout()
    
    # Add frame counter
    frame_text = fig.suptitle("Frame 0")
    
    def update(frame_idx):
        im1.set_data(imgs_db[frame_idx])
        im2.set_data(masked_imgs[frame_idx])
        
        # Update title with target info
        n_targets = len(target_tracks['target_types'])
        ranges = target_tracks['target_ranges'][frame_idx]
        velocities = target_tracks['target_velocities'][frame_idx]
        
        info_str = f"Frame {frame_idx} | "
        for i in range(min(3, n_targets)):  # Show first 3 targets
            info_str += f"T{i+1}: {ranges[i]:.0f}m, {velocities[i]:.1f}m/s | "
        
        frame_text.set_text(info_str)
        return im1, im2, frame_text
    
    ani = animation.FuncAnimation(fig, update, frames=len(imgs_db),
                                  interval=interval_ms, blit=False, repeat=True)
    
    if save_path is not None:
        writer = "pillow" if save_path.endswith(".gif") else None
        ani.save(save_path, dpi=80, writer=writer)
    
    plt.show()
    return ani

def print_target_tracks(target_tracks: Dict[str, List]):
    """Print summary of target tracking data."""
    n_frames = len(target_tracks['frame_idx'])
    n_targets = len(target_tracks['target_types'])
    
    print(f"\nTarget Tracking Summary ({n_frames} frames, {n_targets} targets):")
    print("=" * 60)
    
    for target_idx in range(n_targets):
        target_type = target_tracks['target_types'][target_idx]
        print(f"\nTarget {target_idx + 1} ({target_type}):")
        
        initial_range = target_tracks['target_ranges'][0][target_idx]
        final_range = target_tracks['target_ranges'][-1][target_idx]
        initial_vel = target_tracks['target_velocities'][0][target_idx]
        final_vel = target_tracks['target_velocities'][-1][target_idx]
        
        print(f"  Range: {initial_range:.1f}m → {final_range:.1f}m")
        print(f"  Velocity: {initial_vel:.1f}m/s → {final_vel:.1f}m/s")
        
        # Calculate velocity statistics
        all_velocities = [target_tracks['target_velocities'][f][target_idx] 
                         for f in range(n_frames)]
        vel_std = np.std(all_velocities)
        print(f"  Velocity std: {vel_std:.2f}m/s")