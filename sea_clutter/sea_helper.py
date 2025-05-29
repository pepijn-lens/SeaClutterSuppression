from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Parameters import RadarParams, RealisticTarget
from rd_map import db

import torch

# ────────────────────────────────────────────────────────────────────────────────
# Animation helper
# ────────────────────────────────────────────────────────────────────────────────
def animate_sequence(
    rdm_list: List[np.ndarray],
    target_mask_list: Optional[List[np.ndarray]],
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
    vmax = np.max(imgs_db); vmin = np.min(imgs_db)
    
    # Create subplot layout based on whether masks are provided
    if target_mask_list is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot RD map on left
        im1 = ax1.imshow(imgs_db[0], aspect="auto", cmap="viridis",
                        extent=[velocity[0], velocity[-1], rng[-1], rng[0]],
                        vmin=vmin, vmax=vmax, 
                        interpolation="nearest")
        ax1.set_xlabel("Radial Velocity (m/s)")
        ax1.set_ylabel("Range (m)")
        ax1.set_title("Range-Doppler Map")
        
        # Plot target mask on right
        mask_rd = convert_mask_to_rd(target_mask_list[0], rp)
        im2 = ax2.imshow(mask_rd, aspect="auto", cmap="Reds",
                        extent=[velocity[0], velocity[-1], rng[-1], rng[0]],
                        vmin=0, vmax=1,
                        interpolation="nearest")
        ax2.set_xlabel("Radial Velocity (m/s)")
        ax2.set_ylabel("Range (m)")
        ax2.set_title("Target Mask")
        
        title_text = fig.suptitle("Frame 0")
        plt.tight_layout()
        
        def update(frame_idx):
            im1.set_data(imgs_db[frame_idx])
            mask_rd = convert_mask_to_rd(target_mask_list[frame_idx], rp)
            im2.set_data(mask_rd)
            title_text.set_text(f"Frame {frame_idx}")
            return im1, im2, title_text
            
    else:
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

def convert_mask_to_rd(mask_td: np.ndarray, rp: RadarParams) -> np.ndarray:
    """Convert time-domain mask to range-Doppler domain for visualization."""
    # Apply same windowing and FFT as compute_range_doppler
    window = np.hamming(rp.n_pulses)
    mask_windowed = mask_td.astype(float) * window[np.newaxis, :]
    mask_rd = np.abs(np.fft.fftshift(np.fft.fft(mask_windowed, axis=1), axes=1)) > 0.1
    return mask_rd.astype(float)

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

def plot_sea_clutter_sequence():
    import sys
    import os
    # Add parent directory to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Classification import RadarDataset
    
    # Load the dataset
    data_path = "data/sea_clutter-4_frames.pt"  # Adjust path as needed
    dataset = RadarDataset(data_path)
    
    # Get a sample
    sample_idx = 7000  # You can change this to visualize different samples
    sequence, label = dataset[sample_idx]
    
    # Convert from tensor to numpy if needed
    if torch.is_tensor(sequence):
        sequence_np = sequence.numpy()
    else:
        sequence_np = sequence
    
    print(f"Sample {sample_idx}: Label = {label}")
    print(f"Sequence shape: {sequence_np.shape}")
    
    # # For single images, just display the image
    # fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.imshow(sequence_np.squeeze(0), aspect="auto", cmap="viridis")
    # ax.set_title(f"Radar Image - Label: {label}")
    # ax.set_xlabel("Doppler Bins")
    # ax.set_ylabel("Range Bins")
    # plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    # plt.tight_layout()
    # plt.show()

    if dataset.is_sequence:
        # For sequences, create a list of frames for animation
        rdm_list = [sequence_np[i] for i in range(sequence_np.shape[0])]
        
        # Get radar parameters from metadata if available
        metadata = dataset.get_metadata()
        if 'radar_params' in metadata:
            rp = metadata['radar_params']
        else:
            # Create default radar parameters for visualization
            from Parameters import RadarParams
            rp = RadarParams()
        
        # Animate the sequence
        print(f"Animating {len(rdm_list)} frames...")
        ani = animate_sequence(
            rdm_list=rdm_list,
            target_mask_list=None,  # No masks available from dataset
            rp=rp,
            interval_ms=500,  # 500ms between frames
            save_path=None  # Set to a path if you want to save
        )
    else:
        # For single images, just display the image
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(sequence_np[0], aspect="auto", cmap="viridis", vmin=0, vmax=60,)
        ax.set_title(f"Radar Image - Label: {label}")
        ax.set_xlabel("Doppler Bins")
        ax.set_ylabel("Range Bins")
        plt.colorbar(im, ax=ax, label="Magnitude (dB)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_sea_clutter_sequence()