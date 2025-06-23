from typing import List
import numpy as np
import sea_clutter

def simulate_sequence_with_realistic_targets_and_masks(
    rp: sea_clutter.RadarParams,
    cp: sea_clutter.ClutterParams,
    sp: sea_clutter.SequenceParams,
    targets: List[sea_clutter.RealisticTarget],
    *,
    thermal_noise_db: float = 1,
    target_signal_power: float = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:  # Return RDMs and masks
    """Simulate sequence with multiple realistic targets and generate corresponding masks."""
    dt = 1.0 / sp.frame_rate_hz
    texture = None
    speckle_tail = None
    rdm_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    for frame_idx in range(sp.n_frames):
        if texture is not None and cp.wave_speed_mps != 0:
            shift_bins = int(round(cp.wave_speed_mps * dt / rp.range_resolution))
            texture = np.roll(texture, shift=shift_bins, axis=0)
            
        clutter_td, texture, speckle_tail = sea_clutter.simulate_sea_clutter(
            rp, cp, texture=texture, init_speckle=speckle_tail, thermal_noise_db=thermal_noise_db
        )
        
        # Initialize binary mask for this frame
        target_mask = np.zeros((rp.n_ranges, rp.n_pulses), dtype=np.float32)
        
        # Update and add each target
        for tgt in targets:
            # Update target velocity with realistic variations
            sea_clutter.update_realistic_target_velocity(tgt, rp)
            
            # Override target power with user-controlled value if provided
            target_power = target_signal_power if target_signal_power is not None else tgt.power
            
            # Convert to Target object for add_target_blob function
            simple_target = sea_clutter.Target(
                rng_idx=tgt.rng_idx,
                doppler_hz=tgt.doppler_hz,
                power=target_power,
                size=getattr(tgt, 'size', 1)  # Use target size from realistic target
            )
            
            # Add target to clutter data
            sea_clutter.add_target_blob(clutter_td, simple_target, rp)
            
            # Mark target location in binary mask
            # Convert Doppler frequency to bin index
            # Use proper FFT bin indexing - center frequency is at n_pulses//2 after fftshift
            doppler_bin = int(tgt.doppler_hz / (rp.prf / rp.n_pulses)) + rp.n_pulses // 2
            doppler_bin = np.clip(doppler_bin, 0, rp.n_pulses - 1)
            
            # Mark target in mask with blob size based on target type
            target_size = getattr(tgt, 'size', 1)
            range_half_size = (target_size - 1) // 2
            doppler_blob_size = 1  # Keep Doppler spread consistent
            
            # Range extent
            r_start = max(0, tgt.rng_idx - range_half_size)
            r_end = min(rp.n_ranges, tgt.rng_idx + range_half_size + 1)
            
            # Doppler extent
            d_start = max(0, doppler_bin - doppler_blob_size - 1)
            d_end = min(rp.n_pulses, doppler_bin + doppler_blob_size + 1)
            
            target_mask[r_start:r_end, d_start:d_end] = 1.0
            
            # Update target range based on radial velocity
            range_change = tgt.current_velocity_mps * dt
            new_range = tgt.rng_idx * rp.range_resolution + range_change
            tgt.rng_idx = int(np.clip(new_range / rp.range_resolution, 0, rp.n_ranges - 1))
        
        # Compute RD map
        rdm = sea_clutter.compute_range_doppler(clutter_td, rp, cp)

        rdm_list.append(rdm)
        mask_list.append(target_mask)
    
    return rdm_list, mask_list


if __name__ == "__main__":
    n_targets = 5  # Number of targets to generate
    gen_target_type = sea_clutter.TargetType.SPEEDBOAT  # Type of target to generate
    import random
    gen_rp = sea_clutter.RadarParams()  # Radar parameters
    gen_cp = sea_clutter.ClutterParams()
    gen_cp.mean_power_db = 10.0  # Set clutter power
    gen_cp.bragg_power_rel = 0
    gen_sp = sea_clutter.SequenceParams()  # Sequence parameters

    # Generate targets for this sequence
    gen_targets = []
    if n_targets > 0:
        for _ in range(n_targets):
            gen_target = sea_clutter.create_realistic_target(
                gen_target_type, 
                random.randint(30, gen_rp.n_ranges - 30), 
                gen_rp
            )
            gen_targets.append(gen_target)

    # Generate sequence of RDMs and masks with SNR controls
    gen_rdm_list, gen_mask_list = simulate_sequence_with_realistic_targets_and_masks(
        gen_rp, gen_cp, gen_sp, gen_targets, 
        thermal_noise_db=1, target_signal_power=20
    )

    # Process each frame in the sequence
    gen_processed_sequence = []
    gen_processed_mask_sequence = []

    for gen_rdm, gen_mask in zip(gen_rdm_list, gen_mask_list):
        # Convert to dB scale only (no normalization at storage time)
        # Normalization will be applied during data loading for training/evaluation
        gen_rdm_db = 20 * np.log10(np.abs(gen_rdm) + 1e-10)
        gen_processed_sequence.append(gen_rdm_db)
        gen_processed_mask_sequence.append(gen_mask)

    # Stack frames into sequence arrays
    gen_sequence = np.stack(gen_processed_sequence, axis=0)
    gen_mask_sequence = np.stack(gen_processed_mask_sequence, axis=0)

    # Plot only the first frame of the generated sequence and mask
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gen_sequence[0], cmap='viridis', vmin=0, vmax=45)
    plt.title('RDM Frame 1')
    plt.colorbar(label='Power (dB)')

    plt.subplot(1, 2, 2)
    plt.imshow(gen_mask_sequence[0], cmap='gray')
    plt.title('Mask Frame 1')

    plt.tight_layout()
    plt.show()