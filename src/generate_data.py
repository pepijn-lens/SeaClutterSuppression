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


def generate_large_dataset(
    thermal_noise_db: float = 1
):
    """Generate a large dataset with 25000 samples, 3 frames per sample, max 10 targets."""
    import random
    import torch
    
    # Dataset configuration
    n_samples = 25000  # Increased to ensure samples per class
    n_frames = 3
    max_targets = 10
    samples_per_class = max(1, n_samples // (max_targets + 1))  # Ensure at least 1 sample per class
    
    # Default radar parameters
    base_rp = sea_clutter.RadarParams()
    
    # Target type: speedboat
    target_type = sea_clutter.TargetType.SPEEDBOAT
    
    # Set sequence parameters
    sp = sea_clutter.SequenceParams()
    sp.n_frames = n_frames
    
    all_sequences = []
    all_mask_sequences = []
    all_labels = []
    
    # Generate data for each class (0 to max_targets)
    for n_targets in range(max_targets + 1):
        for sample_idx in range(samples_per_class):
            # Random clutter parameters for this sample
            clutter_params = sea_clutter.ClutterParams(
                mean_power_db=random.uniform(14.0, 18.0),
                shape_param=random.uniform(0.3, 0.99),
                ar_coeff=random.uniform(0.5, 0.99),
                wave_speed_mps=random.uniform(-6.0, 6.0),
                bragg_power_rel=0
            )
            
            # Generate targets for this sequence with random powers
            targets = []
            if n_targets > 0:
                for _ in range(n_targets):
                    # Random target power between 10-20 dB for each target
                    target_power = random.uniform(10.0, 21.0)
                    
                    target = sea_clutter.create_realistic_target(
                        target_type, 
                        random.randint(30, base_rp.n_ranges - 30), 
                        base_rp
                    )
                    # Override the target power with our random value
                    target.power = target_power
                    targets.append(target)
            
            # Generate sequence of RDMs and masks
            rdm_list, mask_list = simulate_sequence_with_realistic_targets_and_masks(
                base_rp, clutter_params, sp, targets, 
                thermal_noise_db=thermal_noise_db
            )
            
            # Process each frame in the sequence
            processed_sequence = []
            processed_mask_sequence = []
            
            for rdm, mask in zip(rdm_list, mask_list):
                # Convert to dB scale
                rdm_db = 20 * np.log10(np.abs(rdm) + 1e-10)
                processed_sequence.append(rdm_db)
                processed_mask_sequence.append(mask)
            
            # Stack frames into sequence arrays
            sequence = np.stack(processed_sequence, axis=0)
            mask_sequence = np.stack(processed_mask_sequence, axis=0)
            
            # Store sequence and label
            all_sequences.append(sequence)
            all_mask_sequences.append(mask_sequence)
            all_labels.append(n_targets)
    
    # Convert to numpy arrays
    sequences = np.array(all_sequences)
    mask_sequences = np.array(all_mask_sequences)
    labels = np.array(all_labels)
    
    # Convert to PyTorch tensors
    sequences_tensor = torch.from_numpy(sequences).float()
    mask_sequences_tensor = torch.from_numpy(mask_sequences).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset dictionary
    dataset = {
        'sequences': sequences_tensor,
        'masks': mask_sequences_tensor,
        'labels': labels_tensor,
        'metadata': {
            'n_samples': n_samples,
            'samples_per_class': samples_per_class,
            'max_targets': max_targets,
            'target_type': target_type.name,
            'n_frames': n_frames,
            'n_ranges': base_rp.n_ranges,
            'n_doppler_bins': base_rp.n_pulses,
            'target_power_range_db': [10.0, 20.0],
            'clutter_param_ranges': {
                'mean_power_db': [14.0, 18.0],  # Fixed to match actual range used
                'shape_param': [0.3, 0.99],
                'ar_coeff': [0.5, 0.99],
                'wave_speed_mps': [-6.0, 6.0]
            },
            'class_names': [f"{i}_targets" for i in range(max_targets + 1)],
            'data_format': 'dB_scale'
        }
    }
    
    # Calculate dataset size in GB
    total_bytes = (
        dataset['sequences'].element_size() * dataset['sequences'].nelement() +
        dataset['masks'].element_size() * dataset['masks'].nelement() +
        dataset['labels'].element_size() * dataset['labels'].nelement()
    )
    dataset_size_gb = total_bytes / (1024**3)
    
    return dataset_size_gb, dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import os
    
    print("Generating large dataset...")
    dataset_size_gb, dataset = generate_large_dataset()
    
    print(f"Dataset generated successfully!")
    print(f"Dataset size: {dataset_size_gb:.2f} GB")
    print(f"Total samples: {len(dataset['sequences'])}")
    print(f"Samples per class: {dataset['metadata']['samples_per_class']}")
    print(f"Number of classes: {dataset['metadata']['max_targets'] + 1}")
    
    # Create data directory if it doesn't exist
    data_dir = "/Users/pepijnlens/Documents/SeaClutterSuppression/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save dataset
    save_path = os.path.join(data_dir, "random.pt")
    print(f"\nSaving dataset to {save_path}...")
    torch.save(dataset, save_path)
    print("Dataset saved successfully!")
    
    # Show 10 example plots
    print("\nGenerating example plots...")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Dataset Examples: RDMs (top) and Target Masks (bottom)', fontsize=16)
    
    # Select examples from different classes
    examples_per_class = max(1, 10 // (dataset['metadata']['max_targets'] + 1))
    plot_idx = 0
    
    for n_targets in range(min(dataset['metadata']['max_targets'] + 1, 10)):
        # Find samples with this number of targets
        target_indices = (dataset['labels'] == n_targets).nonzero(as_tuple=True)[0]
        
        if len(target_indices) > 0:
            # Take first sample from this class
            sample_idx = target_indices[0].item()
            
            # Get the first frame of the sequence
            rdm = dataset['sequences'][sample_idx, 0].numpy()  # First frame
            mask = dataset['masks'][sample_idx, 0].numpy()     # First frame
            
            # Plot RDM
            row = (plot_idx // 5) * 2
            col = plot_idx % 5
            
            im1 = axes[row, col].imshow(rdm, aspect='auto', cmap='viridis', origin='lower')
            axes[row, col].set_title(f'{n_targets} targets - RDM')
            axes[row, col].set_xlabel('Doppler Bin')
            axes[row, col].set_ylabel('Range Bin')
            plt.colorbar(im1, ax=axes[row, col], shrink=0.8)
            
            # Plot mask
            im2 = axes[row + 1, col].imshow(mask, aspect='auto', cmap='Reds', origin='lower', vmin=0, vmax=1)
            axes[row + 1, col].set_title(f'{n_targets} targets - Mask')
            axes[row + 1, col].set_xlabel('Doppler Bin')
            axes[row + 1, col].set_ylabel('Range Bin')
            plt.colorbar(im2, ax=axes[row + 1, col], shrink=0.8)
            
            plot_idx += 1
            
            if plot_idx >= 5:  # Only show first 5 classes
                break
    
    # Hide unused subplots
    for i in range(plot_idx, 5):
        axes[0, i].set_visible(False)
        axes[1, i].set_visible(False)
        if len(axes) > 2:
            axes[2, i].set_visible(False)
            axes[3, i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Sequence shape: {dataset['sequences'].shape}")
    print(f"Mask shape: {dataset['masks'].shape}")
    print(f"Labels shape: {dataset['labels'].shape}")
    
    # Check if dataset is not empty before computing min/max
    if dataset['sequences'].numel() > 0:
        print(f"RDM value range: [{dataset['sequences'].min():.2f}, {dataset['sequences'].max():.2f}] dB")
    else:
        print("Dataset is empty - no samples generated")
    
    # Class distribution
    if dataset['labels'].numel() > 0:
        unique_labels, counts = dataset['labels'].unique(return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label} targets: {count} samples")
    else:
        print("No labels to show - dataset is empty")