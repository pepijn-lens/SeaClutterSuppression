from typing import List
import torch
import numpy as np
import random
from tqdm import tqdm
import sea_clutter

MIN_RANGE = 30  # Minimum range for targets
MAX_RANGE = 128 - 30  # Maximum range for targets

def generate_single_frame_with_targets(
    rp: sea_clutter.RadarParams,
    cp: sea_clutter.ClutterParams,
    n_targets: int,
) -> np.ndarray:
    """Generate a single range-Doppler map with specified number of targets."""
    
    # Generate sea clutter
    clutter_td, _, _ = sea_clutter.simulate_sea_clutter(rp, cp)
    
    # Generate random targets if needed
    if n_targets > 0:
        max_range = rp.n_ranges * rp.range_resolution
        targets = [
            sea_clutter.create_realistic_target(sea_clutter.TargetType.FIXED, random.randint(0, max_range-1), rp) 
            for _ in range(n_targets)
        ]
        
        # Add each target to the clutter data
        for tgt in targets:
            simple_target = sea_clutter.Target(
                rng_idx=tgt.rng_idx,
                doppler_hz=tgt.doppler_hz,
                power=tgt.power
            )
            sea_clutter.add_target_blob(clutter_td, simple_target, rp)
    
    # Compute range-Doppler map
    rdm = sea_clutter.compute_range_doppler(clutter_td, rp, cp)
    
    return rdm

def generate_classification_dataset(
    samples_per_class: int = 2000,
    max_targets: int = 5,
    sea_state: int = 5,
    save_path: str = "sea_clutter_classification_dataset.pt"
) -> None:
    """
    Generate dataset for target count classification.
    
    Args:
        samples_per_class: Number of samples to generate per class
        max_targets: Maximum number of targets (classes will be 0 to max_targets)
        sea_state: WMO sea state to use
        save_path: Path to save the PyTorch file
    """
    
    # Initialize parameters
    rp = sea_clutter.RadarParams()
    cp = sea_clutter.get_clutter_params_for_sea_state(sea_state)
    
    # Set single frame
    sp = sea_clutter.SequenceParams()
    
    print(f"Generating dataset with {samples_per_class} samples per class")
    print(f"Classes: 0 to {max_targets} targets")
    print(f"Sea state: {sea_state}")
    print(f"Range-Doppler map size: {rp.n_ranges} x {rp.n_pulses}")
    
    # Storage for data and labels
    all_images = []
    all_labels = []
    
    # Generate data for each class
    for n_targets in range(max_targets + 1):
        print(f"\nGenerating {samples_per_class} samples for {n_targets} targets...")
        
        class_images = []
        class_labels = []
        
        for i in tqdm(range(samples_per_class), desc=f"Class {n_targets}"):
            # Generate single RDM
            rdm = generate_single_frame_with_targets(rp, cp, n_targets)

            # to dB scale and normalize with mean and std
            rdm = 20 * np.log10(np.abs(rdm) + 1e-10)  # Avoid log(0)
            rdm = (rdm - np.mean(rdm)) / np.std(rdm) + 1e-10

            # plt.figure()
            # plt.imshow(rdm, aspect='auto', origin='lower', cmap='viridis')
            # plt.show()

            # Store image and label
            class_images.append(rdm)
            class_labels.append(n_targets)
        
        all_images.extend(class_images)
        all_labels.extend(class_labels)
    
    # Convert to numpy arrays
    images = np.array(all_images)  # Shape: (total_samples, n_ranges, n_doppler_bins)
    labels = np.array(all_labels)  # Shape: (total_samples,)
    
    print(f"\nDataset generated!")
    print(f"Total samples: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Convert to PyTorch tensors
    images_tensor = torch.from_numpy(images).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset dictionary
    dataset = {
        'images': images_tensor,
        'labels': labels_tensor,
        'metadata': {
            'samples_per_class': samples_per_class,
            'max_targets': max_targets,
            'sea_state': sea_state,
            'n_ranges': rp.n_ranges,
            'n_doppler_bins': rp.n_pulses,
            'range_resolution': rp.range_resolution,
            'class_names': [f"{i}_targets" for i in range(max_targets + 1)]
        }
    }
    
    # Save to file
    torch.save(dataset, save_path)
    print(f"\nDataset saved to: {save_path}")
    print(f"File size: {torch.load(save_path)['images'].element_size() * torch.load(save_path)['images'].nelement() / (1024**2):.1f} MB")



def simulate_sequence_with_realistic_targets_and_masks(
    rp: sea_clutter.RadarParams,
    cp: sea_clutter.ClutterParams,
    sp: sea_clutter.SequenceParams,
    targets: List[sea_clutter.RealisticTarget],
    random_roll: bool = True
) -> tuple[list[np.ndarray], list[np.ndarray]]:  # Return RDMs and masks
    """Simulate sequence with multiple realistic targets and generate corresponding masks."""
    dt = 1.0 / sp.frame_rate_hz
    texture = None
    speckle_tail = None
    rdm_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    if random_roll:
        # Randomly roll the texture to simulate wave motion
        random_roll_bins = random.randint(-1200, 1200)

    for frame_idx in range(sp.n_frames):
        if texture is not None and cp.wave_speed_mps != 0:
            shift_bins = int(round(cp.wave_speed_mps * dt / rp.range_resolution))
            texture = np.roll(texture, shift=shift_bins, axis=0)
            
        clutter_td, texture, speckle_tail = sea_clutter.simulate_sea_clutter(
            rp, cp, texture=texture, init_speckle=speckle_tail
        )
        
        # Initialize binary mask for this frame
        target_mask = np.zeros((rp.n_ranges, rp.n_pulses), dtype=np.float32)
        
        # Update and add each target
        for tgt in targets:
            # Update target velocity with realistic variations
            sea_clutter.update_realistic_target_velocity(tgt, rp)
            
            # Convert to Target object for add_target_blob function
            simple_target = sea_clutter.Target(
                rng_idx=tgt.rng_idx,
                doppler_hz=tgt.doppler_hz,
                power=tgt.power
            )
            
            # Add target to clutter data
            sea_clutter.add_target_blob(clutter_td, simple_target, rp)
            
            # Mark target location in binary mask
            # Convert Doppler frequency to bin index
            doppler_bin = int(tgt.doppler_hz / (rp.prf / rp.n_pulses)) + 64
            doppler_bin = np.clip(doppler_bin, 0, rp.n_pulses - 1)
            
            # Mark target in mask with small blob
            blob_size = 1
            d_start = max(0, doppler_bin - blob_size - 1)
            d_end = min(rp.n_pulses, doppler_bin + blob_size + 1)
            target_mask[tgt.rng_idx, d_start:d_end] = 1.0
            
            # Update target range based on radial velocity
            range_change = tgt.current_velocity_mps * dt
            new_range = tgt.rng_idx * rp.range_resolution + range_change
            tgt.rng_idx = int(np.clip(new_range / rp.range_resolution, 0, rp.n_ranges - 1))
        
        # Compute RD map
        rdm = sea_clutter.compute_range_doppler(clutter_td, rp, cp)

        # Apply same roll to both RDM and mask
        if random_roll:
            rdm = np.roll(rdm, shift=random_roll_bins, axis=1)
            target_mask = np.roll(target_mask, shift=random_roll_bins, axis=1)

        rdm_list.append(rdm)
        mask_list.append(target_mask)
    
    return rdm_list, mask_list

def generate_segmentation_dataset_with_sequences(
    samples_per_class: int = 500,
    max_targets: int = 10,
    sea_state: int = 5,
    n_frames: int = 3,
    save_path: str = "data/sea_clutter_segmentation_sequences.pt"
) -> None:
    """
    Generate dataset for target segmentation with sequences and binary masks.
    
    Args:
        samples_per_class: Number of samples to generate per class
        max_targets: Maximum number of targets (classes will be 0 to max_targets)
        sea_state: WMO sea state to use
        n_frames: Number of frames per sequence
        save_path: Path to save the PyTorch file
    """
    
    # Initialize parameters
    rp = sea_clutter.RadarParams()
    cp = sea_clutter.get_clutter_params_for_sea_state(sea_state)
    
    # Set sequence parameters
    sp = sea_clutter.SequenceParams()
    sp.n_frames = n_frames
    
    print(f"Generating segmentation dataset with {samples_per_class} sequences per class")
    print(f"Classes: 0 to {max_targets} targets")
    print(f"Frames per sequence: {n_frames}")
    print(f"Sea state: {sea_state}")
    print(f"Range-Doppler map size: {rp.n_ranges} x {rp.n_pulses}")
    
    # Storage for data and labels
    all_sequences = []
    all_mask_sequences = []
    all_labels = []
    
    # Generate data for each class
    for n_targets in range(max_targets + 1):
        print(f"\nGenerating {samples_per_class} sequences for {n_targets} targets...")
        
        for i in tqdm(range(samples_per_class), desc=f"Class {n_targets}"):
            # Generate targets for this sequence
            targets = []
            if n_targets > 0:
                for _ in range(n_targets):
                    target = sea_clutter.create_realistic_target(
                        sea_clutter.TargetType.FIXED, 
                        random.randint(MIN_RANGE, MAX_RANGE), 
                        rp
                    )
                    targets.append(target)
            
            # Generate sequence of RDMs and masks
            rdm_list, mask_list = simulate_sequence_with_realistic_targets_and_masks(rp, cp, sp, targets)
            
            # Process each frame in the sequence
            processed_sequence = []
            processed_mask_sequence = []
            
            for rdm, mask in zip(rdm_list, mask_list):
                # Convert to dB scale and normalize
                rdm_db = 20 * np.log10(np.abs(rdm) + 1e-10)
                rdm_normalized = (rdm_db - np.mean(rdm_db)) / (np.std(rdm_db) + 1e-10)
                processed_sequence.append(rdm_normalized)
                processed_mask_sequence.append(mask)
            
            # Stack frames into sequence arrays
            sequence = np.stack(processed_sequence, axis=0)  # Shape: (n_frames, n_ranges, n_doppler_bins)
            mask_sequence = np.stack(processed_mask_sequence, axis=0)  # Shape: (n_frames, n_ranges, n_doppler_bins)
            
            # Store sequence and label
            all_sequences.append(sequence)
            all_mask_sequences.append(mask_sequence)
            all_labels.append(n_targets)
    
    # Convert to numpy arrays
    sequences = np.array(all_sequences)  # Shape: (total_samples, n_frames, n_ranges, n_doppler_bins)
    mask_sequences = np.array(all_mask_sequences)  # Shape: (total_samples, n_frames, n_ranges, n_doppler_bins)
    labels = np.array(all_labels)  # Shape: (total_samples,)
    
    print(f"\nDataset generated!")
    print(f"Total sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Mask sequence shape: {mask_sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Convert to PyTorch tensors
    sequences_tensor = torch.from_numpy(sequences).float()
    mask_sequences_tensor = torch.from_numpy(mask_sequences).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset dictionary
    dataset = {
        'sequences': sequences_tensor,
        'mask_sequences': mask_sequences_tensor,
        'labels': labels_tensor,
        'metadata': {
            'samples_per_class': samples_per_class,
            'max_targets': max_targets,
            'sea_state': sea_state,
            'n_frames': n_frames,
            'n_ranges': rp.n_ranges,
            'n_doppler_bins': rp.n_pulses,
            'range_resolution': rp.range_resolution,
            'class_names': [f"{i}_targets" for i in range(max_targets + 1)],
            'dataset_type': 'sequence_segmentation'
        }
    }
    
    # Save to file
    torch.save(dataset, save_path)
    print(f"\nDataset saved to: {save_path}")
    
    # Calculate file size
    total_size = (sequences_tensor.element_size() * sequences_tensor.nelement() + 
                  mask_sequences_tensor.element_size() * mask_sequences_tensor.nelement()) / (1024**2)
    print(f"File size: {total_size:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic sea clutter datasets")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of sequences to generate for each class")
    parser.add_argument("--max-targets", type=int, default=10,
                        help="Maximum number of targets present in the RD map")
    parser.add_argument("--sea-state", type=int, default=5,
                        help="WMO sea state used for the simulation")
    parser.add_argument("--frames", type=int, default=3,
                        help="Number of frames per sequence")
    parser.add_argument("--save-path", type=str,
                        default="sea_clutter_segmentation_sequences.pt",
                        help="Path where the generated dataset will be saved")

    args = parser.parse_args()

    generate_segmentation_dataset_with_sequences(
        samples_per_class=args.samples,
        max_targets=args.max_targets,
        sea_state=args.sea_state,
        n_frames=args.frames,
        save_path=args.save_path,
    )
