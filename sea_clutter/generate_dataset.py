#!/usr/bin/env python3
"""
Generate dataset for target count classification in sea clutter.
Creates 2000 images per class (0-5 targets) and saves as PyTorch file.
"""

import torch
import numpy as np
import random
from tqdm import tqdm
from Parameters import RadarParams, ClutterParams, SequenceParams, TargetType, get_clutter_params_for_sea_state, create_realistic_target
from sea_clutter.physics import add_target_blob, compute_range_doppler, simulate_sea_clutter
from Parameters import Target


def generate_single_frame_with_targets(
    rp: RadarParams,
    cp: ClutterParams,
    n_targets: int,
) -> np.ndarray:
    """Generate a single range-Doppler map with specified number of targets."""
    
    # Generate sea clutter
    clutter_td, _, _ = simulate_sea_clutter(rp, cp)
    
    # Generate random targets if needed
    if n_targets > 0:
        max_range = rp.n_ranges * rp.range_resolution
        targets = [
            create_realistic_target(TargetType.FIXED, random.randint(0, max_range-1), rp) 
            for _ in range(n_targets)
        ]
        
        # Add each target to the clutter data
        for tgt in targets:
            simple_target = Target(
                rng_idx=tgt.rng_idx,
                doppler_hz=tgt.doppler_hz,
                power=tgt.power
            )
            add_target_blob(clutter_td, simple_target, rp)
    
    # Compute range-Doppler map
    rdm = compute_range_doppler(clutter_td, rp, cp)
    
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
    rp = RadarParams()
    cp = get_clutter_params_for_sea_state(sea_state)
    
    # Set single frame
    sp = SequenceParams()
    
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate sea clutter classification dataset")
    parser.add_argument("--samples", type=int, default=2000, 
                        help="Number of samples per class (default: 2000)")
    parser.add_argument("--max-targets", type=int, default=5,
                        help="Maximum number of targets (default: 5)")
    parser.add_argument("--sea-state", type=int, choices=[1,3,5,7,9], default=5,
                        help="WMO sea state (default: 5)")
    parser.add_argument("--output", type=str, default="data/sea_clutter_dataset.pt",
                        help="Output file path (default: sea_clutter_dataset.pt)")
    
    args = parser.parse_args()

    # rdm_list = simulate_sequence_with_realistic_targets(
    #     RadarParams(),
    #     get_clutter_params_for_sea_state(args.sea_state),
    #     SequenceParams(n_frames=args.frames),
    #     [create_realistic_target(TargetType.FIXED, random.randint(MIN_RANGE, MAX_RANGE), RadarParams()) for _ in range(args.max_targets)]
    # )
    # animate_sequence(
    #     rdm_list, None,
    #     RadarParams(), 
    # )
