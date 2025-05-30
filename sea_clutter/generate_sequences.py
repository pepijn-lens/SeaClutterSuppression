from typing import List
import torch
import numpy as np
import random
from tqdm import tqdm
from Parameters import RadarParams, ClutterParams, SequenceParams, TargetType, get_clutter_params_for_sea_state, create_realistic_target
from sea_clutter.physics import add_target_blob, compute_range_doppler, simulate_sea_clutter
from sea_helper import update_realistic_target_velocity, RealisticTarget
from Parameters import Target

MIN_RANGE = 25  # Minimum range for targets
MAX_RANGE = 128 - 25  # Maximum range for targets

def simulate_sequence_with_realistic_targets(
    rp: RadarParams,
    cp: ClutterParams,
    sp: SequenceParams,
    targets: List[RealisticTarget],
) -> list[np.ndarray]:  # Return only RDMs
    """Simulate sequence with multiple realistic targets."""
    dt = 1.0 / sp.frame_rate_hz
    texture = None
    speckle_tail = None
    rdm_list: list[np.ndarray] = []
    
    for _ in range(sp.n_frames):
        if texture is not None and cp.wave_speed_mps != 0:
            shift_bins = int(round(cp.wave_speed_mps * dt / rp.range_resolution))
            texture = np.roll(texture, shift=shift_bins, axis=0)
            
        clutter_td, texture, speckle_tail = simulate_sea_clutter(
            rp, cp, texture=texture, init_speckle=speckle_tail
        )
        
        # Update and add each target
        for tgt in targets:
            # Update target velocity with realistic variations
            update_realistic_target_velocity(tgt)
            
            # Convert to Target object for add_target_blob function
            simple_target = Target(
                rng_idx=tgt.rng_idx,
                doppler_hz=tgt.doppler_hz,
                power=tgt.power
            )
            
            # Add target to clutter data
            add_target_blob(clutter_td, simple_target, rp)
            
            # Update target range based on radial velocity
            range_change = tgt.current_velocity_mps * dt
            new_range = tgt.rng_idx * rp.range_resolution + range_change
            tgt.rng_idx = int(np.clip(new_range / rp.range_resolution, 0, rp.n_ranges - 1))
        
        # Compute RD map
        rdm = compute_range_doppler(clutter_td, rp, cp)
        rdm_list.append(rdm)
    
    return rdm_list

def generate_classification_dataset_with_sequences(
    samples_per_class: int = 2000,
    max_targets: int = 5,
    sea_state: int = 5,
    n_frames: int = 4,
    save_path: str = "sea_clutter-r_frames.pt"
) -> None:
    """
    Generate dataset for target count classification with sequences.
    
    Args:
        samples_per_class: Number of samples to generate per class
        max_targets: Maximum number of targets (classes will be 0 to max_targets)
        sea_state: WMO sea state to use
        n_frames: Number of frames per sequence
        save_path: Path to save the PyTorch file
    """
    
    # Initialize parameters
    rp = RadarParams()
    cp = get_clutter_params_for_sea_state(sea_state)
    
    # Set sequence parameters
    sp = SequenceParams()
    sp.n_frames = n_frames
    
    print(f"Generating dataset with {samples_per_class} sequences per class")
    print(f"Classes: 0 to {max_targets} targets")
    print(f"Frames per sequence: {n_frames}")
    print(f"Sea state: {sea_state}")
    print(f"Range-Doppler map size: {rp.n_ranges} x {rp.n_pulses}")
    
    # Storage for data and labels
    all_sequences = []
    all_labels = []
    
    # Generate data for each class
    for n_targets in range(max_targets + 1):
        print(f"\nGenerating {samples_per_class} sequences for {n_targets} targets...")
        
        for i in tqdm(range(samples_per_class), desc=f"Class {n_targets}"):
            # Generate targets for this sequence
            targets = []
            if n_targets > 0:
                for _ in range(n_targets):
                    target = create_realistic_target(
                        TargetType.FIXED, 
                        random.randint(MIN_RANGE, MAX_RANGE), 
                        rp
                    )
                    targets.append(target)
            
            # Generate sequence of RDMs
            rdm_list = simulate_sequence_with_realistic_targets(rp, cp, sp, targets)
            
            # Process each frame in the sequence
            processed_sequence = []
            for rdm in rdm_list:
                # Convert to dB scale and normalize
                rdm_db = 20 * np.log10(np.abs(rdm) + 1e-10)
                rdm_normalized = (rdm_db - np.mean(rdm_db)) / (np.std(rdm_db) + 1e-10)
                processed_sequence.append(rdm_normalized)
            
            # Stack frames into sequence array
            sequence = np.stack(processed_sequence, axis=0)  # Shape: (n_frames, n_ranges, n_doppler_bins)
            
            # Store sequence and label
            all_sequences.append(sequence)
            all_labels.append(n_targets)
    
    # Convert to numpy arrays
    sequences = np.array(all_sequences)  # Shape: (total_samples, n_frames, n_ranges, n_doppler_bins)
    labels = np.array(all_labels)  # Shape: (total_samples,)
    
    print(f"\nDataset generated!")
    print(f"Total sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Convert to PyTorch tensors
    sequences_tensor = torch.from_numpy(sequences).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset dictionary
    dataset = {
        'sequences': sequences_tensor,
        'labels': labels_tensor,
        'metadata': {
            'samples_per_class': samples_per_class,
            'max_targets': max_targets,
            'sea_state': sea_state,
            'n_frames': n_frames,
            'n_ranges': rp.n_ranges,
            'n_doppler_bins': rp.n_pulses,
            'range_resolution': rp.range_resolution,
            'class_names': [f"{i}_targets" for i in range(max_targets + 1)]
        }
    }
    
    # Save to file
    torch.save(dataset, save_path)
    print(f"\nDataset saved to: {save_path}")
    
    # Calculate file size
    file_size_mb = sequences_tensor.element_size() * sequences_tensor.nelement() / (1024**2)
    print(f"File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sea clutter classification dataset")
    parser.add_argument("--samples", type=int, default=2000, 
                        help="Number of samples per class (default: 2000)")
    parser.add_argument("--max-targets", type=int, default=5,
                        help="Maximum number of targets (default: 5)")
    parser.add_argument("--sea-state", type=int, choices=[1,3,5,7,9], default=5,
                        help="WMO sea state (default: 5)")
    parser.add_argument("--frames", type=int, default=4,
                        help="Number of frames per sequence (default: 4)")
    parser.add_argument("--output", type=str, default="data/sea_clutter-4_frames-25SCR.pt",
                        help="Output file path (default: sea_clutter-r_frames.pt)")

    args = parser.parse_args()

    generate_classification_dataset_with_sequences(
        samples_per_class=args.samples,
        max_targets=args.max_targets,
        sea_state=args.sea_state,
        n_frames=args.frames,
        save_path=args.output
    )