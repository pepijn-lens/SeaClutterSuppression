#!/usr/bin/env python3
"""
segmentation_dataset_generator.py
=================================
Generate sea clutter datasets with multiple targets for image segmentation tasks.
Creates range-doppler maps with corresponding binary ground truth masks.
Simplified version with fixed radar/sequence parameters and predefined sea states.
"""

import torch
import os
import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import asdict, dataclass
import pickle

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Import your existing classes
from sea_clutter.sea_clutter import (
    RadarParams, ClutterParams, Target, SequenceParams,
    simulate_sea_clutter, compute_range_doppler, add_target_blob,
    get_clutter_params_for_sea_state
)

@dataclass
class SegmentationTarget:
    """Simplified target class for segmentation - blob targets only."""
    rng_idx: int
    doppler_hz: float
    power: float = 0.1
    rng_speed_mps: float = 0.0
    
    def to_base_target(self) -> Target:
        """Convert to base Target class for simulation."""
        return Target(
            rng_idx=self.rng_idx,
            doppler_hz=self.doppler_hz, 
            power=self.power,
            rng_speed_mps=self.rng_speed_mps
        )

class SegmentationDataGenerator:
    """Generate sea clutter data with ground truth masks for segmentation."""
    
    def __init__(self, output_dir: str = "segmentation_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Fixed parameters for all scenarios
        self.radar_params = RadarParams(
            prf=1000.0,
            n_pulses=256,
            n_ranges=512,
            range_resolution=0.5,
            carrier_wavelength=0.03
        )
        
        self.sequence_params = SequenceParams(
            n_frames=5,
            frame_rate_hz=1.0,
            wave_speed_mps=2.0
        )
        
        # Available sea states - updated to only include 1, 3, 5, 7
        self.sea_states = [1, 3, 5, 7]

    def generate_target_configurations(
        self,
        n_scenarios: int = 1000,
        min_targets: int = 0,
        max_targets: int = 5,
        target_power_range: Tuple[float, float] = (0.1, 20.0),
        target_speed_range: Tuple[float, float] = (-15.0, 15.0)
    ) -> List[Dict[str, Any]]:
        """Generate diverse scenarios with multiple blob targets."""
        scenarios = []
        
        for i in range(n_scenarios):
            # Random sea state
            sea_state = int(np.random.choice(self.sea_states))
            cp = get_clutter_params_for_sea_state(sea_state)
            
            # Generate multiple targets with realistic constraints
            n_targets = int(np.random.randint(min_targets, max_targets + 1))
            targets = []
            
            # Ensure targets don't overlap too much
            occupied_positions = []
            
            for _ in range(n_targets):
                # Try to place target without overlap
                attempts = 0
                while attempts < 50:  # Max attempts to avoid infinite loop
                    rng_idx = int(np.random.randint(20, self.radar_params.n_ranges - 20))
                    doppler_hz = float(np.random.uniform(-self.radar_params.prf/3, self.radar_params.prf/3))
                    
                    # Check for overlap with existing targets
                    too_close = False
                    for prev_rng, prev_doppler in occupied_positions:
                        if (abs(rng_idx - prev_rng) < 10 and 
                            abs(doppler_hz - prev_doppler) < 20):
                            too_close = True
                            break
                    
                    if not too_close:
                        break
                    attempts += 1
                
                if attempts < 50:  # Successfully placed
                    # Random target properties
                    power = float(np.random.uniform(*target_power_range))
                    speed = float(np.random.uniform(*target_speed_range))
                    
                    tgt = SegmentationTarget(
                        rng_idx=rng_idx,
                        doppler_hz=doppler_hz,
                        power=power,
                        rng_speed_mps=speed
                    )
                    targets.append(tgt)
                    occupied_positions.append((rng_idx, doppler_hz))
            
            scenarios.append({
                'scenario_id': i,
                'sea_state': sea_state,
                'radar_params': asdict(self.radar_params),
                'clutter_params': asdict(cp),
                'sequence_params': asdict(self.sequence_params),
                'targets': [asdict(t) for t in targets]
            })
            
        return scenarios
    
    def create_ground_truth_mask(
        self,
        targets: List[SegmentationTarget],
        frame_idx: int = 0
    ) -> np.ndarray:
        """Create binary ground truth mask for blob targets."""
        # Convert Doppler frequency to bin index
        doppler_bins = np.fft.fftshift(
            np.fft.fftfreq(self.radar_params.n_pulses, d=1.0/self.radar_params.prf)
        )
        
        # Initialize mask
        mask = np.zeros((self.radar_params.n_ranges, self.radar_params.n_pulses), dtype=np.uint8)
        
        for tgt in targets:
            # Find doppler bin closest to target frequency
            doppler_idx = int(np.argmin(np.abs(doppler_bins - tgt.doppler_hz)))
            
            current_rng_idx = int(np.clip(tgt.rng_idx, 0, self.radar_params.n_ranges - 1))

            # Blob target: 3x3 blob centered on target
            for dr in [-1, 0, 1]:
                for dd in [-1, 0, 1]:
                    r_idx = current_rng_idx + dr
                    d_idx = doppler_idx + dd
                    if 0 <= r_idx < self.radar_params.n_ranges and 0 <= d_idx < self.radar_params.n_pulses:
                        mask[r_idx, d_idx] = 1
        
        return mask
    
    def apply_augmentation(self, rdm_array: np.ndarray, mask_array: np.ndarray, 
                          augmentation_type: str = "none") -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to RDM and mask arrays."""
        if augmentation_type == "none":
            return rdm_array, mask_array
        
        augmented_rdm = rdm_array.copy()
        augmented_mask = mask_array.copy()
        
        if augmentation_type == "shift_range":
            # Shift in range direction (vertical)
            shift = np.random.randint(-20, 21)  # Shift by up to 20 bins
            if shift != 0:
                for frame_idx in range(augmented_rdm.shape[0]):
                    augmented_rdm[frame_idx] = np.roll(augmented_rdm[frame_idx], shift, axis=0)
                    augmented_mask[frame_idx] = np.roll(augmented_mask[frame_idx], shift, axis=0)
        
        elif augmentation_type == "shift_doppler":
            # Shift in Doppler direction (horizontal)
            shift = np.random.randint(-20, 21)  # Shift by up to 20 bins
            if shift != 0:
                for frame_idx in range(augmented_rdm.shape[0]):
                    augmented_rdm[frame_idx] = np.roll(augmented_rdm[frame_idx], shift, axis=1)
                    augmented_mask[frame_idx] = np.roll(augmented_mask[frame_idx], shift, axis=1)
        
        elif augmentation_type == "rotate_90":
            # 90-degree rotation
            for frame_idx in range(augmented_rdm.shape[0]):
                augmented_rdm[frame_idx] = np.rot90(augmented_rdm[frame_idx])
                augmented_mask[frame_idx] = np.rot90(augmented_mask[frame_idx])
        
        elif augmentation_type == "rotate_180":
            # 180-degree rotation
            for frame_idx in range(augmented_rdm.shape[0]):
                augmented_rdm[frame_idx] = np.rot90(augmented_rdm[frame_idx], 2)
                augmented_mask[frame_idx] = np.rot90(augmented_mask[frame_idx], 2)
        
        elif augmentation_type == "flip_range":
            # Flip along range axis
            augmented_rdm = np.flip(augmented_rdm, axis=1)
            augmented_mask = np.flip(augmented_mask, axis=1)
        
        elif augmentation_type == "flip_doppler":
            # Flip along Doppler axis
            augmented_rdm = np.flip(augmented_rdm, axis=2)
            augmented_mask = np.flip(augmented_mask, axis=2)
        
        return augmented_rdm, augmented_mask

    def generate_single_scenario(
        self, 
        scenario: Dict[str, Any],
        apply_augmentation: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Generate single scenario with RDM, ground truth mask, and targets-only mask."""
        # Reconstruct objects
        rp = RadarParams(**scenario['radar_params'])
        cp = ClutterParams(**scenario['clutter_params'])
        sp = SequenceParams(**scenario['sequence_params'])
        seg_targets = [SegmentationTarget(**t) for t in scenario['targets']]
        base_targets = [t.to_base_target() for t in seg_targets]
        
        # Generate sequence
        rdm_list = []
        mask_list = []
        targets_only_list = []
        
        dt = 1.0 / sp.frame_rate_hz
        texture = None
        speckle_tail = None
        
        for frame_idx in range(sp.n_frames):
            # Update wave motion
            if texture is not None and sp.wave_speed_mps != 0:
                shift_bins = int(round(sp.wave_speed_mps * dt / rp.range_resolution))
                texture = np.roll(texture, shift=shift_bins, axis=0)
            
            # Generate clutter
            clutter_td, texture, speckle_tail = simulate_sea_clutter(
                rp, cp, texture=texture, init_speckle=speckle_tail
            )
            
            # Create targets-only version (no clutter)
            targets_only_td = np.zeros_like(clutter_td)
            
            # Add targets to both clutter and targets-only versions
            for i, tgt in enumerate(base_targets):
                add_target_blob(clutter_td, tgt, rp)
                add_target_blob(targets_only_td, tgt, rp)
                # Update position for next frame
                tgt.rng_idx = max(0, min(rp.n_ranges - 1, 
                    tgt.rng_idx + int(tgt.rng_speed_mps * dt / rp.range_resolution)))
                # Also update the segmentation target for mask generation
                seg_targets[i].rng_idx = tgt.rng_idx
            
            # Compute range-doppler maps
            rdm = compute_range_doppler(clutter_td, rp, cp)
            targets_only_rdm = compute_range_doppler(targets_only_td, rp, cp)
            rdm_list.append(rdm)
            targets_only_list.append(targets_only_rdm)
            
            # Create ground truth mask
            mask = self.create_ground_truth_mask(seg_targets, frame_idx)
            mask_list.append(mask)
        
        # Convert to arrays
        rdm_array = np.array(rdm_list)  # Shape: (n_frames, n_ranges, n_doppler)
        mask_array = np.array(mask_list)  # Shape: (n_frames, n_ranges, n_doppler)
        targets_only_array = np.array(targets_only_list)  # Shape: (n_frames, n_ranges, n_doppler)
        
        # Apply augmentation if requested
        if apply_augmentation:
            augmentation_types = ["none", "shift_range", "shift_doppler", "rotate_90", "rotate_180", "flip_range", "flip_doppler"]
            aug_type = np.random.choice(augmentation_types)
            rdm_array, mask_array = self.apply_augmentation(rdm_array, mask_array, aug_type)
            targets_only_array, _ = self.apply_augmentation(targets_only_array, mask_array, aug_type)
            scenario['augmentation'] = aug_type
        else:
            scenario['augmentation'] = "none"
        
        return rdm_array, mask_array, targets_only_array, scenario

def worker_function(args):
    """Worker function for multiprocessing."""
    scenario, output_dir, apply_augmentation = args
    generator = SegmentationDataGenerator(output_dir)
    
    try:
        rdm_data, mask_data, targets_only_data, metadata = generator.generate_single_scenario(
            scenario, apply_augmentation=apply_augmentation
        )
        return rdm_data, mask_data, targets_only_data, metadata, None
    except Exception as e:
        return None, None, None, None, str(e)

class SegmentationBatchProcessor:
    """Batch processor for segmentation datasets."""
    
    def __init__(self, output_dir: str = "segmentation_dataset", n_workers: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

    def save_pt_dataset(
        self,
        scenarios: List[Dict[str, Any]],
        batch_size: int = 50,
        apply_augmentation: bool = True
    ):
        """Save segmentation dataset in .pt (PyTorch) format."""
        print(f"Generating {len(scenarios)} scenarios using {self.n_workers} workers (saving as .pt files)...")
        
        pt_dir = self.output_dir / "pt_dataset"
        pt_dir.mkdir(exist_ok=True)

        for batch_start in range(0, len(scenarios), batch_size):
            batch_end = min(batch_start + batch_size, len(scenarios))
            batch_scenarios = scenarios[batch_start:batch_end]

            print(f"Processing batch {batch_start//batch_size + 1}/{(len(scenarios)-1)//batch_size + 1}")

            args_list = [(scenario, str(self.output_dir), apply_augmentation) for scenario in batch_scenarios]

            with mp.Pool(self.n_workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_function, args_list),
                    total=len(args_list),
                    desc="Generating data"
                ))

            for i, (rdm_data, mask_data, targets_only_data, metadata, error) in enumerate(results):
                if error:
                    print(f"Error in scenario {batch_start + i}: {error}")
                    continue

                scenario_id = metadata['scenario_id']

                # Save as PyTorch file with targets-only data
                torch.save({
                    'rdm': torch.tensor(rdm_data, dtype=torch.float32),
                    'mask': torch.tensor(mask_data, dtype=torch.uint8),
                    'targets_only': torch.tensor(targets_only_data, dtype=torch.float32),
                    'metadata': metadata
                }, pt_dir / f"scenario_{scenario_id:05d}.pt")

        print(f"Segmentation dataset saved in {pt_dir}")
        return pt_dir


def main():
    parser = argparse.ArgumentParser(description="Generate simplified sea clutter segmentation dataset")
    parser.add_argument("--n_scenarios", type=int, default=20000,  # Updated default
                        help="Number of scenarios to generate")
    parser.add_argument("--output_dir", type=str, default="segmentation_dataset",
                        help="Output directory")
    parser.add_argument("--min_targets", type=int, default=0,
                        help="Minimum number of targets per scenario")
    parser.add_argument("--max_targets", type=int, default=5,  # Updated default
                        help="Maximum number of targets per scenario")
    parser.add_argument("--n_workers", type=int, default=10,
                        help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for processing")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate sample visualizations")
    parser.add_argument("--save_pt", action="store_true", default=True,  # Default to True
                    help="Save dataset in .pt (PyTorch) format instead of HDF5")
    parser.add_argument("--no_augmentation", action="store_true",
                    help="Disable data augmentation")

    
    args = parser.parse_args()
    
    # Generate scenarios
    generator = SegmentationDataGenerator(args.output_dir)
    scenarios = generator.generate_target_configurations(
        n_scenarios=args.n_scenarios,
        min_targets=args.min_targets,
        max_targets=args.max_targets
    )
    
    # Save configurations
    with open(Path(args.output_dir) / "scenarios.json", 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    # Generate dataset
    processor = SegmentationBatchProcessor(args.output_dir, args.n_workers)
    
    start_time = time.time()
    if args.save_pt:
        dataset_path = processor.save_pt_dataset(
            scenarios, 
            batch_size=args.batch_size,
            apply_augmentation=not args.no_augmentation
        )
    else:
        dataset_path = processor.save_hdf5_dataset(scenarios, batch_size=args.batch_size)

    elapsed = time.time() - start_time
    
    print(f"Generated {len(scenarios)} scenarios in {elapsed:.1f} seconds")
    print(f"Average: {elapsed/len(scenarios):.3f} seconds per scenario")
    
    # Generate visualizations
    if args.visualize:
        processor.visualize_samples(dataset_path, n_samples=5)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    target_counts = [len(s['targets']) for s in scenarios]
    sea_state_counts = [s['sea_state'] for s in scenarios]
    
    print(f"Target distribution: {np.bincount(target_counts)}")
    print(f"Average targets per scenario: {np.mean(target_counts):.2f}")
    print(f"Sea state distribution: {dict(zip(*np.unique(sea_state_counts, return_counts=True)))}")
    print(f"Fixed parameters:")
    print(f"  - Radar: PRF={generator.radar_params.prf}Hz, {generator.radar_params.n_ranges} range bins, {generator.radar_params.n_pulses} pulses")
    print(f"  - Sequence: {generator.sequence_params.n_frames} frames, {generator.sequence_params.frame_rate_hz}Hz")

if __name__ == "__main__":
    main()