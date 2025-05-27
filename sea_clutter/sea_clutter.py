#!/usr/bin/env python3
"""
sea_clutter_range_doppler.py
=================================
Generate **time-sequences** of synthetic Range-Doppler maps that contain
compound-Gaussian (SIRP) sea clutter plus optional point targets – then
**play them back as an animation or export as a GIF/MP4**.

The model retains the physics-inspired ingredients from the original
single-frame prototype – gamma texture × AR(1) speckle, optional Bragg
peaks – but now supports a third dimension (frames) so adjacent CPIs are
spatio-temporally correlated.  Wave motion is approximated by translating
the texture map toward the radar at a configurable speed.

Quick start
-----------
```bash
python sea_clutter_range_doppler.py          # 5-frame live animation
python sea_clutter_range_doppler.py --gif    # also writes sea_clutter.gif
```
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Tuple
import random

import numpy as np

from sea_helper import animate_sequence, update_realistic_target_velocity, print_target_tracks, animate_targets_with_masks, create_target_mask
from Parameters import RadarParams, ClutterParams, SequenceParams, Target, RealisticTarget, TargetType, get_clutter_params_for_sea_state, create_realistic_target
from rd_map import add_target_blob, compute_range_doppler, simulate_sea_clutter

# ────────────────────────────────────────────────────────────────────────────────
# Demo entry point (with moving blobs)
# ────────────────────────────────────────────────────────────────────────────────

def simulate_sequence_with_realistic_targets(
    rp: RadarParams,
    cp: ClutterParams,
    sp: SequenceParams,
    targets: List[RealisticTarget],
    sea_state: int = 3,
) -> list[np.ndarray]:
    """Simulate sequence with multiple realistic targets."""
    dt = 1.0 / sp.frame_rate_hz
    texture = None
    speckle_tail = None
    rdm_list: list[np.ndarray] = []
    
    for frame_idx in range(sp.n_frames):
        if texture is not None and sp.wave_speed_mps != 0:
            shift_bins = int(round(sp.wave_speed_mps * dt / rp.range_resolution))
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
            add_target_blob(clutter_td, simple_target, rp)
            
            # Update target range based on radial velocity
            range_change = tgt.current_velocity_mps * dt
            new_range = tgt.rng_idx * rp.range_resolution + range_change
            tgt.rng_idx = int(np.clip(new_range / rp.range_resolution, 0, rp.n_ranges - 1))
        
        rdm_list.append(compute_range_doppler(clutter_td, rp, cp))
    
    return rdm_list

def track_targets_and_create_masks(
    rp: RadarParams,
    cp: ClutterParams,
    sp: SequenceParams,
    targets: List[RealisticTarget],
    sea_state: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, List]]:
    """
    Simulate sequence with target tracking and mask creation.
    
    Returns:
        rdm_list: List of range-Doppler maps
        mask_list: List of target masks for each frame
        target_tracks: Dictionary with tracking data
    """
    dt = 1.0 / sp.frame_rate_hz
    texture = None
    speckle_tail = None
    rdm_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    
    # Initialize tracking data
    target_tracks = {
        'frame_idx': [],
        'target_ranges': [],      # List of lists: [frame][target_idx] = range
        'target_velocities': [],  # List of lists: [frame][target_idx] = velocity  
        'target_doppler_hz': [],  # List of lists: [frame][target_idx] = doppler_hz
        'target_types': [tgt.target_type.value for tgt in targets]
    }
    
    for frame_idx in range(sp.n_frames):
        if texture is not None and sp.wave_speed_mps != 0:
            shift_bins = int(round(sp.wave_speed_mps * dt / rp.range_resolution))
            texture = np.roll(texture, shift=shift_bins, axis=0)
            
        clutter_td, texture, speckle_tail = simulate_sea_clutter(
            rp, cp, texture=texture, init_speckle=speckle_tail
        )
        
        # Track current frame data
        frame_ranges = []
        frame_velocities = []
        frame_doppler_hz = []
        
        # Update and add each target
        for tgt in targets:
            # Update target velocity with realistic variations
            update_realistic_target_velocity(tgt, sea_state)
            
            # Record tracking data
            current_range = tgt.rng_idx * rp.range_resolution
            frame_ranges.append(current_range)
            frame_velocities.append(tgt.current_velocity_mps)
            frame_doppler_hz.append(tgt.doppler_hz)
            
            # Convert to Target object for add_target_blob function
            simple_target = Target(
                rng_idx=tgt.rng_idx,
                doppler_hz=tgt.doppler_hz,
                power=tgt.power
            )
            add_target_blob(clutter_td, simple_target, rp)
            
            # Update target range based on radial velocity
            range_change = tgt.current_velocity_mps * dt
            new_range = tgt.rng_idx * rp.range_resolution + range_change
            tgt.rng_idx = int(np.clip(new_range / rp.range_resolution, 0, rp.n_ranges - 1))
        
        # Store tracking data for this frame
        target_tracks['frame_idx'].append(frame_idx)
        target_tracks['target_ranges'].append(frame_ranges)
        target_tracks['target_velocities'].append(frame_velocities)
        target_tracks['target_doppler_hz'].append(frame_doppler_hz)
        
        # Create range-Doppler map and target mask
        rdm = compute_range_doppler(clutter_td, rp, cp)
        mask = create_target_mask(targets, rp)
        
        rdm_list.append(rdm)
        mask_list.append(mask)
    
    return rdm_list, mask_list, target_tracks

def simulate_example_with_tracking(save_gif: bool = False, cp = ClutterParams()) -> None:
    """Enhanced demo with target tracking and masking."""
    rp = RadarParams()
    sp = SequenceParams()  # Longer sequence for better tracking
    
    # Create targets
    targets = [
        create_realistic_target(TargetType.CARGO_SHIP, random.randint(50, 150), rp),
        create_realistic_target(TargetType.FISHING_VESSEL, random.randint(200, 300), rp),
        create_realistic_target(TargetType.PATROL_BOAT, random.randint(350, 450), rp),
    ]
    
    print("Simulating targets with tracking...")
    for i, tgt in enumerate(targets):
        print(f"  {i+1}. {tgt.target_type.value}: Range {tgt.rng_idx*rp.range_resolution:.0f}m, "
              f"Initial velocity {tgt.current_velocity_mps:.1f} m/s")
    
    # Run simulation with tracking
    rdm_list, mask_list, target_tracks = track_targets_and_create_masks(
        rp, cp, sp, targets, sea_state=5
    )
    
    # Print tracking results
    print_target_tracks(target_tracks)
    
    # Animate with masks
    save_path = "sea_clutter_tracked_targets.gif" if save_gif else None
    interval_ms = int(1000.0 / sp.frame_rate_hz)
    animate_targets_with_masks(rdm_list, mask_list, target_tracks, rp, 
                              interval_ms=interval_ms, save_path=save_path)
    
    return rdm_list, mask_list, target_tracks

# Demo with multiple realistic target types.
def simulate_example_with_multiple_targets(save_gif: bool = False, cp = ClutterParams()) -> None:
    rp = RadarParams()
    sp = SequenceParams()  # Longer sequence to see movement
    max_range = rp.n_ranges * rp.range_resolution
    
    # Create a variety of targets
    targets = [
        create_realistic_target(TargetType.CARGO_SHIP, random.randint(0, max_range), rp),
        create_realistic_target(TargetType.FISHING_VESSEL, random.randint(0, max_range), rp),
        create_realistic_target(TargetType.PATROL_BOAT, random.randint(0, max_range), rp),
        create_realistic_target(TargetType.SMALL_CRAFT, random.randint(0, max_range), rp),
        create_realistic_target(TargetType.SPEEDBOAT, random.randint(0, max_range), rp),
    ]
    
    # Print target information
    print("Simulating targets:")
    for i, tgt in enumerate(targets):
        print(f"  {i+1}. {tgt.target_type.value}: Range {tgt.rng_idx*rp.range_resolution:.0f}m, "
              f"Initial velocity {tgt.current_velocity_mps:.1f} m/s")
    
    rdm_list = simulate_sequence_with_realistic_targets(rp, cp, sp, targets)
    save_path = "sea_clutter_realistic_targets.gif" if save_gif else None
    interval_ms = int(1000.0 / sp.frame_rate_hz)
    animate_sequence(rdm_list, rp, interval_ms=interval_ms, save_path=save_path)

# ────────────────────────────────────────────────────────────────────────────────
# Main entry point (argparse setup)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate and animate sea clutter range-Doppler sequences."
    )
    parser.add_argument("--gif", action="store_true",
                        help="Save the animation as 'sea_clutter.gif'.")
    parser.add_argument("--state", type=int, choices=[1,3,5,7,9],
                        default=7,
                        help="WMO sea state (1,3,5,7,9).")
    parser.add_argument("--multi-targets", action="store_true",
                        help="Simulate with multiple realistic target types.")
    parser.add_argument("--track-targets", action="store_true",
                        help="Enable target tracking and masking visualization.")
    args = parser.parse_args()
    
    # grab clutter params for the requested sea state
    cp = get_clutter_params_for_sea_state(args.state)
    
    if args.track_targets:
        simulate_example_with_tracking(save_gif=args.gif, cp=cp)
    else:
        simulate_example_with_multiple_targets(save_gif=args.gif, cp=cp)
