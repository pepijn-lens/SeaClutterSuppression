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
from typing import List
import random

import numpy as np

from sea_helper import animate_sequence, update_realistic_target_velocity
from parameters import RadarParams, ClutterParams, SequenceParams, Target, RealisticTarget, TargetType, get_clutter_params_for_sea_state, create_realistic_target
from physics import add_target_blob, compute_range_doppler, simulate_sea_clutter

# ────────────────────────────────────────────────────────────────────────────────
# Demo entry point (with moving blobs)
# ────────────────────────────────────────────────────────────────────────────────

def simulate_sequence_with_realistic_targets(
    rp: RadarParams,
    cp: ClutterParams,
    sp: SequenceParams,
    targets: List[RealisticTarget],
    random_roll: bool = False
) -> list[np.ndarray]:  # Return only RDMs
    """Simulate sequence with multiple realistic targets."""
    dt = 1.0 / sp.frame_rate_hz
    texture = None
    speckle_tail = None
    rdm_list: list[np.ndarray] = []
    if random_roll:
        # Randomly roll the texture to simulate wave motion
        random_roll_bins = random.randint(-1200, 1200)

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

        rdm = np.roll(rdm, shift=random_roll_bins, axis=1) if random_roll else rdm

        rdm_list.append(rdm)
    
    return rdm_list

def simulate_example_with_multiple_targets(save_gif: bool = False, cp = ClutterParams()) -> None:
    rp = RadarParams()
    cp = cp
    sp = SequenceParams()  # Longer sequence to see movement
    min_range = int(round(20 * sp.n_frames/sp.frame_rate_hz/2))  # Minimum range for targets
    max_range = int(round(rp.n_ranges * rp.range_resolution - (20 * sp.n_frames/sp.frame_rate_hz)/2))  # Maximum range for targets
    
    # # Create a variety of targets
    # targets = [
    #     create_realistic_target(TargetType.CARGO_SHIP, random.randint(0, max_range), rp),
    #     create_realistic_target(TargetType.FISHING_VESSEL, random.randint(0, max_range), rp),
    #     create_realistic_target(TargetType.PATROL_BOAT, random.randint(0, max_range), rp),
    #     create_realistic_target(TargetType.SMALL_CRAFT, random.randint(0, max_range), rp),
    #     create_realistic_target(TargetType.SPEEDBOAT, random.randint(0, max_range), rp),
    # ]
    
    targets = [create_realistic_target(TargetType.FIXED, random.randint(min_range, max_range), rp) for _ in range(10)]

    # Print target information
    print("Simulating targets:")
    for i, tgt in enumerate(targets):
        print(f"  {i+1}. {tgt.target_type.value}: Range {tgt.rng_idx*rp.range_resolution:.0f}m, "
              f"Initial velocity {tgt.current_velocity_mps:.1f} m/s")
    
    rdm_list = simulate_sequence_with_realistic_targets(rp, cp, sp, targets, random_roll=False)
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
                        default=5,
                        help="WMO sea state (1,3,5,7,9).")
    
    args = parser.parse_args()
    
    # grab clutter params for the requested sea state
    cp = get_clutter_params_for_sea_state(args.state)
    
    simulate_example_with_multiple_targets(save_gif=args.gif, cp=cp)
