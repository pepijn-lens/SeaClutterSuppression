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
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ────────────────────────────────────────────────────────────────────────────────
# Dataclass parameter blocks
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class RadarParams:
    prf: float = 1000.0            # Pulse-repetition frequency [Hz]
    n_pulses: int = 256            # Pulses per coherent processing interval (CPI)
    n_ranges: int = 512            # Range bins
    range_resolution: float = 0.5  # [m]
    carrier_wavelength: float = 0.03  # [m] – unused but kept for completeness

@dataclass
class ClutterParams:
    mean_power_db: float = -20.0    # Average clutter power per cell [dB]
    shape_param: float = 0.2        # Gamma shape κ; smaller → heavier tail
    ar_coeff: float = 0.98          # Slow-time AR(1) coefficient
    bragg_offset_hz: Optional[float] = 25.0  # Bragg Doppler [Hz]
    bragg_width_hz: float = 1.0     # Bragg peak width [Hz]
    bragg_power_rel: float = 5.0    # Bragg peak height over background [dB]

@dataclass
class Target:
    rng_idx: int
    doppler_hz: float
    power: float = 10.0             # Linear power for the central cell
    rng_speed_mps: float = 0.0      # Range velocity for motion (m/s)

@dataclass
class SequenceParams:
    n_frames: int = 100             # Frames to simulate
    frame_rate_hz: float = 3.0      # Frames per second
    wave_speed_mps: float = 5.0     # Wave propagation speed toward radar [m/s]

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def db(x: np.ndarray) -> np.ndarray:
    """Safe 10·log₁₀ helper that avoids −inf."""
    return 10.0 * np.log10(np.maximum(x, 1e-15))

# ────────────────────────────────────────────────────────────────────────────────
# Internal clutter routines
# ────────────────────────────────────────────────────────────────────────────────

def _generate_texture(n_ranges: int, n_pulses: int, k: float) -> np.ndarray:
    """Gamma-distributed texture with unit mean."""
    tex = np.random.gamma(shape=k, scale=1.0 / k, size=(n_ranges, 1))
    return np.repeat(tex, n_pulses, axis=1)


def _generate_speckle(
    n_ranges: int,
    n_pulses: int,
    ar: float,
    init_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Complex Gaussian speckle with AR(1) correlation along slow-time."""
    white = (
        np.random.randn(n_ranges, n_pulses) + 1j * np.random.randn(n_ranges, n_pulses)
    ) / np.sqrt(2.0)
    x = np.empty_like(white)
    if init_state is None:
        x[:, 0] = white[:, 0] / np.sqrt(1.0 - ar * ar)
    else:
        x[:, 0] = ar * init_state + white[:, 0]
    for k in range(1, n_pulses):
        x[:, k] = ar * x[:, k - 1] + white[:, k]
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    return x


def _time_to_range_doppler(data: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)


def _inject_bragg(rd: np.ndarray, prf: float, offset_hz: float, width_hz: float, boost_db: float):
    """Multiply RD map by twin Gaussian Bragg peaks."""
    n_pulses = rd.shape[1]
    fd = np.fft.fftshift(np.fft.fftfreq(n_pulses, d=1.0 / prf))
    weight = (
        np.exp(-0.5 * ((fd - offset_hz) / width_hz) ** 2)
        + np.exp(-0.5 * ((fd + offset_hz) / width_hz) ** 2)
    )
    rd *= 1.0 + (10.0 ** (boost_db / 10.0) - 1.0) * weight[np.newaxis, :]

def get_clutter_params_for_sea_state(state: int) -> ClutterParams:
    """Map WMO sea states (1..9) to model parameters."""
    configs = {
        1: {'mean_power_db': -30.0, 'shape_param': 2.0, 'ar_coeff': 0.995},
        3: {'mean_power_db': -25.0, 'shape_param': 1.0, 'ar_coeff': 0.990},
        5: {'mean_power_db': -20.0, 'shape_param': 0.5, 'ar_coeff': 0.980},
        7: {'mean_power_db': -15.0, 'shape_param': 0.2, 'ar_coeff': 0.950},
        9: {'mean_power_db': -10.0, 'shape_param': 0.1, 'ar_coeff': 0.900},
    }
    if state not in configs:
        raise ValueError(f"Unsupported sea state {state}; choose from {list(configs)}")
    params = configs[state]
    return ClutterParams(**params)

# ────────────────────────────────────────────────────────────────────────────────
# Single-frame API
# ────────────────────────────────────────────────────────────────────────────────

def simulate_sea_clutter(
    rp: RadarParams,
    cp: ClutterParams,
    *,
    texture: np.ndarray | None = None,
    init_speckle: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if texture is None:
        texture = _generate_texture(rp.n_ranges, rp.n_pulses, cp.shape_param)
    speckle = _generate_speckle(rp.n_ranges, rp.n_pulses, cp.ar_coeff, init_state=init_speckle)
    clutter_td = np.sqrt(texture) * speckle * 10.0 ** (cp.mean_power_db / 20.0)
    return clutter_td, texture, speckle[:, -1]


def add_target_blob(signal_td: np.ndarray, tgt: Target, rp: RadarParams):
    """Add a 3x1 range blob around tgt.rng_idx with central peak and weaker neighbors."""
    n = np.arange(rp.n_pulses)
    phase = np.exp(1j * 2.0 * np.pi * tgt.doppler_hz * n / rp.prf)
    amp_center = np.sqrt(tgt.power)
    # Surrounding cell weight (e.g. 70% of center)
    neighbor_weight = 0.7
    for offset in (-1, 0, 1):
        idx = tgt.rng_idx + offset
        if 0 <= idx < rp.n_ranges:
            weight = amp_center * (1.0 if offset == 0 else neighbor_weight)
            signal_td[idx] += weight * phase


def compute_range_doppler(signal_td: np.ndarray, rp: RadarParams, cp: ClutterParams) -> np.ndarray:
    rd = _time_to_range_doppler(signal_td)
    if cp.bragg_offset_hz is not None:
        _inject_bragg(rd, rp.prf, cp.bragg_offset_hz, cp.bragg_width_hz, cp.bragg_power_rel)
    return rd

# ────────────────────────────────────────────────────────────────────────────────
# Sequence generator with moving blob target
# ────────────────────────────────────────────────────────────────────────────────

def simulate_sequence(
    rp: RadarParams,
    cp: ClutterParams,
    sp: SequenceParams,
    targets: Optional[List[Target]] = None,
) -> List[np.ndarray]:
    if targets is None:
        targets = []
    dt = 1.0 / sp.frame_rate_hz
    # precompute per-target range shift per frame
    rng_shifts = [int(round(t.rng_speed_mps * dt / rp.range_resolution)) for t in targets]

    texture = None
    speckle_tail = None
    rdm_list: List[np.ndarray] = []

    for frame_idx in range(sp.n_frames):
        if texture is not None and sp.wave_speed_mps != 0:
            shift_bins = int(round(sp.wave_speed_mps * dt / rp.range_resolution))
            texture = np.roll(texture, shift=shift_bins, axis=0)

        clutter_td, texture, speckle_tail = simulate_sea_clutter(
            rp, cp, texture=texture, init_speckle=speckle_tail
        )
        # add each target blob at current position
        for i, tgt in enumerate(targets):
            add_target_blob(clutter_td, tgt, rp)
            # update target position for next frame
            tgt.rng_idx = max(0, min(rp.n_ranges - 1, tgt.rng_idx + rng_shifts[i]))

        rdm_list.append(compute_range_doppler(clutter_td, rp, cp))

    return rdm_list

# ────────────────────────────────────────────────────────────────────────────────
# Animation helper (unchanged)
# ────────────────────────────────────────────────────────────────────────────────
def animate_sequence(
    rdm_list: List[np.ndarray],
    rp: RadarParams,
    *,
    interval_ms: int = 1000,
    save_path: Optional[str] = None,
):
    fd = np.fft.fftshift(np.fft.fftfreq(rp.n_pulses, d=1.0 / rp.prf))
    rng = np.arange(rp.n_ranges) * rp.range_resolution
    imgs_db = [db(np.abs(rd) ** 2) for rd in rdm_list]
    vmax = max(map(np.max, imgs_db)); vmin = vmax - 50.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(imgs_db[0], aspect="auto", cmap="viridis",
                   extent=[fd[0], fd[-1], rng[-1], rng[0]],
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Doppler (Hz)"); ax.set_ylabel("Range (m)")
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

# ────────────────────────────────────────────────────────────────────────────────
# Demo entry point (with moving blob)
# ────────────────────────────────────────────────────────────────────────────────
def simulate_example(save_gif: bool = False, cp = ClutterParams()) -> None:
    rp = RadarParams(); cp = cp; sp = SequenceParams()
    # Blob target: central range bin=180, Doppler=10Hz, power=1, moving inward at 2 m/s
    tgt = Target(rng_idx=180, doppler_hz=10.0, power=1.0, rng_speed_mps=-2.0)
    rdm_list = simulate_sequence(rp, cp, sp, targets=[tgt])
    save_path = "sea_clutter.gif" if save_gif else None
    interval_ms = int(1000.0 / sp.frame_rate_hz)
    animate_sequence(rdm_list, rp, interval_ms=interval_ms, save_path=save_path)

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
    # run the built-in demo but override the ClutterParams
    simulate_example(save_gif=args.gif, cp=cp)
