from __future__ import annotations

from typing import Optional

import numpy as np
from .parameters import RadarParams, ClutterParams, Target


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


def _generate_thermal_noise(n_ranges: int, n_pulses: int, noise_power_db: float) -> np.ndarray:
    """Generate complex Gaussian thermal noise."""
    noise = (
        np.random.randn(n_ranges, n_pulses) + 1j * np.random.randn(n_ranges, n_pulses)
    ) / np.sqrt(2.0)
    # Scale to desired power level
    noise *= 10.0 ** (noise_power_db / 20.0)
    return noise

def simulate_sea_clutter(
    rp: RadarParams,
    cp: ClutterParams,
    *,
    texture: np.ndarray | None = None,
    init_speckle: np.ndarray | None = None,
    thermal_noise_db: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if texture is None:
        texture = _generate_texture(rp.n_ranges, rp.n_pulses, cp.shape_param)
    speckle = _generate_speckle(rp.n_ranges, rp.n_pulses, cp.ar_coeff, init_state=init_speckle)
    clutter_td = texture * speckle * 10.0 ** (cp.mean_power_db / 20.0)
    
    # Add thermal noise if specified
    if thermal_noise_db is not None:
        thermal_noise = _generate_thermal_noise(rp.n_ranges, rp.n_pulses, thermal_noise_db)
        clutter_td += thermal_noise
    
    return clutter_td, texture, speckle[:, -1]


def add_target_blob(signal_td: np.ndarray, tgt: Target, rp: RadarParams):
    """Add a target at tgt.rng_idx with given power and Doppler, spreading over multiple range bins based on size."""
    n = np.arange(rp.n_pulses)
    phase = np.exp(1j * 2.0 * np.pi * tgt.doppler_hz * n / rp.prf)
    # Fix: Use consistent dB to amplitude scaling
    amp_center = 10.0 ** (tgt.power / 20.0)
    
    # Get target size (default to 1 if not specified)
    target_size = getattr(tgt, 'size', 1)
    
    # Calculate range bins to fill
    center_idx = tgt.rng_idx
    half_size = (target_size - 1) // 2
    
    # Create power distribution across range bins
    for offset in range(-half_size, half_size + 1):
        idx = center_idx + offset
        if 0 <= idx < rp.n_ranges:
            # Reduce power for edge pixels to create realistic target profile
            if offset == 0:
                # Center pixel gets full power
                power_factor = 1.0
            else:
                # Edge pixels get reduced power (70% of center)
                power_factor = 0.7
            
            signal_td[idx] += amp_center * power_factor * phase


def compute_range_doppler(signal_td: np.ndarray, rp: RadarParams, cp: ClutterParams) -> np.ndarray:
    # Apply Hamming window along the slow-time (pulse) axis
    window = np.hamming(rp.n_pulses)
    signal_td_windowed = signal_td * window[np.newaxis, :]
    rd = _time_to_range_doppler(signal_td_windowed)
    if cp.bragg_offset_hz is not None:
        _inject_bragg(rd, rp.prf, cp.bragg_offset_hz, cp.bragg_width_hz, cp.bragg_power_rel)
    return rd