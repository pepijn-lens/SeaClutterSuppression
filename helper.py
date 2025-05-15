import os 
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 

c0 = 299792458

def plot_doppler(radar, rd_map):
    if not os.path.exists("heatmaps"):
        os.makedirs("heatmaps")

    # Calculate range bins (y-axis)
    num_samples = rd_map.shape[1]  # Fast-time samples (columns)
    max_range = (c0 * radar.PRI) / 2 * (num_samples/(radar.PRI*radar.fs)) # Max range (m)
    range_bins = np.linspace(0, max_range, num_samples)

    # Calculate velocity bins (x-axis)
    num_pulses = rd_map.shape[0] # Slow-time pulses (rows)
    lambda_c = c0 / radar.fc  # Wavelength (m)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_pulses, radar.PRI))  # Doppler frequencies (Hz)
    velocity_bins = doppler_freq * (lambda_c / 2) * (num_pulses/radar.n_pulses) # Velocity (m/s)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        rd_map.T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[-1], range_bins[0]],  # Set axis limits
        aspect='auto',
        interpolation='nearest',
    )

    # Add labels and title
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (m)')
    plt.title('Range-Doppler Heatmap')
    plt.colorbar(label='Magnitude (dB)')

    # Show plot
    plt.grid(False)
    plt.show()
    plt.close()


def calculate_resolution(radar, min_dopp, max_dopp):
    lambda_c = c0 / radar.fc  # wavelength

    # Doppler frequencies (Hz)
    f_doppler = np.fft.fftshift(np.fft.fftfreq(radar.n_pulses, d=radar.PRI))

    # Doppler velocities (m/s)
    v_doppler = f_doppler * lambda_c / 2

    # Find the indices for velocity between -25 and 25 m/s
    indices = np.where((v_doppler >= min_dopp) & (v_doppler <= max_dopp))[0]
    return indices

def measure_range_velocity(radar, rd_map):
    """
    Calculate and print the measured range and velocity from the range-Doppler map.
    Range and velocity bins are computed internally using radar parameters.
    
    Args:
        rd_map (torch.Tensor): 2D range-Doppler map (shape: [n_pulses, n_samples]).
    
    Returns:
        tuple: (measured_range, measured_velocity) in meters and m/s, respectively.
    """
    # Radar parameters
    num_samples = rd_map.shape[1]  # Fast-time samples (from rd_map columns)
    num_pulses = rd_map.shape[0]   # Slow-time pulses (from rd_map rows)
    c = 3e8  # Speed of light (m/s)
    lambda_ = c / radar.fc  # Wavelength (m)

    # Compute range bins (fast-time axis)
    max_range = (c * radar.PRI) / 2  # Max range (m)
    range_bins = np.linspace(0, max_range, num_samples)

    # Compute velocity bins (slow-time axis)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_pulses, radar.PRI))  # Doppler frequencies (Hz)
    velocity_bins = doppler_freq * (lambda_ / 2)  # Velocity (m/s)

    # Find the peak in the range-Doppler map
    rd_magnitude = torch.abs(rd_map)
    peak_idx = torch.argmax(rd_magnitude)  # Flattened index of the peak
    peak_row, peak_col = torch.div(peak_idx, rd_magnitude.shape[1], rounding_mode='floor'), peak_idx % rd_magnitude.shape[1]

    # Map peak indices to range and velocity
    measured_range = range_bins[peak_col.cpu().numpy()]
    measured_velocity = velocity_bins[peak_row.cpu().numpy()]

    # Print measured range and velocity
    print(f"Measured Range: {measured_range:.2f} m")
    print(f"Measured Velocity: {measured_velocity:.2f} m/s")

    return measured_range, measured_velocity
    

def load_burst(track, burst):
    file_path = f"data/track_{track}/burst_{burst}.parquet"
    data = pd.read_parquet(file_path)

    tensor_data = torch.tensor(data.values, dtype=torch.float32)
    tensor_data = tensor_data.view(64, 4096, 2)

    magnitude = 20 * torch.log10(torch.sqrt(tensor_data[..., 0] ** 2 + tensor_data[..., 1] ** 2) + 1e-10)

    magnitude = (magnitude - magnitude.mean()) / magnitude.std()
    magnitude = magnitude.unsqueeze(0)  # (1, 64, 4096)

    return magnitude

# Calculate global statistics across your dataset
def calculate_dataset_stats(track_numbers, burst_numbers):
    all_means = []
    all_stds = []
    
    for track in track_numbers:
        for burst in burst_numbers:
            try:
                # Load the data without normalization
                file_path = f"data/track_{track}/burst_{burst}.parquet"
                data = pd.read_parquet(file_path)
                tensor_data = torch.tensor(data.values, dtype=torch.float32)
                tensor_data = tensor_data.view(64, 4096, 2)
                magnitude = 20 * torch.log10(torch.sqrt(tensor_data[..., 0] ** 2 + tensor_data[..., 1] ** 2) + 1e-10)
                magnitude = (magnitude - magnitude.mean()) / magnitude.std()

                # Get stats
                mean = magnitude.mean().item()
                std = magnitude.std().item()
                
                all_means.append(mean)
                all_stds.append(std)
                
            except Exception as e:
                print(f"Error processing Track {track}, Burst {burst}: {e}")
    
    # Calculate average stats
    avg_mean = sum(all_means) / len(all_means) if all_means else 0
    avg_std = sum(all_stds) / len(all_stds) if all_stds else 1
    
    return avg_mean, avg_std

