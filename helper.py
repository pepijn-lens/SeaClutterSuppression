import os 
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
from new_swin import radar_swin_t
from Classification import RadarDataset

c0 = 299792458

def plot_doppler(radar, rd_map):
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
        # extent=[velocity_bins[0], velocity_bins[-1], range_bins[-1], range_bins[0]],  # Set axis limits
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


def plot_attention(attn_map, head=0, window_idx=0):
    """
    Plot attention map for a single head and window.
    attn_map: (B, H, num_windows, win², win²)
    """
    # Remove batch dimension
    attn = attn_map[0, head, window_idx]  # (win², win²)

    plt.figure(figsize=(6, 5))
    plt.imshow(attn.detach().cpu(), cmap='viridis')
    plt.colorbar()
    plt.title(f'Head {head} - Window {window_idx}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

# ...existing code...

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "optuna_trial_6-20dBRCStraining.pt"
    dataset_name = "20dB_RCS"

    model = radar_swin_t(
        in_channels=1,
        num_classes=6,
        hidden_dim=128,
        window_size=(2, 8),
        layers=6,
        heads=4,
        head_dim=32,
        patch_size=8
    ).to(device)
    model.load_state_dict(torch.load(f"optuna/{model_name}", map_location=device))
    model.eval()
    # print(model)

    # Load dataset and sample
    dataset = RadarDataset(data_path=f"data/{dataset_name}.pt")
    img, label = dataset[23000]
    # plt.figure()
    # plt.imshow(img[0].squeeze(0).numpy())
    # plt.savefig("attention_maps/sample_image.png")
    # plt.show()
    img = img.unsqueeze(0).to(device)  # (1, 1, 64, 512)

    # Hook to capture attention maps
    attention_maps = []

    def hook_fn(module, input, output):
        # output: (out, attn)
        if isinstance(output, tuple):
            attention_maps.append(output[1].detach().cpu())

    window_attention = model.stage1.layers[2][0].attention_block.fn.fn
    handle = window_attention.register_forward_hook(hook_fn)

    orig_forward = window_attention.forward
    def forward_with_attention(x, **kwargs):
        return orig_forward(x, return_attention=True, **kwargs)
    window_attention.forward = forward_with_attention

    try:
        with torch.no_grad():
            _ = model(img)
    finally:
        window_attention.forward = orig_forward
        handle.remove()

    # Save all attention maps (for all heads and windows)
    if attention_maps:
        attn = attention_maps[0]  # shape: (B, heads, windows, N, N)
        dir ="attention_maps/layer3/"
        os.makedirs(dir, exist_ok=True)
        num_heads = attn.shape[1]
        num_windows = attn.shape[2]
        for head in range(num_heads):
            for window_idx in range(num_windows):
                attn_img = attn[0, head, window_idx].numpy()
                plt.figure(figsize=(6, 5))
                plt.imshow(attn_img, cmap="viridis")
                plt.colorbar()
                plt.title(f"Head {head} - Window {window_idx}")
                plt.xlabel('Key Tokens')
                plt.ylabel('Query Tokens')
                plt.tight_layout()
                plt.savefig(f"{dir}attn_head{head}_window{window_idx}.png")
                plt.close()
        print(f"Saved {num_heads * num_windows} attention maps to 'attention_maps/'")
    else:
        print("No attention maps captured.")
# ...existing code...