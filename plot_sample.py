import torch
import numpy as np

def plot_sample(index=5000):
    # Load the data
    data = torch.load('local_data/random3.pt')

    # Access the data
    sequences = data['sequences']
    labels = data['labels']
    metadata = data['metadata']
    
    # # Get radar parameters from metadata
    # n_ranges = metadata['n_ranges']
    # n_doppler_bins = metadata['n_doppler_bins']
    # range_resolution = 1  # in meters
    # prf = metadata['prf']  # Pulse Repetition Frequency in Hz
    # carrier_wavelength = metadata['carrier_wavelength']  # in meters

    # Get the sample at the specified index
    rd_map = sequences[index]  # Shape: [3, 128, 128] - 3 frames, 128 range bins, 128 Doppler bins
    label = labels[index]

    # # Calculate range and velocity axes
    # # Range axis: from 0 to max_range based on range resolution
    # max_range = n_ranges * range_resolution
    # range_axis = np.linspace(0, max_range, n_ranges)
    
    # # Velocity axis: based on PRF and wavelength
    # # Maximum unambiguous velocity = λ * PRF / 4
    # max_velocity = carrier_wavelength * prf / 4
    # velocity_axis = np.linspace(-max_velocity, max_velocity, n_doppler_bins)

    import matplotlib.pyplot as plt
    
    # Get the target mask for comparison
    masks = data['masks']
    target_mask = masks[index]  # Shape: [128, 128]
    
    # Create subplot for first frame and target mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first frame RD map
    ax1 = axes[0]
    im1 = ax1.imshow(rd_map[0].squeeze(), 
                    cmap='viridis', 
                    aspect='auto',
                    vmin=0, vmax=40,
                    # extent=[velocity_axis[0], velocity_axis[-1], 
                    #        range_axis[-1], range_axis[0]]
                    )  # Note: range is flipped for proper display
    
    ax1.set_xlabel('Velocity (m/s)')
    ax1.set_ylabel('Range (m)')
    ax1.set_title(f'RD Map - Frame 1\nLabel: {label.item()} targets')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    # Plot target mask
    ax2 = axes[1]
    im2 = ax2.imshow(target_mask[0].squeeze(),  # Use first frame of target mask
                    cmap='Reds', 
                    aspect='auto',
                    # extent=[velocity_axis[0], velocity_axis[-1], 
                    #        range_axis[-1], range_axis[0]]
                    )  # Note: range is flipped for proper display
    
    ax2.set_xlabel('Velocity (m/s)')
    ax2.set_ylabel('Range (m)')
    ax2.set_title(f'Target Mask\n{label.item()} targets')
    plt.colorbar(im2, ax=ax2, label='Target Presence')
    
    plt.tight_layout()
    plt.suptitle(f'RD Map and Target Mask - Sample {index}', y=1.02)
    plt.show()
    
    # Print radar parameters for reference
    print(f"\nRadar Parameters:")
    # print(f"Range resolution: {range_resolution} m")
    # print(f"Max range: {max_range} m")
    # print(f"PRF: {prf} Hz")
    # print(f"Carrier wavelength: {carrier_wavelength} m")
    # print(f"Max unambiguous velocity: ±{max_velocity:.2f} m/s")
    print(f"Number of targets: {label.item()}")

if __name__ == "__main__":
    plot_sample()
