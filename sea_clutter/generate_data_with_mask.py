from tqdm import tqdm
from parameters import RadarParams, ClutterParams, SequenceParams, TargetType, get_clutter_params_for_sea_state, create_realistic_target, Target
from physics import add_target_blob, compute_range_doppler, simulate_sea_clutter
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_single_frame_with_targets_and_mask(
    rp: RadarParams,
    cp: ClutterParams,
    n_targets: int,
    random_roll: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single range-Doppler map with specified number of targets and corresponding binary mask."""
    
    # Generate sea clutter
    clutter_td, _, _ = simulate_sea_clutter(rp, cp)
    
    # Initialize binary mask (same size as RDM)
    target_mask = np.zeros((rp.n_ranges, rp.n_pulses), dtype=np.float32)

    if random_roll:
        # Apply random roll to simulate different clutter patterns
        roll_amount = random.randint(-1200, 1200)
    
    # Generate random targets if needed
    if n_targets > 0:
        max_range = rp.n_ranges * rp.range_resolution
        targets = [
            create_realistic_target(TargetType.FIXED, random.randint(1, max_range-1), rp) 
            for _ in range(n_targets)
        ]
        
        # Add each target to the clutter data and mark in mask
        for tgt in targets:
            simple_target = Target(
                rng_idx=tgt.rng_idx,
                doppler_hz=tgt.doppler_hz,
                power=tgt.power
            )
            add_target_blob(clutter_td, simple_target, rp)
            
            # Mark target location in binary mask
            # Convert Doppler frequency to bin index
            doppler_bin = int(tgt.doppler_hz / (rp.prf / rp.n_pulses)) + 64
            doppler_bin = np.clip(doppler_bin, 0, rp.n_pulses - 1)
            
            # Mark target in mask (you might want to expand this to a small blob)
            target_mask[tgt.rng_idx, doppler_bin] = 1.0
            
            # Optional: Create small blob around target location for better visibility
            blob_size = 1  # Adjust as needed
            # r_start = max(0, tgt.rng_idx - blob_size)
            # r_end = min(rp.n_ranges, tgt.rng_idx + blob_size + 1)
            d_start = max(0, doppler_bin - blob_size-1)
            d_end = min(rp.n_pulses, doppler_bin + blob_size+1)
            target_mask[tgt.rng_idx, d_start:d_end] = 1.0
    
    # Compute range-Doppler map
    rdm = compute_range_doppler(clutter_td, rp, cp)

    rdm = np.roll(rdm, shift=roll_amount, axis=1) if random_roll else rdm
    target_mask = np.roll(target_mask, shift=roll_amount, axis=1) if random_roll else target_mask
    
    return rdm, target_mask


def generate_segmentation_dataset(
    samples_per_class: int = 2000,
    max_targets: int = 5,
    sea_state: int = 5,
    save_path: str = "data/sea_clutter_segmentation_dataset.pt"
) -> None:
    """
    Generate dataset for target segmentation with binary masks.
    
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
    
    print(f"Generating segmentation dataset with {samples_per_class} samples per class")
    print(f"Classes: 0 to {max_targets} targets")
    print(f"Sea state: {sea_state}")
    print(f"Range-Doppler map size: {rp.n_ranges} x {rp.n_pulses}")
    
    # Storage for data and binary masks
    all_images = []
    all_masks = []
    all_labels = []  # Keep labels for reference (number of targets)
    
    # Generate data for each class
    for n_targets in range(max_targets + 1):
        print(f"\nGenerating {samples_per_class} samples for {n_targets} targets...")
        
        class_images = []
        class_masks = []
        class_labels = []
        
        for i in tqdm(range(samples_per_class), desc=f"Class {n_targets}"):
            # Generate single RDM with target mask
            rdm, target_mask = generate_single_frame_with_targets_and_mask(rp, cp, n_targets)

            # Convert RDM to dB scale and normalize
            rdm = 20 * np.log10(np.abs(rdm) + 1e-10)  # Avoid log(0)
            rdm = (rdm - np.mean(rdm)) / np.std(rdm) + 1e-10

            # Store image, mask, and label
            class_images.append(rdm)
            class_masks.append(target_mask)
            class_labels.append(n_targets)
        
        all_images.extend(class_images)
        all_masks.extend(class_masks)
        all_labels.extend(class_labels)
    
    # Convert to numpy arrays
    images = np.array(all_images)  # Shape: (total_samples, n_ranges, n_doppler_bins)
    masks = np.array(all_masks)   # Shape: (total_samples, n_ranges, n_doppler_bins)
    labels = np.array(all_labels)  # Shape: (total_samples,)
    
    print(f"\nDataset generated!")
    print(f"Total samples: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Target pixels per class: {[np.sum(masks[labels == i]) / samples_per_class for i in range(max_targets + 1)]}")
    
    # Convert to PyTorch tensors
    images_tensor = torch.from_numpy(images).float()
    masks_tensor = torch.from_numpy(masks).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset dictionary
    dataset = {
        'images': images_tensor,
        'masks': masks_tensor,  # Binary target masks
        'labels': labels_tensor,  # Number of targets (for reference)
        'metadata': {
            'samples_per_class': samples_per_class,
            'max_targets': max_targets,
            'sea_state': sea_state,
            'n_ranges': rp.n_ranges,
            'n_doppler_bins': rp.n_pulses,
            'range_resolution': rp.range_resolution,
            'class_names': [f"{i}_targets" for i in range(max_targets + 1)],
            'dataset_type': 'segmentation'  # Indicate this is for segmentation
        }
    }
    
    # Save to file
    torch.save(dataset, save_path)
    print(f"\nDataset saved to: {save_path}")
    
    # Calculate file size
    total_size = (images_tensor.element_size() * images_tensor.nelement() + 
                  masks_tensor.element_size() * masks_tensor.nelement()) / (1024**2)
    print(f"File size: {total_size:.1f} MB")


def visualize_sample(dataset_path: str, sample_idx: int = None, figsize: tuple = (15, 5)):
    """
    Visualize a sample from the segmentation dataset.
    
    Args:
        dataset_path: Path to the saved dataset
        sample_idx: Index of sample to visualize (random if None)
        figsize: Figure size for the plot
    """
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = torch.load(dataset_path)
    
    images = dataset['images']
    masks = dataset['masks']
    labels = dataset['labels']
    metadata = dataset['metadata']
    
    print(f"Dataset info:")
    print(f"  Total samples: {len(images)}")
    print(f"  Image shape: {images.shape}")
    print(f"  Classes: {metadata['class_names']}")
    
    # Select sample
    if sample_idx is None:
        sample_idx = random.randint(0, len(images) - 1)
    
    sample_idx = min(sample_idx, len(images) - 1)
    
    # Get sample data
    image = images[sample_idx].numpy()
    mask = masks[sample_idx].numpy()
    label = labels[sample_idx].item()
    
    print(f"\nVisualizing sample {sample_idx}:")
    print(f"  Number of targets: {label}")
    print(f"  Target pixels: {np.sum(mask)}")
    print(f"  Image min/max: {image.min():.2f} / {image.max():.2f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Original Range-Doppler Map
    im1 = axes[0].imshow(image, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'Range-Doppler Map\n({label} targets)')
    axes[0].set_xlabel('Doppler Bin')
    axes[0].set_ylabel('Range Bin')
    plt.colorbar(im1, ax=axes[0], label='Normalized dB')
    
    # Plot 2: Binary Target Mask
    im2 = axes[1].imshow(mask, aspect='auto', origin='lower', cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Target Mask\n(Binary)')
    axes[1].set_xlabel('Doppler Bin')
    axes[1].set_ylabel('Range Bin')
    plt.colorbar(im2, ax=axes[1], label='Target Presence')
    
    # Plot 3: Overlay (RDM with target highlights)
    # Create overlay by combining RDM and mask
    overlay_image = image.copy()
    
    # Create custom colormap for overlay
    # Use the viridis colormap for the background and red for targets
    axes[2].imshow(image, aspect='auto', origin='lower', cmap='viridis', alpha=0.8)
    
    # Overlay targets in red
    target_overlay = np.ma.masked_where(mask == 0, mask)
    axes[2].imshow(target_overlay, aspect='auto', origin='lower', cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    
    axes[2].set_title('RDM with Target Overlay\n(Red = Targets)')
    axes[2].set_xlabel('Doppler Bin')
    axes[2].set_ylabel('Range Bin')
    
    plt.tight_layout()
    plt.show()
    
    return sample_idx, image, mask, label


def visualize_class_distribution(dataset_path: str, figsize: tuple = (12, 8)):
    """
    Visualize the class distribution and show sample images from each class.
    
    Args:
        dataset_path: Path to the saved dataset
        figsize: Figure size for the plot
    """
    
    # Load dataset
    dataset = torch.load(dataset_path)
    labels = dataset['labels']
    metadata = dataset['metadata']
    
    # Count samples per class
    unique_labels, counts = torch.unique(labels, return_counts=True)
    
    # Create figure
    fig, axes = plt.subplots(2, len(unique_labels), figsize=figsize)
    if len(unique_labels) == 1:
        axes = axes.reshape(2, 1)
    
    print("Class distribution:")
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        print(f"  {label} targets: {count} samples")
        
        # Find first sample of this class
        class_indices = torch.where(labels == label)[0]
        sample_idx = class_indices[0]
        
        image = dataset['images'][sample_idx].numpy()
        mask = dataset['masks'][sample_idx].numpy()
        
        # Plot RDM
        axes[0, i].imshow(image, aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'{label} targets\nRDM')
        axes[0, i].set_xlabel('Doppler Bin')
        if i == 0:
            axes[0, i].set_ylabel('Range Bin')
        
        # Plot mask
        axes[1, i].imshow(mask, aspect='auto', origin='lower', cmap='Reds', vmin=0, vmax=1)
        axes[1, i].set_title('Target Mask')
        axes[1, i].set_xlabel('Doppler Bin')
        if i == 0:
            axes[1, i].set_ylabel('Range Bin')
    
    plt.tight_layout()
    plt.show()


def interactive_visualization(dataset_path: str):
    """
    Interactive visualization that lets you browse through samples.
    """
    
    dataset = torch.load(dataset_path)
    total_samples = len(dataset['images'])
    
    print(f"Interactive visualization mode")
    print(f"Total samples: {total_samples}")
    print("Commands:")
    print("  Enter sample index (0-{}) or 'r' for random or 'q' to quit".format(total_samples-1))
    
    while True:
        try:
            user_input = input("\nEnter sample index or command: ").strip().lower()
            
            if user_input == 'q' or user_input == 'quit':
                break
            elif user_input == 'r' or user_input == 'random':
                sample_idx = None
            else:
                sample_idx = int(user_input)
                if sample_idx < 0 or sample_idx >= total_samples:
                    print(f"Index out of range. Please enter 0-{total_samples-1}")
                    continue
            
            visualize_sample(dataset_path, sample_idx)
            
        except ValueError:
            print("Invalid input. Please enter a number, 'r', or 'q'")
        except KeyboardInterrupt:
            break
    
    print("Visualization ended.")


# Example usage
if __name__ == "__main__":
    # for sea_state in [1, 3, 5, 7, 9]:
    #     generate_segmentation_dataset(
    #         samples_per_class=2000,
    #         max_targets=10,
    #         sea_state=sea_state,
    #         save_path=f'data/sea_clutter_segmentation_{sea_state}_state.pt'
    #     )
    # generate_segmentation_dataset(
    #     samples_per_class=1000,
    #     max_targets=10,
    #     sea_state=5,
    #     save_path='data/sea_clutter_segmentation_roll_RCS.pt'
    # )


    # Path to your dataset
    dataset_path = "data/sea_clutter_segmentation_roll_RCS.pt"
    
    # Visualize a random sample
    print("=== Single Sample Visualization ===")
    visualize_sample(dataset_path)
    
    # Visualize class distribution
    print("\n=== Class Distribution Visualization ===")
    visualize_class_distribution(dataset_path)
    
    # Interactive mode (uncomment to use)
    print("\n=== Interactive Mode ===")
    interactive_visualization(dataset_path)