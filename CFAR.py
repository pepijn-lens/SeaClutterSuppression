import torch
import numpy as np
import matplotlib.pyplot as plt
from load_segmentation_data import RadarSegmentationDataset

def ca_cfar_2d(
    radar_image: torch.Tensor,
    guard_cells: int = 1,
    training_cells: int = 8,
    pfa: float = 1e-10,
    threshold_multiplier: float = 1.0  # Add this parameter
) -> torch.Tensor:
    """
    Apply 2D CA-CFAR on a normalized radar image.
    
    Args:
        radar_image (torch.Tensor): Input radar image of shape (1, H, W) or (H, W)
        guard_cells (int): Number of guard cells around the CUT in both dimensions
        training_cells (int): Number of training cells in both dimensions
        pfa (float): Desired probability of false alarm
        threshold_multiplier (float): Additional multiplier to raise/lower threshold
    
    Returns:
        torch.Tensor: Binary detection map of the same size as input image
    """
    if radar_image.dim() == 3:
        radar_image = radar_image.squeeze(0)  # (H, W)

    H, W = radar_image.shape
    detection_map = torch.zeros_like(radar_image)

    total_train = (2 * training_cells + 2 * guard_cells + 1)**2 - (2 * guard_cells + 1)**2

    # Compute threshold scaling factor for CA-CFAR
    alpha = total_train * (pfa ** (-1 / total_train) - 1)
    
    # Apply additional threshold multiplier
    alpha *= threshold_multiplier

    padded = torch.nn.functional.pad(radar_image, (training_cells + guard_cells,) * 4, mode='constant', value=-np.inf)

    for i in range(H):
        for j in range(W):
            i_p = i + training_cells + guard_cells
            j_p = j + training_cells + guard_cells

            window = padded[i_p - training_cells - guard_cells: i_p + training_cells + guard_cells + 1,
                            j_p - training_cells - guard_cells: j_p + training_cells + guard_cells + 1]

            # Mask out guard cells and CUT
            cut_area = padded[i_p - guard_cells: i_p + guard_cells + 1,
                              j_p - guard_cells: j_p + guard_cells + 1]
            noise_level = (window.sum() - cut_area.sum()) / total_train

            threshold = alpha * noise_level
            if radar_image[i, j] > threshold:
                detection_map[i, j] = 1

    return detection_map

def test_cfar_parameters(dataset, sample_idx=0):
    """Test different CFAR parameters to find optimal settings"""
    sample = dataset.get_sample_with_info(sample_idx)
    image = sample['image']
    mask_gt = sample['mask']
    
    # Test different parameter combinations
    pfa_values = [1e-6, 1e-8, 1e-10, 1e-12]
    threshold_multipliers = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    fig, axes = plt.subplots(len(pfa_values), len(threshold_multipliers), figsize=(20, 16))
    
    for i, pfa in enumerate(pfa_values):
        for j, mult in enumerate(threshold_multipliers):
            detection = ca_cfar_2d(image, pfa=pfa, threshold_multiplier=mult)
            
            axes[i, j].imshow(detection.numpy(), cmap='Greens')
            axes[i, j].set_title(f'PFA={pfa:.0e}, Mult={mult}')
            axes[i, j].axis('off')
    
    plt.suptitle('CFAR Parameter Tuning')
    plt.tight_layout()
    plt.show()
    
    # Show original and ground truth for reference
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(image.squeeze().numpy(), cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(mask_gt.squeeze().numpy(), cmap='hot')
    axs[1].set_title("Ground Truth")
    for ax in axs:
        ax.axis('off')
    plt.show()

# Updated example usage with parameter testing
def example_visualize(dataset, sample_idx=0, pfa=1e-10, threshold_multiplier=1.0):
    sample = dataset.get_sample_with_info(sample_idx)
    image = sample['image']  # Shape: (1, H, W)
    mask_gt = sample['mask']

    detection = ca_cfar_2d(image, pfa=pfa, threshold_multiplier=threshold_multiplier)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.squeeze().numpy(), cmap='gray')
    axs[0].set_title("Radar Image")

    axs[1].imshow(mask_gt.squeeze().numpy(), cmap='hot')
    axs[1].set_title("Ground Truth Mask")

    axs[2].imshow(detection.numpy(), cmap='Greens')
    axs[2].set_title(f"CFAR Detection (PFA={pfa:.0e}, Mult={threshold_multiplier})")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load test dataset
    dataset_path = '/Users/pepijnlens/Documents/transformers/data/sea_clutter_segmentation_highSCR.pt'
    test_dataset = RadarSegmentationDataset(
        dataset_path=dataset_path,
        split='test',
        normalize=False
    )

    # Test different parameters
    print("Testing CFAR parameters...")
    test_cfar_parameters(test_dataset, sample_idx=2889)
    
    # Example with higher threshold
    print("Example with higher threshold multiplier...")
    example_visualize(test_dataset, sample_idx=2889, pfa=1e-8, threshold_multiplier=5.0)
