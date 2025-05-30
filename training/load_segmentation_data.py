import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RadarSegmentationDataset(Dataset):
    """
    PyTorch Dataset class for radar target segmentation with U-Net.
    
    This dataset loads Range-Doppler Maps and their corresponding binary target masks.
    """
    
    def __init__(
        self, 
        dataset_path: str,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        add_channel_dim: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to the saved PyTorch dataset file
            split: One of 'train', 'val', 'test', or 'all'
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation  
            test_ratio: Fraction of data for testing
            random_state: Random seed for reproducible splits
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to masks
            normalize: Whether to normalize images to [0, 1] range
            add_channel_dim: Whether to add channel dimension for U-Net (C, H, W)
        """
        
        assert split in ['train', 'val', 'test', 'all'], f"Invalid split: {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.split = split
        self.add_channel_dim = add_channel_dim
        
        # Load the dataset
        print(f"Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, map_location='cpu')
        
        self.images = dataset['images']
        self.masks = dataset['masks']
        self.labels = dataset['labels']  # Number of targets (for reference)
        self.metadata = dataset['metadata']
        
        # Store original statistics for potential denormalization
        self.original_mean = self.images.mean().item()
        self.original_std = self.images.std().item()
        
        # print(f"Loaded dataset with {len(self.images)} samples")
        # print(f"Image shape: {self.images.shape}")
        # print(f"Mask shape: {self.masks.shape}")
        
        # Create train/val/test splits
        if split != 'all':
            self._create_splits(train_ratio, val_ratio, test_ratio, random_state)
        
        # print(f"Using {split} split with {len(self.images)} samples")

    
    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int):
        """Create train/validation/test splits."""
        
        n_samples = len(self.images)
        indices = np.arange(n_samples)
        
        # First split: separate train from (val + test)
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=self.labels.numpy()  # Stratify by number of targets
        )
        
        # Second split: separate val from test
        if val_ratio > 0 and test_ratio > 0:
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_state + 1,
                stratify=self.labels[temp_indices].numpy()
            )
        elif val_ratio > 0:
            val_indices = temp_indices
            test_indices = np.array([])
        else:
            val_indices = np.array([])
            test_indices = temp_indices
        
        # Select the appropriate split
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = val_indices
        else:  # test
            split_indices = test_indices
        
        # Filter data based on split
        self.images = self.images[split_indices]
        self.masks = self.masks[split_indices]
        self.labels = self.labels[split_indices]
        
        # print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, mask) tensors
        """
        
        # Get image and mask
        image = self.images[idx].clone()  # Shape: (H, W)
        mask = self.masks[idx].clone()    # Shape: (H, W)
        
        # Add channel dimension if requested (for U-Net: C=1, H, W)
        if self.add_channel_dim:
            image = image.unsqueeze(0)  # Shape: (1, H, W)
            mask = mask.unsqueeze(0)    # Shape: (1, H, W)

        return image, mask
    
    def get_sample_with_info(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with additional information.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image, mask, label, and metadata
        """
        
        image, mask = self.__getitem__(idx)
        
        return {
            'image': image,
            'mask': mask,
            'n_targets': self.labels[idx].item(),
            'idx': idx,
            'split': self.split
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of target counts in the current split."""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        return {label.item(): count.item() for label, count in zip(unique_labels, counts)}
    
    def get_dataset_stats(self) -> Dict[str, float]:
        """Get statistics about the dataset."""
        return {
            'image_mean': self.images.mean().item(),
            'image_std': self.images.std().item(),
            'image_min': self.images.min().item(),
            'image_max': self.images.max().item(),
            'mask_mean': self.masks.mean().item(),  # Fraction of target pixels
            'total_target_pixels': self.masks.sum().item(),
        }


def create_data_loaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_path: Path to the dataset file
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        train_ratio, val_ratio, test_ratio: Data split ratios
        random_state: Random seed for reproducible splits
        pin_memory: Whether to pin memory for faster GPU transfer
        normalize: Whether to normalize images
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets for each split
    train_dataset = RadarSegmentationDataset(
        dataset_path=dataset_path,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    
    val_dataset = RadarSegmentationDataset(
        dataset_path=dataset_path,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    
    test_dataset = RadarSegmentationDataset(
        dataset_path=dataset_path,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # print(f"\nData Loaders created:")
    # print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    # print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    # print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    # # Print class distributions
    # print(f"\nClass distributions:")
    # print(f"  Train: {train_dataset.get_class_distribution()}")
    # print(f"  Val:   {val_dataset.get_class_distribution()}")
    # print(f"  Test:  {test_dataset.get_class_distribution()}")
    
    return train_loader, val_loader, test_loader

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

# Example usage and testing
if __name__ == "__main__":
    
    visualize_sample("data/sea_clutter_segmentation_lowSCR.pt", sample_idx=3000)

    # Example 1: Create individual dataset
    dataset_path = "data/sea_clutter_segmentation_lowSCR.pt"
    
    print("=== Testing Dataset Class ===")
    dataset = RadarSegmentationDataset(dataset_path, split='train')
    
    # Test getting a sample
    image, mask = dataset[0]
    print(f"Sample shape - Image: {image.shape}, Mask: {mask.shape}")
    
    # Get dataset statistics
    stats = dataset.get_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    # Example 2: Create data loaders
    print("\n=== Creating Data Loaders ===")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=8,
        num_workers=2
    )
    
    # Test batch loading
    print("\n=== Testing Batch Loading ===")
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images {images.shape}, Masks {masks.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
        print(f"  Target pixel ratio: {masks.mean():.4f}")
        if batch_idx == 2:  # Only show first few batches
            break