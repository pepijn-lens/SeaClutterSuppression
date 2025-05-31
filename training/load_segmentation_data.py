import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

class RadarSegmentationDataset(Dataset):
    """
    PyTorch Dataset class for radar target segmentation with U-Net using sequence data.
    
    This dataset loads Range-Doppler Map sequences and their corresponding binary target masks.
    """
    
    def __init__(
        self, 
        dataset_path: str,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        mask_strategy: str = 'last'  # 'middle', 'last', 'aggregate'
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
            mask_strategy: How to handle sequence masks:
                - 'middle': Use middle frame mask
                - 'last': Use last frame mask
                - 'aggregate': Combine all masks (logical OR)
        """
        
        assert split in ['train', 'val', 'test', 'all'], f"Invalid split: {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        assert mask_strategy in ['middle', 'last', 'aggregate'], f"Invalid mask_strategy: {mask_strategy}"
        
        self.split = split
        self.mask_strategy = mask_strategy
        
        # Load the dataset
        print(f"Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, map_location='cpu')
        
        # Check if this is sequence data or single frame data
        if 'sequences' in dataset:
            # Sequence data
            self.sequences = dataset['sequences']  # Shape: (N, n_frames, H, W)
            self.mask_sequences = dataset['mask_sequences']  # Shape: (N, n_frames, H, W)
            self.is_sequence = True
            self.n_frames = self.sequences.shape[1]
            print(f"Loaded sequence dataset with {self.n_frames} frames per sequence")
        else:
            # Single frame data - convert to sequence format for compatibility
            self.sequences = dataset['images'].unsqueeze(1)  # Add frame dimension
            self.mask_sequences = dataset['masks'].unsqueeze(1)
            self.is_sequence = False
            self.n_frames = 1
            print("Loaded single frame dataset, converted to sequence format")
        
        self.labels = dataset['labels']  # Number of targets (for reference)
        self.metadata = dataset['metadata']
        
        # Store original statistics for potential denormalization
        self.original_mean = self.sequences.mean().item()
        self.original_std = self.sequences.std().item()
        
        print(f"Loaded dataset with {len(self.sequences)} samples")
        print(f"Sequence shape: {self.sequences.shape}")
        print(f"Mask sequence shape: {self.mask_sequences.shape}")
        
        # Create train/val/test splits
        if split != 'all':
            self._create_splits(train_ratio, val_ratio, test_ratio, random_state)
        
        print(f"Using {split} split with {len(self.sequences)} samples")

    
    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int):
        """Create train/validation/test splits."""
        
        n_samples = len(self.sequences)
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
        self.sequences = self.sequences[split_indices]
        self.mask_sequences = self.mask_sequences[split_indices]
        self.labels = self.labels[split_indices]
        
        print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence_as_channels, target_mask) tensors
        """
        
        # Get sequence and mask sequence
        sequence = self.sequences[idx].clone()  # Shape: (n_frames, H, W)
        mask_sequence = self.mask_sequences[idx].clone()  # Shape: (n_frames, H, W)
        
        # Use sequence frames as channels: (n_frames, H, W) -> this becomes (C, H, W) for U-Net
        image = sequence  # Shape: (3, H, W) for 3 frames
        
        # Handle mask based on strategy
        if self.mask_strategy == 'middle':
            # Use middle frame mask
            middle_idx = self.n_frames // 2
            mask = mask_sequence[middle_idx]  # Shape: (H, W)
        elif self.mask_strategy == 'last':
            # Use last frame mask
            mask = mask_sequence[-1]  # Shape: (H, W)
        elif self.mask_strategy == 'aggregate':
            # Aggregate all masks (logical OR)
            mask = torch.clamp(mask_sequence.sum(dim=0), 0, 1)  # Shape: (H, W)
        
        # Add channel dimension to mask for U-Net
        mask = mask.unsqueeze(0)  # Shape: (1, H, W)

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
            'sequence': self.sequences[idx],  # Full sequence
            'mask_sequence': self.mask_sequences[idx],  # Full mask sequence
            'n_targets': self.labels[idx].item(),
            'idx': idx,
            'split': self.split,
            'n_frames': self.n_frames,
            'mask_strategy': self.mask_strategy
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of target counts in the current split."""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        return {label.item(): count.item() for label, count in zip(unique_labels, counts)}
    
    def get_dataset_stats(self) -> Dict[str, float]:
        """Get statistics about the dataset."""
        return {
            'sequence_mean': self.sequences.mean().item(),
            'sequence_std': self.sequences.std().item(),
            'sequence_min': self.sequences.min().item(),
            'sequence_max': self.sequences.max().item(),
            'mask_mean': self.mask_sequences.mean().item(),  # Fraction of target pixels
            'total_target_pixels': self.mask_sequences.sum().item(),
            'n_frames': self.n_frames,
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
    mask_strategy: str = 'last'
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
        mask_strategy: How to handle sequence masks ('middle', 'last', 'aggregate')
        
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
        mask_strategy=mask_strategy
    )
    
    val_dataset = RadarSegmentationDataset(
        dataset_path=dataset_path,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        mask_strategy=mask_strategy
    )
    
    test_dataset = RadarSegmentationDataset(
        dataset_path=dataset_path,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        mask_strategy=mask_strategy
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
    
    print(f"\nData Loaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"  Mask strategy: {mask_strategy}")
    print(f"  Frames per sequence: {train_dataset.n_frames}")
    
    return train_loader, val_loader, test_loader

def visualize_sequence_sample(dataset_path: str, sample_idx: int = None):
    """
    Visualize a sequence sample from the segmentation dataset.
    
    Args:
        dataset_path: Path to the saved dataset
        sample_idx: Index of sample to visualize (random if None)
        figsize: Figure size for the plot
    """
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = torch.load(dataset_path)
    
    # Check if sequence data
    if 'sequences' in dataset:
        sequences = dataset['sequences']
        mask_sequences = dataset['mask_sequences']
        labels = dataset['labels']
        metadata = dataset['metadata']
        n_frames = sequences.shape[1]
    else:
        print("This appears to be single-frame data, not sequence data")
        return
    
    print(f"Dataset info:")
    print(f"  Total samples: {len(sequences)}")
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Frames per sequence: {n_frames}")
    
    # Select sample
    if sample_idx is None:
        sample_idx = random.randint(0, len(sequences) - 1)
    
    sample_idx = min(sample_idx, len(sequences) - 1)
    
    # Get sample data
    sequence = sequences[sample_idx].numpy()  # Shape: (n_frames, H, W)
    mask_sequence = mask_sequences[sample_idx].numpy()  # Shape: (n_frames, H, W)
    label = labels[sample_idx].item()
    
    print(f"\nVisualizing sample {sample_idx}:")
    print(f"  Number of targets: {label}")
    print(f"  Sequence shape: {sequence.shape}")
    
    # Create figure with subplots: 2 rows (RDM, masks) x n_frames columns
    fig, axes = plt.subplots(2, n_frames)
    if n_frames == 1:
        axes = axes.reshape(2, 1)
    
    for frame_idx in range(n_frames):
        rdm = sequence[frame_idx]
        mask = mask_sequence[frame_idx]
        
        # Plot RDM
        im1 = axes[0, frame_idx].imshow(rdm, aspect='auto', origin='lower', cmap='viridis')
        axes[0, frame_idx].set_title(f'Frame {frame_idx + 1}\nRDM ({label} targets)')
        axes[0, frame_idx].set_xlabel('Doppler Bin')
        if frame_idx == 0:
            axes[0, frame_idx].set_ylabel('Range Bin')
        
        # Plot mask
        im2 = axes[1, frame_idx].imshow(mask, aspect='auto', origin='lower', cmap='Reds', vmin=0, vmax=1)
        axes[1, frame_idx].set_title(f'Frame {frame_idx + 1}\nTarget Mask')
        axes[1, frame_idx].set_xlabel('Doppler Bin')
        if frame_idx == 0:
            axes[1, frame_idx].set_ylabel('Range Bin')
    
    plt.tight_layout()
    plt.show()
    
    return sample_idx, sequence, mask_sequence, label

def visualize_sample(dataset_path: str, sample_idx: int = None):
    """
    Visualize a sample - automatically detects if sequence or single frame data.
    """
    dataset = torch.load(dataset_path)
    
    if 'sequences' in dataset:
        return visualize_sequence_sample(dataset_path, sample_idx)
    else:
        # Use original single-frame visualization
        return visualize_single_frame_sample(dataset_path, sample_idx)

def visualize_single_frame_sample(dataset_path: str, sample_idx: int = None):
    """Original single-frame visualization function."""
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
    fig, axes = plt.subplots(1, 3)
    
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
    axes[2].imshow(image, aspect='auto', origin='lower', cmap='viridis', alpha=0.8)
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
    
    # Test with sequence data
    dataset_path = "data/sea_clutter_segmentation_sequences.pt"
    
    print("=== Testing Sequence Dataset Class ===")
    dataset = RadarSegmentationDataset(dataset_path, split='train', mask_strategy='middle')
    
    # Test getting a sample
    image, mask = dataset[3000]
    print(f"Sample shape - Image: {image.shape}, Mask: {mask.shape}")
    
    # Get dataset statistics
    stats = dataset.get_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    # Test data loaders
    print("\n=== Creating Data Loaders ===")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=16,
        num_workers=0,
        mask_strategy='last'
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
    
    # Visualize a sequence sample
    print("\n=== Visualizing Sequence Sample ===")
    visualize_sample(dataset_path, sample_idx=3000)