import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from typing import Optional


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
        visualize: Optional[bool] = False
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
        """
        
        assert split in ['train', 'val', 'test', 'all'], f"Invalid split: {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.split = split
        self.visualize = visualize
        
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
            Tuple of (image, target_mask) tensors
        """
        
        # Get sequence and mask sequence
        sequence = self.sequences[idx].clone()  # Shape: (n_frames, H, W)
        mask_sequence = self.mask_sequences[idx].clone()  # Shape: (n_frames, H, W)

        if not self.visualize:
            mask_sequence = mask_sequence[-1]  # Take the last frame mask by default

        # For sequence data, use all frames as channels
        if self.is_sequence:
            # Use all frames as channels: (n_frames, H, W) = (3, H, W)
            image = sequence  # Shape: (3, H, W) for 3-frame sequences
            mask = mask_sequence.unsqueeze(0)  # Shape: (1, H, W) - single channel mask
        else:
            # Single frame data, add channel dimension
            image = sequence[0].unsqueeze(0)  # Shape: (1, H, W) - single channel
            mask = mask[0].unsqueeze(0)  # Shape: (1, H, W) - single channel
        
        # Keep mask as single channel without extra dimension
        # mask stays as (H, W) - U-Net expects this for binary segmentation
        return image, mask

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
    
    print(f"\nData Loaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"  Frames per sequence: {train_dataset.n_frames}")
    
    return train_loader, val_loader, test_loader


def visualize_sample(
    dataset: RadarSegmentationDataset, 
    idx: int = 0, 
    figsize: Tuple[int, int] = (15, 8)
) -> None:
    """
    Visualize a single sample from the dataset showing all frames with target mask overlay.
    
    Args:
        dataset: RadarSegmentationDataset instance
        idx: Index of sample to visualize
        figsize: Figure size for the plot
    """
    
    # Get a sample
    image, mask_sequence = dataset[idx]  # mask_sequence: (n_frames, H, W)
    frame_count = mask_sequence.shape[0]
    fig, axes = plt.subplots(1, frame_count + 1, figsize=figsize)
    for i in range(frame_count):
        axes[i].imshow(image[i], cmap='viridis', aspect='auto')
        mask_overlay = mask_sequence[i].squeeze()
        mask_overlay = np.ma.masked_where(mask_overlay == 0, mask_overlay)
        axes[i].imshow(mask_overlay, cmap='Reds', alpha=0.7, aspect='auto')
        axes[i].set_title(f'Frame {i+1} + Mask')
        axes[i].axis('off')
    axes[frame_count].imshow(mask_sequence[-1].squeeze(), cmap='Reds', aspect='auto')
    axes[frame_count].set_title('Target Mask Only')
    axes[frame_count].axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Sample {idx}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask sequence shape: {mask_sequence.shape}")

if __name__ == "__main__":
    visualize_sample(RadarSegmentationDataset('single_frame_test.pt', visualize=True), idx=122)