#!/usr/bin/env python3
"""
Verify and visualize the generated classification dataset.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def verify_dataset(dataset_path: str = "sea_clutter_data.pt"):
    """Load and verify the generated dataset."""
    
    # Load dataset
    dataset = torch.load(dataset_path)
    images = dataset['images']
    labels = dataset['labels']
    metadata = dataset['metadata']
    
    print("Dataset Information:")
    print(f"  Total samples: {len(images)}")
    print(f"  Image shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Samples per class: {metadata['samples_per_class']}")
    print(f"  Max targets: {metadata['max_targets']}")
    print(f"  Sea state: {metadata['sea_state']}")
    print(f"  Range resolution: {metadata['range_resolution']:.1f}m")
    
    # Check class distribution
    unique, counts = torch.unique(labels, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  {cls} targets: {count} samples")
    
    # Show sample images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(6):  # Show one sample per class
        class_indices = torch.where(labels == i)[0]
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            img = images[sample_idx].numpy()
            
            axes[i].imshow(img, aspect='auto', origin='lower', cmap='viridis')
            axes[i].set_title(f'{i} targets')
            axes[i].set_xlabel('Doppler bin')
            axes[i].set_ylabel('Range bin')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify classification dataset")
    parser.add_argument("--dataset", type=str, default="/Users/pepijnlens/Documents/transformers/sea_clutter_classification_dataset.pt",
                        help="Path to dataset file")
    
    args = parser.parse_args()
    verify_dataset(args.dataset)