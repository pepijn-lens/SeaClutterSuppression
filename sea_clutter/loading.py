import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([
            f for f in os.listdir(data_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(path)
        return data['rdm'], data['mask'], data['metadata']

def visualize_sample(rdm, mask, metadata, save_path=None):
    """
    Visualize a sequence of Range-Doppler maps and ground truth masks.
    """
    rdm_np = rdm.numpy()
    mask_np = mask.numpy()
    
    n_frames = rdm_np.shape[0]
    
    fig, axes = plt.subplots(3, n_frames, figsize=(4 * n_frames, 10))
    
    for i in range(n_frames):
        # Convert RDM to dB scale for visibility
        rdm_db = 10 * np.log10(np.maximum(np.abs(rdm_np[i])**2, 1e-15))
        
        # 1. RDM
        ax1 = axes[0, i]
        im1 = ax1.imshow(rdm_db, aspect='auto', cmap='viridis')
        ax1.set_title(f'RDM Frame {i}')
        ax1.set_xlabel('Doppler')
        ax1.set_ylabel('Range')
        fig.colorbar(im1, ax=ax1)

        # 2. Mask
        ax2 = axes[1, i]
        im2 = ax2.imshow(mask_np[i], aspect='auto', cmap='gray')
        ax2.set_title(f'Mask Frame {i}')
        ax2.set_xlabel('Doppler')
        ax2.set_ylabel('Range')

        # 3. Overlay
        ax3 = axes[2, i]
        im3 = ax3.imshow(rdm_db, aspect='auto', cmap='viridis')
        ax3.contour(mask_np[i], levels=[0.5], colors='red', linewidths=2)
        ax3.set_title(f'Overlay Frame {i}')
        ax3.set_xlabel('Doppler')
        ax3.set_ylabel('Range')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# Example usage
if __name__ == "__main__":
    dataset = SegmentationDataset("sea_clutter/segmentation_dataset/pt_dataset")
    rdm, mask, metadata = dataset[192]  # Load first sample
    visualize_sample(rdm, mask, metadata)
