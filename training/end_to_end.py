import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
import cv2

from end_to_end_helper import plot_performance_analysis, print_performance_report, evaluate_target_count_performance_from_loader, show_dataset_stats, analyze_single_sample


# Use your U-Net architecture from unet_training.py
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out_conv(x)

class ClusteringModule:
    """Clustering module to extract centroids from binary maps"""
    def __init__(self, min_area=3, eps=3, min_samples=1):
        self.min_area = min_area
        self.eps = eps
        self.min_samples = min_samples
    
    def extract_centroids(self, binary_map, threshold=0.5):
        """
        Extract centroids from binary map using connected components and DBSCAN
        
        Args:
            binary_map: numpy array or torch tensor of shape (H, W) with values in [0, 1]
            threshold: threshold for binarization
            
        Returns:
            centroids: list of (x, y) coordinates
        """
        # Convert to numpy if it's a tensor
        if torch.is_tensor(binary_map):
            binary_map = binary_map.cpu().numpy()
        
        # Threshold the binary map
        binary = (binary_map > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        centroids = []
        
        for label in range(1, num_labels):  # Skip background (label 0)
            # Get component mask
            component_mask = (labels == label)
            
            # Filter by minimum area
            if np.sum(component_mask) < self.min_area:
                continue
            
            # Get coordinates of pixels in this component
            coords = np.column_stack(np.where(component_mask))
            
            if len(coords) < self.min_samples:
                # If too few points, use simple centroid
                centroid_y, centroid_x = np.mean(coords, axis=0)
                centroids.append((centroid_x, centroid_y))
            else:
                # Use DBSCAN for sub-clustering within the component
                clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                cluster_labels = clustering.fit_predict(coords)
                
                # Get centroids of each cluster
                unique_labels = np.unique(cluster_labels)
                for cluster_label in unique_labels:
                    if cluster_label == -1:  # Skip noise points
                        continue
                    
                    cluster_coords = coords[cluster_labels == cluster_label]
                    centroid_y, centroid_x = np.mean(cluster_coords, axis=0)
                    centroids.append((centroid_x, centroid_y))
        
        return centroids

class EndToEndTargetDetector(nn.Module):
    """End-to-end model: Range-Doppler map -> Binary segmentation -> Target centroids"""
    def __init__(self, unet_weights_path=None, clustering_params=None, n_channels=3):
        super(EndToEndTargetDetector, self).__init__()
        
        # Initialize U-Net with your architecture
        self.unet = UNet(n_channels=n_channels, n_classes=1)  # Changed to n_classes=1 for segmentation
        
        # Load pre-trained weights if provided
        if unet_weights_path:
            self.unet.load_state_dict(torch.load(unet_weights_path, map_location='mps'))
        
        # Move model to MPS
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
        
        # Initialize clustering module
        clustering_params = clustering_params or {}
        self.clustering = ClusteringModule(**clustering_params)
        
    def forward(self, range_doppler_map):
        """
        Forward pass
        
        Args:
            range_doppler_map: torch.Tensor of shape (B, C, 128, 128) where C is n_channels
            
        Returns:
            batch_centroids: list of lists containing (x, y) coordinates for each sample
        """
        # Ensure input is on the correct device
        if range_doppler_map.device != self.device:
            range_doppler_map = range_doppler_map.to(self.device)
        
        # Get binary segmentation from U-Net (keep on MPS)
        binary_maps = torch.sigmoid(self.unet(range_doppler_map))  # Shape: (B, 1, 128, 128)
        
        # Only convert to CPU/NumPy for clustering operations
        batch_centroids = []
        
        for i in range(binary_maps.shape[0]):
            # Move single sample to CPU for clustering
            binary_map_cpu = binary_maps[i, 0].cpu().numpy()  # Shape: (128, 128)
            centroids = self.clustering.extract_centroids(binary_map_cpu)
            batch_centroids.append(centroids)
        
        return batch_centroids
    
    def predict_single(self, range_doppler_map):
        """
        Predict centroids for a single range-doppler map
        
        Args:
            range_doppler_map: numpy array of shape (128, 128) or (C, 128, 128) or torch.Tensor
            
        Returns:
            centroids: list of (x, y) coordinates
        """
        # Ensure input is torch tensor with correct shape
        if isinstance(range_doppler_map, np.ndarray):
            range_doppler_map = torch.from_numpy(range_doppler_map).float()
        
        if len(range_doppler_map.shape) == 2:
            # Single channel, add channel and batch dims
            range_doppler_map = range_doppler_map.unsqueeze(0).unsqueeze(0)
        elif len(range_doppler_map.shape) == 3:
            # Multi-channel, add batch dim
            range_doppler_map = range_doppler_map.unsqueeze(0)
        
        # Move to MPS device
        range_doppler_map = range_doppler_map.to(self.device)
        
        self.eval()
        with torch.no_grad():
            batch_centroids = self.forward(range_doppler_map)
        
        return batch_centroids[0]  # Return centroids for first (and only) sample


def comprehensive_evaluation():
    """Run a comprehensive evaluation of the end-to-end model using test data"""
    
    # Load dataset using the same method as in training
    dataset_path = "/Users/pepijnlens/Documents/transformers/data/sea_clutter_single_frame.pt"
    
    # Import the data loading function from your training file
    from training.load_segmentation_data import create_data_loaders
    
    # Create data loaders same as in training
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=16,
        mask_strategy='last',
    )
    
    # Use test loader for evaluation
    eval_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    eval_dataset_name = "test" if len(test_loader.dataset) > 0 else "validation"
    
    print(f"Using {eval_dataset_name} dataset for evaluation")
    print(f"Dataset size: {len(eval_loader.dataset)} samples")
    
    # Model setup - get actual channels from a sample
    sample_batch = next(iter(eval_loader))
    sample_image = sample_batch[0][0]  # First image from first batch
    
    # For sequence data with 3 frames, n_channels should be 3
    if len(sample_image.shape) == 3:  # (C, H, W)
        n_channels = sample_image.shape[0]
    else:  # Single channel
        n_channels = 1
    
    print(f"Detected {n_channels} input channels from sample shape: {sample_image.shape}")
    
    model_path = "/Users/pepijnlens/Documents/transformers/models/unet_single_frame.pt"
    
    # Create model with detected parameters
    model = EndToEndTargetDetector(
        unet_weights_path=model_path,
        n_channels=n_channels,
        clustering_params={
            'min_area': 3,
            'eps': 1,
            'min_samples': 1
        }
    )
    
    # Evaluate performance on test data
    print("Running comprehensive evaluation on test data...")
    results = evaluate_target_count_performance_from_loader(model, eval_loader, eval_dataset_name)
    
    # Print report
    print_performance_report(results)
    
    # Create plots
    plot_performance_analysis(results, save_path=f'target_count_performance_analysis_{eval_dataset_name}.png')
    
    return results

def interactive_sample_explorer():
    """
    Interactive method to explore samples from the dataset using the end-to-end target detector
    """
    print("Loading dataset and model...")
    
    # Load dataset
    dataset_path = "/Users/pepijnlens/Documents/seacluttersuppression/data/sea_clutter_single_frame.pt"
    from load_segmentation_data import create_data_loaders
    
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=1,  # Load one sample at a time
        mask_strategy='last',
    )
    
    # Use test loader for exploration, fallback to validation
    data_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    dataset_name = "test" if len(test_loader.dataset) > 0 else "validation"
    dataset = data_loader.dataset
    
    print(f"Loaded {dataset_name} dataset with {len(dataset)} samples")
    
    # Load model
    sample_batch = next(iter(data_loader))
    sample_image = sample_batch[0][0]
    n_channels = sample_image.shape[0] if len(sample_image.shape) == 3 else 1
    
    model_path = "/Users/pepijnlens/Documents/seacluttersuppression/models/unet_single_frame.pt"
    model = EndToEndTargetDetector(
        unet_weights_path=model_path,
        n_channels=n_channels,
        clustering_params={
            'min_area': 3,
            'eps': 1,
            'min_samples': 1
        }
    )
    
    print(f"Model loaded with {n_channels} input channels")
    print("\n" + "="*60)
    print("INTERACTIVE SAMPLE EXPLORER")
    print("="*60)
    print("Commands:")
    print("  Enter a number (0-{}) to analyze a specific sample".format(len(dataset)-1))
    print("  'random' or 'r' for a random sample")
    print("  'stats' or 's' to show dataset statistics")
    print("  'quit' or 'q' to exit")
    print("="*60)
    
    while True:
        try:
            user_input = input(f"\nEnter sample index (0-{len(dataset)-1}) or command: ").strip().lower()
            
            if user_input in ['quit', 'q', 'exit']:
                print("Goodbye!")
                break
            
            elif user_input in ['random', 'r']:
                sample_idx = np.random.randint(0, len(dataset))
                print(f"Randomly selected sample {sample_idx}")
            
            elif user_input in ['stats', 's']:
                show_dataset_stats(dataset)
                continue
                
            else:
                try:
                    sample_idx = int(user_input)
                    if sample_idx < 0 or sample_idx >= len(dataset):
                        print(f"Invalid index. Please enter a number between 0 and {len(dataset)-1}")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a number, 'random', 'stats', or 'quit'")
                    continue
            
            # Process the selected sample
            analyze_single_sample(model, dataset, sample_idx)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    # print("Running comprehensive end-to-end performance evaluation on 3-channel sequence data...")
    # # Run comprehensive evaluation on the 3-frame U-Net with the specified dataset
    # results = comprehensive_evaluation()
    print("Starting interactive sample explorer...")
    interactive_sample_explorer()