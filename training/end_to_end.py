import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt

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
            binary_map: numpy array of shape (H, W) with values in [0, 1]
            threshold: threshold for binarization
            
        Returns:
            centroids: list of (x, y) coordinates
        """
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
        self.unet = UNet(n_channels=n_channels, n_classes=n_channels)
        
        # Load pre-trained weights if provided
        if unet_weights_path:
            self.unet.load_state_dict(torch.load(unet_weights_path, map_location='mps'))
        
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
        # Get binary segmentation from U-Net
        binary_maps = torch.sigmoid(self.unet(range_doppler_map))  # Shape: (B, 1, 128, 128)
        
        # Convert to numpy and extract centroids for each sample in batch
        batch_centroids = []
        binary_maps_np = binary_maps.detach().to('mps')
        
        for i in range(binary_maps_np.shape[0]):
            binary_map = binary_maps_np[i, 0]  # Shape: (128, 128)
            centroids = self.clustering.extract_centroids(binary_map)
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

def count_ground_truth_targets(ground_truth_mask, min_area=3):
    """
    Count the number of targets in ground truth mask using connected components
    
    Args:
        ground_truth_mask: Binary mask with ground truth targets
        min_area: Minimum area to consider as a valid target
    
    Returns:
        int: Number of ground truth targets
    """
    if torch.is_tensor(ground_truth_mask):
        gt_mask = ground_truth_mask.cpu().numpy()
    else:
        gt_mask = ground_truth_mask
    
    # Binarize the mask
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(gt_binary)
    
    # Count components that meet minimum area requirement
    valid_targets = 0
    for label in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == label)
        if np.sum(component_mask) >= min_area:
            valid_targets += 1
    
    return valid_targets

def print_performance_report(results):
    """Print a comprehensive performance report"""
    print("\n" + "="*70)
    print("END-TO-END TARGET COUNT PERFORMANCE REPORT")
    print("="*70)
    
    print(f"Evaluation samples: {results['num_samples']}")
    print(f"Mean targets (predicted): {results['mean_predicted']:.2f} ± {results['std_predicted']:.2f}")
    print(f"Mean targets (ground truth): {results['mean_ground_truth']:.2f} ± {results['std_ground_truth']:.2f}")
    
    print(f"\nACCURACY METRICS:")
    print(f"  Perfect predictions (exact count): {results['perfect_accuracy']:.1%}")
    print(f"  Within ±1 target: {results['within_1_accuracy']:.1%}")
    print(f"  Within ±2 targets: {results['within_2_accuracy']:.1%}")
    
    print(f"\nERROR METRICS:")
    print(f"  Mean Absolute Error: {results['mean_absolute_error']:.2f} ± {results['std_absolute_error']:.2f}")
    print(f"  Root Mean Square Error: {results['rmse']:.2f}")
    print(f"  Mean Relative Error: {results['mean_relative_error']:.1%}")
    
    print(f"\nSTATISTICAL METRICS:")
    print(f"  Correlation coefficient: {results['correlation']:.3f}")
    print(f"  R² score: {results['r2_score']:.3f}")

def plot_performance_analysis(results, save_path=None):
    """Create comprehensive performance analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    predicted = results['predicted_counts']
    ground_truth = results['ground_truth_counts']
    abs_errors = results['absolute_errors']
    
    # 1. Scatter plot: Predicted vs Ground Truth
    axes[0].scatter(ground_truth, predicted, alpha=0.6, s=50)
    axes[0].plot([0, max(ground_truth.max(), predicted.max())], 
                 [0, max(ground_truth.max(), predicted.max())], 'r--', alpha=0.8, label='Perfect Prediction')
    axes[0].set_xlabel('Ground Truth Target Count')
    axes[0].set_ylabel('Predicted Target Count')
    axes[0].set_title(f'Predicted vs Ground Truth\n(r = {results["correlation"]:.3f}, R² = {results["r2_score"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    axes[1].hist(abs_errors, bins=range(int(abs_errors.max()) + 2), alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Absolute Error (|Predicted - Ground Truth|)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Error Distribution\n(MAE = {results["mean_absolute_error"]:.2f})')
    axes[1].set_ylim(0, 2950)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Target count distributions
    bins = range(max(ground_truth.max(), predicted.max()) + 2)
    axes[2].hist(ground_truth, bins=bins, alpha=0.7, label='Ground Truth', edgecolor='black')
    axes[2].hist(predicted, bins=bins, alpha=0.7, label='Predicted', edgecolor='black')
    axes[2].set_xlabel('Target Count')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Target Count Distributions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Error vs Ground Truth Count
    axes[3].scatter(ground_truth, abs_errors, alpha=0.6)
    axes[3].set_xlabel('Ground Truth Target Count')
    axes[3].set_ylabel('Absolute Error')
    axes[3].set_title('Error vs Ground Truth Count')
    axes[3].grid(True, alpha=0.3)
    
    # 5. Cumulative accuracy
    max_error = int(abs_errors.max())
    accuracies = []
    error_thresholds = range(max_error + 1)
    for threshold in error_thresholds:
        accuracy = np.mean(abs_errors <= threshold)
        accuracies.append(accuracy)
    
    axes[4].plot(error_thresholds, accuracies, 'bo-', linewidth=2, markersize=6)
    axes[4].set_xlabel('Error Threshold')
    axes[4].set_ylabel('Cumulative Accuracy')
    axes[4].set_title('Cumulative Accuracy vs Error Threshold')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim(0, 1)
    
    # 6. Performance summary text
    axes[5].axis('off')
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    Samples evaluated: {results['num_samples']}
    
    Accuracy:
    • Perfect predictions: {results['perfect_accuracy']:.1%}
    • Within ±1 target: {results['within_1_accuracy']:.1%}
    • Within ±2 targets: {results['within_2_accuracy']:.1%}
    
    Errors:
    • Mean Absolute Error: {results['mean_absolute_error']:.2f}
    • RMSE: {results['rmse']:.2f}
    • Mean Relative Error: {results['mean_relative_error']:.1%}
    
    Statistics:
    • Correlation: {results['correlation']:.3f}
    • R² Score: {results['r2_score']:.3f}
    """
    axes[5].text(0.05, 0.95, summary_text, transform=axes[5].transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance analysis saved to {save_path}")
    
    plt.show()
    
def evaluate_target_count_performance_from_loader(model, data_loader, dataset_name="test"):
    """
    Evaluate the end-to-end model performance using target count metrics from a DataLoader
    
    Args:
        model: EndToEndTargetDetector model
        data_loader: DataLoader with test/validation data
        dataset_name: Name of the dataset for logging
    
    Returns:
        dict: Performance metrics
    """
    predicted_counts = []
    ground_truth_counts = []
    absolute_errors = []
    relative_errors = []
    
    total_samples = len(data_loader.dataset)
    print(f"Evaluating target count performance on {total_samples} {dataset_name} samples...")
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(data_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx+1}/{len(data_loader)}...")
            
            # Process each sample in the batch
            for i in range(images.shape[0]):
                range_doppler_map = images[i]
                ground_truth_mask = masks[i]
                
                # Get predictions
                try:
                    centroids = model.predict_single(range_doppler_map)
                    predicted_count = len(centroids)
                except Exception as e:
                    print(f"Error processing sample {sample_count}: {e}")
                    sample_count += 1
                    continue
                
                # Count ground truth targets using connected components
                gt_count = count_ground_truth_targets(ground_truth_mask)
                
                # Calculate errors
                abs_error = abs(predicted_count - gt_count)
                rel_error = abs_error / max(gt_count, 1)  # Avoid division by zero
                
                predicted_counts.append(predicted_count)
                ground_truth_counts.append(gt_count)
                absolute_errors.append(abs_error)
                relative_errors.append(rel_error)
                
                sample_count += 1
    
    # Calculate metrics
    predicted_counts = np.array(predicted_counts)
    ground_truth_counts = np.array(ground_truth_counts)
    absolute_errors = np.array(absolute_errors)
    relative_errors = np.array(relative_errors)
    
    # Count-based metrics
    perfect_predictions = np.sum(absolute_errors == 0)
    within_1_target = np.sum(absolute_errors <= 1)
    within_2_targets = np.sum(absolute_errors <= 2)
    
    # Statistical metrics
    mean_abs_error = np.mean(absolute_errors)
    std_abs_error = np.std(absolute_errors)
    mean_rel_error = np.mean(relative_errors)
    correlation = np.corrcoef(predicted_counts, ground_truth_counts)[0, 1]
    
    # Regression metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(ground_truth_counts, predicted_counts)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth_counts, predicted_counts)
    
    results = {
        'dataset_name': dataset_name,
        'num_samples': len(predicted_counts),
        'predicted_counts': predicted_counts,
        'ground_truth_counts': ground_truth_counts,
        'absolute_errors': absolute_errors,
        'relative_errors': relative_errors,
        
        # Accuracy metrics
        'perfect_accuracy': perfect_predictions / len(predicted_counts),
        'within_1_accuracy': within_1_target / len(predicted_counts),
        'within_2_accuracy': within_2_targets / len(predicted_counts),
        
        # Error metrics
        'mean_absolute_error': mean_abs_error,
        'std_absolute_error': std_abs_error,
        'mean_relative_error': mean_rel_error,
        'rmse': rmse,
        
        # Statistical metrics
        'correlation': correlation,
        'r2_score': r2,
        
        # Count statistics
        'mean_predicted': np.mean(predicted_counts),
        'mean_ground_truth': np.mean(ground_truth_counts),
        'std_predicted': np.std(predicted_counts),
        'std_ground_truth': np.std(ground_truth_counts),
    }
    
    return results


if __name__ == "__main__":
    print("Running comprehensive end-to-end performance evaluation on 3-channel sequence data...")
    # Run comprehensive evaluation on the 3-frame U-Net with the specified dataset
    results = comprehensive_evaluation()
