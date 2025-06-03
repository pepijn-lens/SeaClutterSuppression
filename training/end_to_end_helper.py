import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.metrics import mean_squared_error, r2_score

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
    num_labels, labels = cv2.connectedComponents(gt_binary.squeeze(0))
    
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

def plot_performance_analysis(results, save_path='end_to_end_analysis/performance_analysis.png'):
    """Create comprehensive performance analysis plots"""
    
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    predicted = results['predicted_counts']
    ground_truth = results['ground_truth_counts']
    abs_errors = results['absolute_errors']

    len_predicted = len(predicted)
    
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
    axes[1].set_ylim(0, 3000)
    axes[1].set_xlim(0, ground_truth.max() - ground_truth.max()//2)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Target count distributions
    bins = range(max(ground_truth.max(), predicted.max()) + 2)
    axes[2].hist(ground_truth, bins=bins, alpha=0.7, label='Ground Truth', edgecolor='black')
    axes[2].hist(predicted, bins=bins, alpha=0.7, label='Predicted', edgecolor='black')
    axes[2].set_xlabel('Target Count')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Target Count Distributions')
    axes[2].set_xlim(0, ground_truth.max() + 1)
    axes[2].set_ylim(0, len_predicted // 10 + np.ceil(len_predicted * 0.04))
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
        plt.savefig(f'{save_path}/target_count.png', dpi=300, bbox_inches='tight')
        print(f"Performance analysis saved to {save_path}")
    else:
        plt.show()
    
def evaluate_target_count_performance(model, data_loader, dataset_name="test"):
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
                range_doppler_map = images[i].to('mps')
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

def analyze_single_sample(model, dataset, sample_idx):
    """
    Analyze a single sample with the end-to-end model and display results
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING SAMPLE {sample_idx}")
    print(f"{'='*50}")
    
    # Get sample data
    image, mask = dataset[sample_idx]
    
    print(f"Input shape: {image.shape}")
    print(f"Ground truth mask shape: {mask.shape}")
    
    # Get ground truth target count
    gt_count = count_ground_truth_targets(mask.squeeze(0))
    print(f"Ground truth targets: {gt_count}")
    
    # Get model predictions
    try:
        centroids = model.predict_single(image)
        predicted_count = len(centroids)
        
        print(f"Predicted targets: {predicted_count}")
        print(f"Absolute error: {abs(predicted_count - gt_count)}")
        
        if centroids:
            print(f"Predicted centroids:")
            for i, (x, y) in enumerate(centroids):
                print(f"  Target {i+1}: ({x:.1f}, {y:.1f})")
        else:
            print("No targets detected")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Create visualization
    visualize_sample_results(image, mask, centroids, sample_idx, gt_count, predicted_count)

def visualize_sample_results(image, mask, centroids, sample_idx, gt_count, predicted_count):
    """
    Create a visualization of the sample results
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image (show last channel if multi-channel)
    if len(image.shape) > 1:
        input_display = image[-1].cpu().numpy() if torch.is_tensor(image) else image[-1]
    else:
        input_display = image.cpu().numpy() if torch.is_tensor(image) else image
    
    axes[0].imshow(input_display, cmap='viridis')
    axes[0].set_title(f'Input Range-Doppler Map\nSample {sample_idx}')
    axes[0].axis('off')
    
    # Ground truth mask
    gt_display = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    if len(gt_display.shape) > 1:
        gt_display = gt_display[-1]  # Take last channel if multi-channel
    
    axes[1].imshow(gt_display, cmap='hot', alpha=0.8)
    axes[1].imshow(input_display, cmap='viridis', alpha=0.3)
    axes[1].set_title(f'Ground Truth Overlay\n{gt_count} targets')
    axes[1].axis('off')
    
    # Predictions overlay
    axes[2].imshow(input_display, cmap='viridis', alpha=0.5)
    
    # Plot predicted centroids with labels
    if centroids:
        x_coords, y_coords = zip(*centroids)
        axes[2].scatter(x_coords, y_coords, c='red', s=100, marker='x', linewidths=3, label='Predicted')
        
        # Add labels for each target
        for i, (x, y) in enumerate(centroids):
            # Add text label with a slight offset to avoid overlapping with the marker
            axes[2].annotate(i+1, 
                           xy=(x, y), 
                           xytext=(x+2, y-2),  # Offset the text slightly
                           fontsize=10, 
                           fontweight='bold',
                           color='white',
                           ha='center', va='center')
    
    axes[2].set_title(f'Predictions Overlay\n{predicted_count} targets (Error: {abs(predicted_count - gt_count)})')
    axes[2].legend()
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_dataset_stats(dataset):
    """
    Show basic statistics about the dataset
    """
    print(f"\n{'='*40}")
    print("DATASET STATISTICS")
    print(f"{'='*40}")
    print(f"Total samples: {len(dataset)}")
    
    # Sample a few examples to get statistics
    sample_size = min(100, len(dataset))
    target_counts = []
    
    print(f"Analyzing {sample_size} samples for statistics...")
    
    for i in range(0, len(dataset), max(1, len(dataset) // sample_size)):
        if len(target_counts) >= sample_size:
            break
        _, mask = dataset[i]
        count = count_ground_truth_targets(mask)
        target_counts.append(count)
    
    target_counts = np.array(target_counts)
    
    print(f"Target count statistics (from {len(target_counts)} samples):")
    print(f"  Mean: {np.mean(target_counts):.2f}")
    print(f"  Std: {np.std(target_counts):.2f}")
    print(f"  Min: {np.min(target_counts)}")
    print(f"  Max: {np.max(target_counts)}")
    print(f"  Median: {np.median(target_counts):.1f}")
    
    # Show distribution
    unique, counts = np.unique(target_counts, return_counts=True)
    print(f"\nTarget count distribution:")
    for target_count, frequency in zip(unique, counts):
        percentage = (frequency / len(target_counts)) * 100
        print(f"  {target_count} targets: {frequency} samples ({percentage:.1f}%)")

