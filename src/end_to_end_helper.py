import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns



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
    print(f"  False Negative Rate: {results['false_negative_rate']:.1%}")
    print(f"  False Positive Rate: {results['false_positive_rate']:.1%}")

def plot_performance_analysis(results, save_path=None, marimo=False):
    """Create comprehensive performance analysis plots"""
    if save_path is not None and not marimo:
        os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    predicted = results['predicted_counts']
    ground_truth = results['ground_truth_counts']
    abs_errors = results['absolute_errors']
    
    # 1. Confusion matrix: Predicted vs Ground Truth
    # Compute confusion matrix
    max_count = max(ground_truth.max(), predicted.max())
    labels = np.arange(0, max_count + 1)
    cm = confusion_matrix(ground_truth, predicted, labels=labels)

    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=True)
    axes[0].set_xlabel('Predicted Target Count')
    axes[0].set_ylabel('Ground Truth Target Count')
    axes[0].set_title('Confusion Matrix\n(Predicted vs Ground Truth)')
    
    # 2. Cumulative accuracy
    max_error = int(abs_errors.max())
    accuracies = []
    error_thresholds = range(max_error + 1)
    for threshold in error_thresholds:
        accuracy = np.mean(abs_errors <= threshold)
        accuracies.append(accuracy)
    
    axes[1].plot(error_thresholds, accuracies, 'bo-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Error Threshold')
    axes[1].set_ylabel('Cumulative Accuracy')
    axes[1].set_title('Cumulative Accuracy vs Error Threshold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # 3. Performance summary text
    axes[2].axis('off')
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    Samples evaluated: {results['num_samples']}
    
    Accuracy:
    • Perfect predictions: {results['perfect_accuracy']:.1%}
    • False Negative Rate: {results['false_negative_rate']:.1%}
    • False Positive Rate: {results['false_positive_rate']:.1%}
    """
    axes[2].text(0.05, 0.95, summary_text, transform=axes[2].transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/target_count.png', dpi=300, bbox_inches='tight')
        print(f"Performance analysis saved to {save_path}")
    elif marimo:
        return fig
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
    
    total_samples = len(data_loader.dataset)
    print(f"Evaluating target count performance on {total_samples} {dataset_name} samples...")
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx+1}/{len(data_loader)}...")
            
            # Handle different batch formats
            if len(batch_data) == 3:  # (images, masks, labels)
                images, masks, labels = batch_data
            else:
                print(f"Unexpected batch format with {len(batch_data)} elements. Expected 3 elements: (images, masks, labels)")
                continue
            
            # Process each sample in the batch
            for i in range(images.shape[0]):
                range_doppler_map = images[i].to('mps' if torch.backends.mps.is_available() else 'cpu')
                
                # Get ground truth count from labels
                if torch.is_tensor(labels):
                    gt_count = int(labels[i].item())
                else:
                    gt_count = int(labels[i])
                
                # Get predictions
                try:
                    centroids = model.predict_single(range_doppler_map)
                    predicted_count = len(centroids)
                except Exception as e:
                    print(f"Error processing sample {sample_count}: {e}")
                    sample_count += 1
                    continue
                
                # Calculate errors
                abs_error = abs(predicted_count - gt_count)
                
                predicted_counts.append(predicted_count)
                ground_truth_counts.append(gt_count)
                absolute_errors.append(abs_error)
                
                sample_count += 1
    
    # Calculate metrics
    predicted_counts = np.array(predicted_counts)
    ground_truth_counts = np.array(ground_truth_counts)
    absolute_errors = np.array(absolute_errors)
    
    # Count-based metrics
    perfect_predictions = np.sum(absolute_errors == 0)
    
    # Calculate False Negative Rate and False Positive Rate
    # False Negative Rate: missed targets when there are ground truth targets
    target_samples = np.sum(ground_truth_counts > 0)
    false_negatives = np.sum((ground_truth_counts > 0) & (predicted_counts < ground_truth_counts))
    false_negative_rate = false_negatives / max(target_samples, 1) if target_samples > 0 else 0.0
    
    # False Positive Rate: over-predictions (predicted > ground truth)
    over_predictions = np.sum(predicted_counts > ground_truth_counts)
    false_positive_rate = over_predictions / len(predicted_counts)
    
    results = {
        'dataset_name': dataset_name,
        'num_samples': len(predicted_counts),
        'predicted_counts': predicted_counts,
        'ground_truth_counts': ground_truth_counts,
        'absolute_errors': absolute_errors,
        
        # Accuracy metrics
        'perfect_accuracy': perfect_predictions / len(predicted_counts),
        'false_negative_rate': false_negative_rate,
        'false_positive_rate': false_positive_rate,
        
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
    sample_data = dataset[sample_idx]
    
    if len(sample_data) == 3:  # (image, mask, label)
        image, mask, label = sample_data
        gt_count = int(label.item()) if torch.is_tensor(label) else int(label)
    else:
        print(f"Unexpected sample format with {len(sample_data)} elements. Expected 3 elements: (image, mask, label)")
        return
    
    print(f"Input shape: {image.shape}")
    if mask is not None:
        print(f"Ground truth mask shape: {mask.shape}")
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
        
        sample_data = dataset[i]
        
        # Extract target count from labels
        if len(sample_data) == 3:  # (image, mask, label)
            _, _, label = sample_data
            count = int(label.item()) if torch.is_tensor(label) else int(label)
        else:
            print(f"Warning: Unexpected sample format for sample {i}, skipping...")
            continue
            
        target_counts.append(count)
    
    if not target_counts:
        print("Warning: No valid target counts found in dataset")
        return
    
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

