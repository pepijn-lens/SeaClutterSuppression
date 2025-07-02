import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.spatial.distance import cdist
import marimo as mo


def analyze_single_sample(model, dataset, sample_idx, distance_threshold=5.0, marimo_var=False):
    """
    Analyze a single sample with the end-to-end model and display results including spatial evaluation
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
    
    # Extract ground truth centroids from mask
    if mask is not None:
        gt_centroids = extract_ground_truth_centroids(mask)
        print(f"Ground truth centroids: {len(gt_centroids)}")
        for i, (x, y) in enumerate(gt_centroids):
            print(f"  GT Target {i+1}: ({x:.1f}, {y:.1f})")
    else:
        gt_centroids = []
        print("No mask available for spatial evaluation")
    
    # Get model predictions
    try:
        pred_centroids = model.predict_single(image)
        predicted_count = len(pred_centroids)
        
        print(f"\nPredicted targets: {predicted_count}")
        print(f"Count-based absolute error: {abs(predicted_count - gt_count)}")
        
        if pred_centroids:
            print(f"Predicted centroids:")
            for i, (x, y) in enumerate(pred_centroids):
                print(f"  Pred Target {i+1}: ({x:.1f}, {y:.1f})")
        else:
            print("No targets detected")
        
        # Perform spatial matching if we have ground truth centroids
        if gt_centroids:
            matching_result = spatial_target_matching(pred_centroids, gt_centroids, distance_threshold)
            
            print(f"\nSPATIAL EVALUATION (threshold: {distance_threshold} pixels):")
            print(f"  True Positives:  {matching_result['true_positives']}")
            print(f"  False Positives: {matching_result['false_positives']}")
            print(f"  False Negatives: {matching_result['false_negatives']}")
            
            if matching_result['true_positives'] > 0:
                precision = matching_result['true_positives'] / (matching_result['true_positives'] + matching_result['false_positives'])
                recall = matching_result['true_positives'] / (matching_result['true_positives'] + matching_result['false_negatives'])
                f1 = 2 * (precision * recall) / (precision + recall)
                
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                
                if matching_result['matched_distances']:
                    avg_distance = np.mean(matching_result['matched_distances'])
                    print(f"  Average match distance: {avg_distance:.2f} pixels")
            
            # Show matches
            if matching_result['matches']:
                print(f"\nMatched pairs:")
                for pred_idx, gt_idx in matching_result['matches']:
                    pred_x, pred_y = pred_centroids[pred_idx]
                    gt_x, gt_y = gt_centroids[gt_idx]
                    distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                    print(f"  Pred({pred_x:.1f}, {pred_y:.1f}) <-> GT({gt_x:.1f}, {gt_y:.1f}) [dist: {distance:.2f}]")
        else:
            matching_result = None
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Create visualization
    result_fig = visualize_sample_results(image, mask, pred_centroids, sample_idx, gt_count, predicted_count, 
                           gt_centroids, matching_result, marimo_var=marimo_var)
    
    if marimo_var:
        return result_fig

def visualize_sample_results(image, mask, pred_centroids, sample_idx, gt_count, predicted_count, 
                           gt_centroids=None, matching_result=None, marimo_var=False):
    """
    Create a visualization of the sample results including spatial matching
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
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
    
    # Plot ground truth centroids if available
    legend_handles = []
    if gt_centroids:
        gt_x, gt_y = zip(*gt_centroids) if gt_centroids else ([], [])
        scatter_gt = axes[1].scatter(gt_x, gt_y, c='yellow', s=150, marker='o', 
                       linewidths=2, edgecolors='black', label='GT Centroids')
        legend_handles.append(scatter_gt)
        
        # Add labels for GT targets
        for i, (x, y) in enumerate(gt_centroids):
            axes[1].annotate(f'GT{i+1}', xy=(x, y), xytext=(x+3, y-3),
                           fontsize=8, fontweight='bold', color='yellow',
                           ha='center', va='center')
    
    axes[1].set_title(f'Ground Truth Overlay\n{gt_count} targets')
    if legend_handles:
        axes[1].legend()
    axes[1].axis('off')
    
    # Predictions overlay
    axes[2].imshow(input_display, cmap='viridis', alpha=0.5)
    
    # Plot predicted centroids
    legend_handles = []
    if pred_centroids:
        pred_x, pred_y = zip(*pred_centroids)
        scatter_pred = axes[2].scatter(pred_x, pred_y, c='red', s=100, marker='x', 
                       linewidths=3, label='Predicted')
        legend_handles.append(scatter_pred)
        
        # Add labels for each predicted target
        for i, (x, y) in enumerate(pred_centroids):
            axes[2].annotate(f'P{i+1}', xy=(x, y), xytext=(x+2, y-2),
                           fontsize=8, fontweight='bold', color='white',
                           ha='center', va='center')
    
    axes[2].set_title(f'Predictions Overlay\n{predicted_count} targets (Count Error: {abs(predicted_count - gt_count)})')
    if legend_handles:
        axes[2].legend()
    axes[2].axis('off')
    
    # Spatial matching visualization
    axes[3].imshow(input_display, cmap='viridis', alpha=0.4)
    
    legend_handles = []
    if gt_centroids and pred_centroids and matching_result:
        # Plot all ground truth (yellow circles)
        gt_x, gt_y = zip(*gt_centroids)
        scatter_gt = axes[3].scatter(gt_x, gt_y, c='yellow', s=150, marker='o', 
                       linewidths=2, label='Ground Truth', alpha=0.7)
        legend_handles.append(scatter_gt)
        
        # Plot all predictions with different colors based on matching
        matched_pred_indices = set([match[0] for match in matching_result['matches']])
        
        # True positives (green)
        tp_coords = [pred_centroids[i] for i in matched_pred_indices]
        if tp_coords:
            tp_x, tp_y = zip(*tp_coords)
            scatter_tp = axes[3].scatter(tp_x, tp_y, c='green', s=100, marker='x', 
                           linewidths=3, label='True Positives')
            legend_handles.append(scatter_tp)
        
        # False positives (red)
        fp_coords = [pred_centroids[i] for i in range(len(pred_centroids)) if i not in matched_pred_indices]
        if fp_coords:
            fp_x, fp_y = zip(*fp_coords)
            scatter_fp = axes[3].scatter(fp_x, fp_y, c='red', s=100, marker='x', 
                           linewidths=3, label='False Positives')
            legend_handles.append(scatter_fp)
        
        # Draw lines connecting matches
        for pred_idx, gt_idx in matching_result['matches']:
            pred_x, pred_y = pred_centroids[pred_idx]
            gt_x, gt_y = gt_centroids[gt_idx]
            axes[3].plot([pred_x, gt_x], [pred_y, gt_y], 'g--', alpha=0.7, linewidth=2)
        
        # Calculate metrics for title
        tp = matching_result['true_positives']
        fp = matching_result['false_positives']
        fn = matching_result['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        title = f'Spatial Matching\nTP:{tp} FP:{fp} FN:{fn}\nF1: {f1:.3f}'
    else:
        # Fallback: just show predictions
        if pred_centroids:
            pred_x, pred_y = zip(*pred_centroids)
            scatter_pred = axes[3].scatter(pred_x, pred_y, c='red', s=100, marker='x', 
                           linewidths=3, label='Predicted')
            legend_handles.append(scatter_pred)
        title = 'Predictions\n(No spatial matching available)'
    
    axes[3].set_title(title)
    if legend_handles:
        axes[3].legend()
    axes[3].axis('off')
    
    plt.tight_layout()
    if marimo_var:
        return fig
    else:
        plt.show()


def spatial_target_matching(predicted_centroids, ground_truth_centroids, distance_threshold=5.0):
    """
    Match predicted targets to ground truth targets based on spatial distance.
    
    Args:
        predicted_centroids: List of (x, y) coordinates for predicted targets
        ground_truth_centroids: List of (x, y) coordinates for ground truth targets
        distance_threshold: Maximum distance for a match to be considered valid
    
    Returns:
        dict: Contains matched pairs, true positives, false positives, false negatives
    """
    if not predicted_centroids:
        predicted_centroids = []
    if not ground_truth_centroids:
        ground_truth_centroids = []
    
    # Convert to numpy arrays for easier computation
    if len(predicted_centroids) > 0:
        pred_array = np.array(predicted_centroids)
    else:
        pred_array = np.empty((0, 2))
    
    if len(ground_truth_centroids) > 0:
        gt_array = np.array(ground_truth_centroids)
    else:
        gt_array = np.empty((0, 2))
    
    # Handle empty cases
    if len(pred_array) == 0 and len(gt_array) == 0:
        return {
            'matches': [],
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'matched_distances': []
        }
    elif len(pred_array) == 0:
        return {
            'matches': [],
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(gt_array),
            'matched_distances': []
        }
    elif len(gt_array) == 0:
        return {
            'matches': [],
            'true_positives': 0,
            'false_positives': len(pred_array),
            'false_negatives': 0,
            'matched_distances': []
        }
    
    # Calculate distance matrix between all predicted and ground truth points
    distance_matrix = cdist(pred_array, gt_array, metric='euclidean')
    
    # Find optimal matching using a greedy approach, ensuring one-to-one matching
    matches = []
    matched_pred_indices = set()
    matched_gt_indices = set()
    matched_distances = []
    
    # Create list of all potential matches within threshold
    potential_matches = []
    for i in range(len(pred_array)):
        for j in range(len(gt_array)):
            if distance_matrix[i, j] <= distance_threshold:
                potential_matches.append((distance_matrix[i, j], i, j))
    
    # Sort by distance and greedily assign matches, removing matched GTs
    potential_matches.sort(key=lambda x: x[0])
    
    for distance, pred_idx, gt_idx in potential_matches:
        if pred_idx not in matched_pred_indices and gt_idx not in matched_gt_indices:
            matches.append((pred_idx, gt_idx))
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)
            matched_distances.append(distance)
            # Once a GT is matched, it cannot be matched again (enforced by matched_gt_indices)
    
    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(pred_array) - true_positives
    false_negatives = len(gt_array) - true_positives
    
    return {
        'matches': matches,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'matched_distances': matched_distances
    }


def extract_ground_truth_centroids(mask, threshold=0.5):
    """
    Extract ground truth target centroids from a segmentation mask.
    
    Args:
        mask: Ground truth mask (H x W) or (C x H x W)
        threshold: Threshold for binarizing the mask
    
    Returns:
        List of (x, y) coordinates for ground truth targets
    """
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Handle multi-channel masks
    if len(mask_np.shape) > 2:
        mask_np = mask_np[-1]  # Take last channel
    
    # Binarize mask
    binary_mask = mask_np > threshold
    
    # Find connected components (simple approach using basic clustering)
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(binary_mask)
    
    centroids = []
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        y_coords, x_coords = np.where(component_mask)
        if len(y_coords) > 0:
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            centroids.append((centroid_x, centroid_y))
    
    return centroids


def evaluate_spatial_performance(model, data_loader, distance_threshold=5.0, dataset_name="test"):
    """
    Evaluate model performance using spatial matching between predicted and ground truth targets.
    
    Args:
        model: EndToEndTargetDetector model
        data_loader: DataLoader with test/validation data
        distance_threshold: Maximum distance for considering a prediction as correct
        dataset_name: Name of the dataset for logging
    
    Returns:
        dict: Comprehensive performance metrics including precision, recall, F1-score
    """
    all_matches = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_distances = []
    
    sample_results = []  # Store per-sample results for analysis
    
    total_samples = len(data_loader.dataset)
    print(f"Evaluating spatial performance on {total_samples} {dataset_name} samples...")
    print(f"Distance threshold: {distance_threshold} pixels")
    
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
                mask = masks[i] if masks is not None else None
                
                # Extract ground truth centroids from mask
                if mask is not None:
                    gt_centroids = extract_ground_truth_centroids(mask)
                else:
                    # Fallback: use label count to create dummy centroids (not ideal)
                    gt_count = int(labels[i].item()) if torch.is_tensor(labels) else int(labels[i])
                    print(f"Warning: No mask available for sample {sample_count}, using count-based evaluation")
                    # This is a fallback - in practice, you need masks for spatial evaluation
                    gt_centroids = [(0, 0)] * gt_count  # Dummy centroids
                
                # Get model predictions
                try:
                    pred_centroids = model.predict_single(range_doppler_map)
                except Exception as e:
                    print(f"Error processing sample {sample_count}: {e}")
                    sample_count += 1
                    continue
                
                # Perform spatial matching
                matching_result = spatial_target_matching(
                    pred_centroids, gt_centroids, distance_threshold
                )
                
                # Accumulate statistics
                total_tp += matching_result['true_positives']
                total_fp += matching_result['false_positives']
                total_fn += matching_result['false_negatives']
                all_matched_distances.extend(matching_result['matched_distances'])
                
                # Store sample result for detailed analysis
                sample_results.append({
                    'sample_idx': sample_count,
                    'predicted_count': len(pred_centroids),
                    'ground_truth_count': len(gt_centroids),
                    'true_positives': matching_result['true_positives'],
                    'false_positives': matching_result['false_positives'],
                    'false_negatives': matching_result['false_negatives'],
                    'predicted_centroids': pred_centroids,
                    'ground_truth_centroids': gt_centroids,
                    'matches': matching_result['matches']
                })
                
                sample_count += 1
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'dataset_name': dataset_name,
        'distance_threshold': distance_threshold,
        'num_samples': sample_count,
        
        # Spatial metrics
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        
        # Distance statistics
        'mean_match_distance': np.mean(all_matched_distances) if all_matched_distances else 0.0,
        'std_match_distance': np.std(all_matched_distances) if all_matched_distances else 0.0,
        'max_match_distance': np.max(all_matched_distances) if all_matched_distances else 0.0,
        
        # Per-sample results for detailed analysis
        'sample_results': sample_results,
        
        # Count-based metrics for comparison
        'total_predicted': total_tp + total_fp,
        'total_ground_truth': total_tp + total_fn,
    }
    
    return results


def print_spatial_performance_report(results):
    """Print a comprehensive spatial performance report"""
    print("\n" + "="*70)
    print("SPATIAL TARGET DETECTION PERFORMANCE REPORT")
    print("="*70)
    
    print(f"Dataset: {results['dataset_name']}")
    print(f"Evaluation samples: {results['num_samples']}")
    print(f"Distance threshold: {results['distance_threshold']} pixels")
    
    print(f"\nSPATIAL MATCHING RESULTS:")
    print(f"  True Positives:  {results['total_true_positives']}")
    print(f"  False Positives: {results['total_false_positives']}")
    print(f"  False Negatives: {results['total_false_negatives']}")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall:    {results['recall']:.3f}")
    print(f"  F1-Score:  {results['f1_score']:.3f}")
    
    print(f"\nCOUNT COMPARISON:")
    print(f"  Total predicted targets: {results['total_predicted']}")
    print(f"  Total ground truth targets: {results['total_ground_truth']}")
    
    if results['mean_match_distance'] > 0:
        print(f"\nMATCH DISTANCE STATISTICS:")
        print(f"  Mean distance: {results['mean_match_distance']:.2f} ± {results['std_match_distance']:.2f} pixels")
        print(f"  Max distance:  {results['max_match_distance']:.2f} pixels")


def plot_spatial_performance_analysis(results, save_path=None, marimo=False):
    """Create comprehensive spatial performance analysis plots"""
    if save_path is not None and not marimo:
        os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Precision, Recall, F1-Score bar chart
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [results['precision'], results['recall'], results['f1_score']]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = axes[0, 0].bar(metrics, values, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confusion matrix style visualization
    confusion_data = np.array([[results['total_true_positives'], results['total_false_negatives']],
                              [results['total_false_positives'], 0]])  # Bottom right is not applicable
    
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Predicted Present', 'Predicted Absent'],
                yticklabels=['Actually Present', 'Actually Absent'], 
                cbar=True)
    axes[0, 1].set_title('Detection Confusion Matrix')
    
    # 3. Per-sample performance distribution
    sample_f1_scores = []
    for sample in results['sample_results']:
        tp = sample['true_positives']
        fp = sample['false_positives']
        fn = sample['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        sample_f1_scores.append(f1)
    
    axes[1, 0].hist(sample_f1_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('F1-Score')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Distribution of Per-Sample F1-Scores')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    SPATIAL DETECTION SUMMARY
    
    Samples: {results['num_samples']}
    Distance threshold: {results['distance_threshold']} px
    
    Detection Results:
    • True Positives: {results['total_true_positives']}
    • False Positives: {results['total_false_positives']}
    • False Negatives: {results['total_false_negatives']}
    
    Performance:
    • Precision: {results['precision']:.3f}
    • Recall: {results['recall']:.3f}
    • F1-Score: {results['f1_score']:.3f}
    
    Average match distance: {results['mean_match_distance']:.2f} px
    """
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/spatial_performance.png', dpi=300, bbox_inches='tight')
        print(f"Spatial performance analysis saved to {save_path}")
    elif marimo:
        return fig
    else:
        plt.show()

