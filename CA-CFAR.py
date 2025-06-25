import numpy as np
from sklearn.cluster import DBSCAN
import time

import matplotlib.pyplot as plt

# Import required functions
from sea_clutter.load_data import create_data_loaders
from src.end_to_end_helper import extract_ground_truth_centroids, spatial_target_matching

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_range_bins = 128
num_doppler_bins = 128

# CFAR parameters
pfa_values = [1e-4]  # Different false alarm probabilities
guard_cells = 2
training_cells = 10

# Dataset parameters
dataset_path = '/Users/pepijnlens/Documents/SeaClutterSuppression/local_data/10SNR_clutter.pt'
distance_threshold = 1.5  # Maximum distance for spatial target matching (pixels)

def ca_cfar_detector(rd_map_db, pfa, guard_cells=1, training_cells=10):
    """CA-CFAR detector implementation following MATLAB phased.CFARDetector"""

    # Convert from dB to linear magnitude
    rd_map_mag = 10 ** (rd_map_db / 20.0)
    
    # Convert to squared magnitude (square law detector output)
    rd_map_squared = rd_map_mag ** 2
    
    padding_size = training_cells + guard_cells
    rd_map_padded = np.pad(rd_map_squared, padding_size, mode='symmetric')
    
    # Calculate total number of training cells
    # Training cells are arranged in a rectangular band around the guard cells
    # Total training cells = (2*training_cells+1)^2 - (2*guard_cells+1)^2
    total_training_window = (2 * training_cells + 1) ** 2
    guard_window = (2 * guard_cells + 1) ** 2
    num_training_cells = total_training_window - guard_window
    
    # CA-CFAR threshold factor: α = N * (Pfa^(-1/N) - 1)
    # where N is the number of training cells
    alpha = num_training_cells * (pfa**(-1.0/num_training_cells) - 1.0)
    
    detections = np.zeros_like(rd_map_mag, dtype=bool)
    
    for i in range(num_range_bins):
        for j in range(num_doppler_bins):
            # Position in padded array
            pi, pj = i + padding_size, j + padding_size
            
            # Collect training cells (excluding guard cells and CUT)
            training_cells_values = []
            
            # Iterate over the training window
            for di in range(-training_cells, training_cells + 1):
                for dj in range(-training_cells, training_cells + 1):
                    # Skip guard cells and center cell (CUT)
                    if abs(di) > guard_cells or abs(dj) > guard_cells:
                        training_cells_values.append(rd_map_padded[pi + di, pj + dj])
            
            # Calculate noise power estimate (average of training cells)
            noise_power_estimate = np.mean(training_cells_values)
            
            # Calculate threshold
            threshold = alpha * noise_power_estimate
            
            # Detection test on squared magnitude
            if rd_map_squared[i, j] > threshold:
                detections[i, j] = True
    
    return detections.astype(int)


def cluster_detections(detections):
    """Cluster detection points using DBSCAN"""
    detection_points = np.array(np.where(detections == 1)).T
    if len(detection_points) == 0:
        return []
    
    clustering = DBSCAN(eps=1, min_samples=1).fit(detection_points)
    clusters = []
    for cluster_id in set(clustering.labels_):
        cluster_points = detection_points[clustering.labels_ == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        clusters.append((centroid[1], centroid[0]))  # Return as (x, y) coordinates
    
    return clusters

def visualize_sample(last_frame, last_mask, detections, detection_clusters, gt_centroids, sample_idx, pfa):
    """Visualize the ground truth mask and CFAR detections for a sample (overlapped)"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Show the ground truth mask as background
    ax.imshow(last_mask, cmap='gray', aspect='auto', alpha=0.5, label='GT Mask')

    # Overlay CFAR detections as a transparent mask
    ax.imshow(detections, cmap='Blues', aspect='auto', alpha=0.3, label='CFAR Detections')

    # Overlay ground truth centroids
    if gt_centroids:
        gt_x, gt_y = zip(*gt_centroids)
        ax.scatter(gt_x, gt_y, c='red', marker='x', s=100, linewidth=3, label='GT Centroids')

    # Overlay detection centroids
    if detection_clusters:
        det_x, det_y = zip(*detection_clusters)
        ax.scatter(det_x, det_y, c='blue', marker='o', s=100, alpha=0.7, label='CFAR Detections')

    ax.set_title(f'Sample {sample_idx}: GT Mask & CFAR Detections (PFA={pfa:.0e})')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Load dataset using the specified function
print("Loading dataset...")
_, _, test_loader = create_data_loaders(
    dataset_path=dataset_path,
    batch_size=1,  # Process one sample at a time
    num_workers=0,  # No multiprocessing for simpler debugging
    normalize=False,  # No normalization for raw data

)

print(f"Test dataset loaded: {len(test_loader)} samples")

# Run CFAR analysis on test set
print("Running CFAR analysis on test set...")
print(f"Testing with PFA values: {pfa_values}")
print("-" * 60)

results = {}
start_time = time.time()
inference_times = []  # Store individual inference times

for pfa_idx, pfa in enumerate(pfa_values):
    print(f"\nPFA: {pfa:.0e}")
    
    tp_total = 0
    fn_total = 0
    fp_total = 0
    total_targets = 0
    total_samples = 0
    
    results[pfa] = {'sample_results': [], 'tp': 0, 'fp': 0, 'fn': 0}
    
    # Process all test samples
    for batch_idx, (sequences, masks, labels) in enumerate(test_loader):
        # Extract last frame from sequence (shape: [1, frames, H, W])
        last_frame = sequences[0, -1].numpy()  # Last frame of the sequence
        last_mask = masks[0, -1].numpy()  # Last mask of the sequence
        
        # Extract ground truth centroids from mask
        gt_centroids = extract_ground_truth_centroids(last_mask)
        
        # Measure inference time for CA-CFAR detection
        inference_start = time.time()
        detections = ca_cfar_detector(last_frame, pfa, guard_cells, training_cells)
        inference_end = time.time()
        detection_clusters = cluster_detections(detections)
        
        # Store inference time
        sample_inference_time = inference_end - inference_start
        inference_times.append(sample_inference_time)
        
        # Perform spatial target matching
        matching_result = spatial_target_matching(
            detection_clusters, 
            gt_centroids, 
            distance_threshold
        )
        
        # Extract metrics
        tp = matching_result['true_positives']
        fp = matching_result['false_positives'] 
        fn = matching_result['false_negatives']
        
        # Accumulate totals
        tp_total += tp
        fp_total += fp
        fn_total += fn
        total_targets += len(gt_centroids)
        total_samples += 1
        
        # Store sample result
        results[pfa]['sample_results'].append({
            'sample_idx': batch_idx,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_targets': len(gt_centroids),
            'pred_targets': len(detection_clusters),
            'inference_time': sample_inference_time
        })
        
        # Visualize this sample
        visualize_sample(last_frame, last_mask, detections, detection_clusters, gt_centroids, batch_idx, pfa)
        
        # Progress update
        if (batch_idx + 1) % 50 == 0 or batch_idx == len(test_loader) - 1:
            progress_pct = 100 * (batch_idx + 1) / len(test_loader)
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} samples ({progress_pct:.1f}%)", end='\r')
    
    # Store final results for this PFA
    results[pfa]['tp'] = tp_total
    results[pfa]['fp'] = fp_total
    results[pfa]['fn'] = fn_total
    
    # Calculate metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Results for PFA {pfa:.0e}:")
    print(f"    True Positives: {tp_total}")
    print(f"    False Positives: {fp_total}")
    print(f"    False Negatives: {fn_total}")
    print(f"    Total Ground Truth Targets: {total_targets}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    F1-Score: {f1_score:.3f}")

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"CFAR analysis completed!")
print(f"Total time: {total_time:.1f} seconds")
print(f"Processed {total_samples} test samples")

# Calculate and print timing statistics
if inference_times:
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    
    print(f"\nInference Time Statistics:")
    print(f"Mean inference time per sample: {mean_inference_time*1000:.2f} ms")
    print(f"Std deviation: {std_inference_time*1000:.2f} ms")
    print(f"Min inference time: {min_inference_time*1000:.2f} ms")
    print(f"Max inference time: {max_inference_time*1000:.2f} ms")
    print(f"Total inference time: {sum(inference_times):.2f} seconds")

# Print final numerical results
print("\nFinal Results Summary:")
print("="*80)
print(f"{'PFA':<12} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'TP':<6} {'FP':<6} {'FN':<6}")
print("-" * 80)

for pfa in pfa_values:
    tp = results[pfa]['tp']
    fp = results[pfa]['fp']
    fn = results[pfa]['fn']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{pfa:<12.0e} {precision:<10.3f} {recall:<8.3f} {f1_score:<8.3f} {tp:<6d} {fp:<6d} {fn:<6d}")

print(f"\nDataset: {dataset_path}")
print(f"Distance threshold: {distance_threshold} pixels")
print(f"Guard cells: {guard_cells}")
print(f"Training cells: {training_cells}")
print(f"Analysis completed successfully!")
