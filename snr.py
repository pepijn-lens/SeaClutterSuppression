import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import time
import torch

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_range_bins = 128
num_doppler_bins = 128
max_distance = 3  # Maximum pixel distance for target matching

# CFAR parameters
pfa_values = [1e-4, 1e-3]  # Different false alarm probabilities
guard_cells = 3
training_cells = 10

# Dataset parameters
dataset_path = '/Users/pepijnlens/Documents/SeaClutterSuppression/data/my_sea_clutter_dataset.pt'
num_samples_to_test = 100  # Number of samples from dataset to test

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
    
    clustering = DBSCAN(eps=1.5, min_samples=1).fit(detection_points)
    clusters = []
    for cluster_id in set(clustering.labels_):
        cluster_points = detection_points[clustering.labels_ == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        clusters.append(centroid)
    
    return clusters

def match_targets_to_detections(true_targets, detection_clusters, max_dist=0):
    """Match true targets to detection clusters based on distance"""
    if len(detection_clusters) == 0:
        return 0, len(true_targets), 0
    
    if len(true_targets) == 0:
        return 0, 0, len(detection_clusters)
    
    true_targets = np.array(true_targets)
    detection_clusters = np.array(detection_clusters)
    
    # Calculate distances between all target-detection pairs
    distances = cdist(true_targets, detection_clusters)
    
    # Find matches within max_distance
    matched_targets = set()
    matched_detections = set()
    
    for i in range(len(true_targets)):
        for j in range(len(detection_clusters)):
            if distances[i, j] <= max_dist and i not in matched_targets and j not in matched_detections:
                matched_targets.add(i)
                matched_detections.add(j)
    
    true_positives = len(matched_targets)
    false_negatives = len(true_targets) - true_positives
    false_positives = len(detection_clusters) - true_positives
    
    return true_positives, false_negatives, false_positives

def extract_rd_map_and_targets(data, sample_idx):
    """Extract range-doppler map and target locations from dataset"""
    # Use the first frame (index 0) from the sequence
    rd_map = data['sequences'][sample_idx, 0].numpy()  # Shape: (128, 128)
    num_targets = data['labels'][sample_idx].item()
    
    target_locations = []
    if num_targets > 0:
        # Get target mask for the first frame
        target_mask = data['masks'][sample_idx, 0].numpy()  # Shape: (128, 128)
        
        # Find target positions from the mask
        target_positions = np.where(target_mask > 0)
        
        # Group nearby pixels into clusters to get target centroids
        if len(target_positions[0]) > 0:
            target_points = np.column_stack(target_positions)
            
            # Use DBSCAN to cluster target pixels
            if len(target_points) > 0:
                clustering = DBSCAN(eps=2.0, min_samples=1).fit(target_points)
                
                # Calculate centroid for each cluster
                for cluster_id in set(clustering.labels_):
                    if cluster_id != -1:  # Ignore noise points
                        cluster_points = target_points[clustering.labels_ == cluster_id]
                        centroid = np.mean(cluster_points, axis=0)
                        r_idx, d_idx = int(centroid[0]), int(centroid[1])
                        target_locations.append((r_idx, d_idx))
    
    # Return the dB data directly since it's already in dB format
    return rd_map, target_locations, num_targets

# Load dataset
print("Loading dataset...")
data = torch.load(dataset_path)
print(f"Dataset loaded: {len(data['sequences'])} samples")
print(f"Sample shape: {data['sequences'][0].shape}")
print(f"Number of frames per sample: {data['sequences'].shape[1]}")
print(f"Max targets in dataset: {data['metadata']['max_targets']}")
print(f"Target type: {data['metadata']['target_type']}")
print(f"SNR parameters: {data['metadata']['snr_params']}")

# Run CFAR analysis on dataset
print("Running CFAR analysis on dataset...")
print(f"Testing {num_samples_to_test} random samples with {len(pfa_values)} PFA values")
print(f"PFA values: {pfa_values}")
print("-" * 60)

# Generate random sample indices for consistent sampling across all PFA values
total_samples = len(data['sequences'])
random_sample_indices = np.random.choice(total_samples, size=min(num_samples_to_test, total_samples), replace=False)
print(f"Selected {len(random_sample_indices)} random samples from {total_samples} total samples")

results = {}
start_time = time.time()
total_configs = len(pfa_values)
config_count = 0

for pfa_idx, pfa in enumerate(pfa_values):
    config_count += 1
    config_start_time = time.time()
    
    print(f"\nConfiguration {config_count}/{total_configs} ({100*config_count/total_configs:.1f}%)")
    print(f"PFA: {pfa:.0e}")
    
    tp_total = 0
    fn_total = 0
    fp_total = 0
    total_targets = 0
    total_cells = 0
    
    results[pfa] = {'sample_idx': [], 'pd': [], 'pfa_empirical': [], 'tp': [], 'fp': [], 'fn': []}
    
    # Progress bar setup for samples
    sample_progress_interval = max(1, len(random_sample_indices) // 10)  # Update every 10%
    
    for i, sample_idx in enumerate(random_sample_indices):
        if i % sample_progress_interval == 0 or i == len(random_sample_indices) - 1:
            progress_pct = 100 * (i + 1) / len(random_sample_indices)
            print(f"  Sample {i + 1}/{len(random_sample_indices)} (idx {sample_idx}, {progress_pct:.0f}%)", end='\r')
        
        rd_map_db, true_targets, num_targets_sample = extract_rd_map_and_targets(data, sample_idx)
        detections = ca_cfar_detector(rd_map_db, pfa, guard_cells, training_cells)
        detection_clusters = cluster_detections(detections)
        
        tp, fn, fp = match_targets_to_detections(true_targets, detection_clusters, max_distance)
        
        tp_total += tp
        fn_total += fn
        fp_total += fp
        total_targets += len(true_targets)
        total_cells += num_range_bins * num_doppler_bins - len(true_targets)
        
        # Store per-sample results
        results[pfa]['sample_idx'].append(sample_idx)
        results[pfa]['tp'].append(tp)
        results[pfa]['fp'].append(fp)
        results[pfa]['fn'].append(fn)
    
    # Calculate overall metrics for this PFA
    detection_probability = tp_total / total_targets if total_targets > 0 else 0
    empirical_pfa = fp_total / total_cells if total_cells > 0 else 0
    
    results[pfa]['pd'] = detection_probability
    results[pfa]['pfa_empirical'] = empirical_pfa
    
    # Time estimation
    config_time = time.time() - config_start_time
    elapsed_time = time.time() - start_time
    remaining_configs = total_configs - config_count
    
    print(f"\n  Results: Pd = {detection_probability:.3f}, Empirical PFA = {empirical_pfa:.2e}")
    print(f"  True Positives: {tp_total}, False Positives: {fp_total}, False Negatives: {fn_total}")
    print(f"  Total Targets: {total_targets}, Total Samples: {len(random_sample_indices)}")
    
    if remaining_configs > 0:
        avg_time_per_config = elapsed_time / config_count
        estimated_remaining_time = remaining_configs * avg_time_per_config
        print(f"  Config time: {config_time:.1f}s, Est. remaining: {estimated_remaining_time/60:.1f} min")
    else:
        print(f"  Config time: {config_time:.1f}s")

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"CFAR analysis completed!")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Average time per configuration: {total_time/total_configs:.1f} seconds")

# Generate and save sample range-doppler maps with detections for visualization
print("\nGenerating sample range-doppler visualization maps...")
# Use first few random samples for visualization
visualization_indices = random_sample_indices[:3]  # Use first 3 random samples

for sample_idx in visualization_indices:
    rd_map_db, true_targets, num_targets_sample = extract_rd_map_and_targets(data, sample_idx)
    
    # Create visualization for different PFA values
    fig, axes = plt.subplots(2, len(pfa_values), figsize=(4*len(pfa_values), 8))
    if len(pfa_values) == 1:
        axes = axes.reshape(-1, 1)
    
    # Data is already in dB format
    
    for pfa_idx, pfa in enumerate(pfa_values):
        detections = ca_cfar_detector(rd_map_db, pfa, guard_cells, training_cells)
        detection_clusters = cluster_detections(detections)
        
        # Top row: Range-Doppler Map
        ax1 = axes[0, pfa_idx]
        im1 = ax1.imshow(rd_map_db, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title(f'Range-Doppler Map\nSample {sample_idx}, PFA: {pfa:.0e}')
        ax1.set_xlabel('Doppler Bin')
        ax1.set_ylabel('Range Bin')
        plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
        
        # Mark true targets
        for r_idx, d_idx in true_targets:
            ax1.plot(d_idx, r_idx, 'ro', markersize=8, markerfacecolor='none', 
                    markeredgewidth=2, label='True Targets' if (r_idx, d_idx) == true_targets[0] else "")
        
        # Bottom row: CFAR Detections
        ax2 = axes[1, pfa_idx]
        im2 = ax2.imshow(rd_map_db, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title(f'CFAR Detections\nPFA: {pfa:.0e}')
        ax2.set_xlabel('Doppler Bin')
        ax2.set_ylabel('Range Bin')
        plt.colorbar(im2, ax=ax2, label='Magnitude (dB)')
        
        # Mark true targets
        for r_idx, d_idx in true_targets:
            ax2.plot(d_idx, r_idx, 'ro', markersize=8, markerfacecolor='none', 
                    markeredgewidth=2, label='True Targets' if (r_idx, d_idx) == true_targets[0] else "")
        
        # Mark CFAR detections
        detection_points = np.where(detections == 1)
        if len(detection_points[0]) > 0:
            ax2.plot(detection_points[1], detection_points[0], 'go', markersize=4, 
                    markerfacecolor='none', markeredgewidth=1, label='CFAR Detections')
        
        # Mark detection clusters (centroids)
        if detection_clusters:
            cluster_array = np.array(detection_clusters)
            ax2.plot(cluster_array[:, 1], cluster_array[:, 0], 'yo', markersize=10, 
                    markerfacecolor='none', markeredgewidth=2, label='Detection Clusters')
        
        if pfa_idx == 0:  # Add legend only to first subplot
            ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    filename = f'rd_map_sample_{sample_idx}_cfar_comparison.png'
    plt.savefig(f'snr_plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")

# Plot PFA analysis results
plt.figure(figsize=(15, 10))

# Plot 1: Detection Probability vs PFA
plt.subplot(2, 3, 1)
pfa_list = list(results.keys())
pd_list = [results[pfa]['pd'] for pfa in pfa_list]
pfa_empirical_list = [results[pfa]['pfa_empirical'] for pfa in pfa_list]

plt.semilogx(pfa_list, pd_list, 'bo-', linewidth=2, markersize=8, label='Detection Probability')
plt.xlabel('Design PFA')
plt.ylabel('Detection Probability')
plt.title('CA-CFAR Detection Probability vs Design PFA')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1.05)

# Plot 2: Empirical vs Design PFA
plt.subplot(2, 3, 2)
plt.loglog(pfa_list, pfa_empirical_list, 'ro-', linewidth=2, markersize=8, label='Empirical PFA')
plt.loglog(pfa_list, pfa_list, 'k--', linewidth=1, label='Design PFA')
plt.xlabel('Design PFA')
plt.ylabel('Empirical PFA')
plt.title('Empirical vs Design PFA')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: True Positives, False Positives, False Negatives
plt.subplot(2, 3, 3)
tp_totals = []
fp_totals = []
fn_totals = []

for pfa in pfa_list:
    tp_totals.append(sum(results[pfa]['tp']))
    fp_totals.append(sum(results[pfa]['fp']))
    fn_totals.append(sum(results[pfa]['fn']))

x_pos = np.arange(len(pfa_list))
width = 0.25

plt.bar(x_pos - width, tp_totals, width, label='True Positives', color='green', alpha=0.7)
plt.bar(x_pos, fp_totals, width, label='False Positives', color='red', alpha=0.7)
plt.bar(x_pos + width, fn_totals, width, label='False Negatives', color='orange', alpha=0.7)

plt.xlabel('PFA Configuration')
plt.ylabel('Count')
plt.title('Detection Counts by PFA')
plt.xticks(x_pos, [f'{pfa:.0e}' for pfa in pfa_list])
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: ROC Curve
plt.subplot(2, 3, 4)
plt.loglog(pfa_empirical_list, pd_list, 'bo-', linewidth=2, markersize=8)
plt.xlabel('False Alarm Rate (Empirical)')
plt.ylabel('Detection Probability')
plt.title('ROC Curve')
plt.grid(True, alpha=0.3)

# Plot 5: Performance Summary Table
plt.subplot(2, 3, 5)
plt.axis('off')
summary_text = "CA-CFAR PFA Analysis Summary\n\n"
summary_text += f"Number of random samples tested: {len(random_sample_indices)}\n"
summary_text += f"Total samples in dataset: {len(data['sequences'])}\n"
summary_text += f"Dataset: {dataset_path.split('/')[-1]}\n"
summary_text += f"Data format: dB values\n"
summary_text += f"Target matching distance: {max_distance} pixels\n"
summary_text += f"Guard cells: {guard_cells}\n"
summary_text += f"Training cells: {training_cells}\n"
summary_text += f"Range bins: {num_range_bins}\n"
summary_text += f"Doppler bins: {num_doppler_bins}\n\n"

summary_text += "Results by PFA:\n"
for pfa in pfa_list:
    summary_text += f"PFA {pfa:.0e}:\n"
    summary_text += f"  Pd = {results[pfa]['pd']:.3f}\n"
    summary_text += f"  Empirical PFA = {results[pfa]['pfa_empirical']:.2e}\n"
    summary_text += f"  TP = {sum(results[pfa]['tp'])}\n"
    summary_text += f"  FP = {sum(results[pfa]['fp'])}\n"
    summary_text += f"  FN = {sum(results[pfa]['fn'])}\n\n"

plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='top', fontfamily='monospace')

# Plot 6: Detection Performance by Sample
plt.subplot(2, 3, 6)
# Show performance variation across samples for the best PFA
best_pfa = pfa_list[np.argmax(pd_list)]
sample_indices = results[best_pfa]['sample_idx']
sample_pd = []

for i, sample_idx in enumerate(sample_indices):
    tp = results[best_pfa]['tp'][i]
    fn = results[best_pfa]['fn'][i]
    total_targets_sample = tp + fn
    if total_targets_sample > 0:
        sample_pd.append(tp / total_targets_sample)
    else:
        sample_pd.append(0)

plt.hist(sample_pd, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Detection Probability per Sample')
plt.ylabel('Frequency')
plt.title(f'Per-Sample Detection Performance\n(PFA = {best_pfa:.0e})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('snr_plots/ca_cfar_pfa_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print numerical results
print("\nNumerical Results Summary:")
print("="*60)
print(f"{'PFA':<12} {'Pd':<8} {'Emp. PFA':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
print("-" * 60)
for pfa in pfa_list:
    pd = results[pfa]['pd']
    emp_pfa = results[pfa]['pfa_empirical']
    tp = sum(results[pfa]['tp'])
    fp = sum(results[pfa]['fp'])
    fn = sum(results[pfa]['fn'])
    print(f"{pfa:<12.0e} {pd:<8.3f} {emp_pfa:<12.2e} {tp:<6d} {fp:<6d} {fn:<6d}")

print(f"\nBest Detection Probability: {max(pd_list):.3f} at PFA = {pfa_list[np.argmax(pd_list)]:.0e}")
print(f"Dataset: {dataset_path}")
print(f"Analysis completed successfully!")
