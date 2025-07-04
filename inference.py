import torch
import models
from sea_clutter import create_data_loaders
import numpy as np

_, val_loader, test_loader = create_data_loaders(
    dataset_path="/Users/pepijnlens/Documents/SeaClutterSuppression/local_data/noise.pt",
    batch_size=1,  # Load one sample at a time
    train_ratio=0.01,
    val_ratio=0.01,  # No validation set needed for this analysis
    test_ratio=0.98  # Use all data for testing
)

dataset = test_loader.dataset

sigmoid_segmentation_maps = []

unet = models.UNet(n_channels=3, n_classes=1, base_filters=64).to('mps')
unet.load_state_dict(torch.load("pretrained/behemoth.pt", map_location='mps' if torch.backends.mps.is_available() else 'cpu'))

unet.eval()

with torch.no_grad():
    for i, (image, _, _) in enumerate(dataset):
        image = image.unsqueeze(0).to('mps')  # Add batch dimension
        segmentation_map = unet(image).squeeze(0)
        sigmoid_segmentation_map = torch.sigmoid(segmentation_map.squeeze(0))
        sigmoid_segmentation_maps.append(sigmoid_segmentation_map.detach().cpu().numpy())

# Get all sigmoid values from all individual samples
all_sigmoid_values = np.concatenate([s.flatten() for s in sigmoid_segmentation_maps])
total_pixels = len(all_sigmoid_values)
num_images = len(sigmoid_segmentation_maps)
pixels_per_image = 128 * 128

# Calculate mean sigmoid map
mean_sigmoid_map = np.mean(sigmoid_segmentation_maps, axis=0)

print(f"Dataset info (no targets):")
print(f"Total images: {num_images}")
print(f"Pixels per image: {pixels_per_image}")
print(f"Total pixels: {total_pixels}")
print(f"Mean of all sigmoid values: {np.mean(all_sigmoid_values):.10f}")
print(f"Max sigmoid value: {np.max(all_sigmoid_values):.10f}")
print(f"Min sigmoid value: {np.min(all_sigmoid_values):.10f}")

# Sort all sigmoid values in descending order
sorted_sigmoid_values = np.sort(all_sigmoid_values)[::-1]

# Calculate thresholds for different false alarm rates
far_rates = [1e-2, 1e-3, 1e-4, 1e-5]

# Clustering information based on experiments
avg_cluster_size_targets_no_clutter = 1.9  # Average cluster size for targets without clutter

print(f"\nThresholds for False Alarm Rates (accounting for clustering):")
print(f"Using average cluster size of {avg_cluster_size_targets_no_clutter} for noise-only images")

for far in far_rates:
    # Calculate the number of false alarm clusters we want
    target_false_alarm_clusters = int(far * total_pixels)
    
    # Convert clusters to expected pixels (clusters × average cluster size)
    target_false_alarm_pixels = int(target_false_alarm_clusters * avg_cluster_size_targets_no_clutter)
    
    if target_false_alarm_pixels == 0:
        threshold = np.max(all_sigmoid_values)
        false_alarm_pixels = 0
        false_alarm_clusters = 0
    elif target_false_alarm_pixels >= len(sorted_sigmoid_values):
        threshold = np.min(all_sigmoid_values)
        false_alarm_pixels = len(sorted_sigmoid_values)
        false_alarm_clusters = int(false_alarm_pixels / avg_cluster_size_targets_no_clutter)
    else:
        threshold = sorted_sigmoid_values[target_false_alarm_pixels - 1]
        false_alarm_pixels = np.sum(all_sigmoid_values > threshold)
        false_alarm_clusters = int(false_alarm_pixels / avg_cluster_size_targets_no_clutter)
    
    actual_far = false_alarm_clusters / total_pixels
    avg_false_alarm_clusters_per_image = false_alarm_clusters / num_images
    avg_false_alarm_pixels_per_image = false_alarm_pixels / num_images
    
    print(f"FAR {far}:")
    print(f"  Target false alarm clusters: {target_false_alarm_clusters}")
    print(f"  Expected false alarm pixels: {target_false_alarm_pixels}")
    print(f"  Threshold: {threshold:.10f}")
    print(f"  Actual false alarm pixels: {false_alarm_pixels}")
    print(f"  Estimated false alarm clusters: {false_alarm_clusters}")
    print(f"  Actual FAR (clusters): {actual_far:.2e}")
    print(f"  Avg false alarm clusters per image: {avg_false_alarm_clusters_per_image:.1f}")
    print(f"  Avg false alarm pixels per image: {avg_false_alarm_pixels_per_image:.1f}")
    print()




