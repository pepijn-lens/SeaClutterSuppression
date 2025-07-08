import torch
import models
from sea_clutter import create_data_loaders
import numpy as np

_, val_loader, test_loader = create_data_loaders(
    dataset_path="/Users/pepijnlens/Documents/SeaClutterSuppression/data/noise.pt",
    batch_size=1,  # Load one sample at a time
    train_ratio=0.01,
    val_ratio=0.01,  # No validation set needed for this analysis
    test_ratio=0.98  # Use all data for testing
)

dataset = test_loader.dataset

sigmoid_segmentation_maps = []

unet = models.UNet(n_channels=3, n_classes=1, base_filters=64).to('mps')
unet.load_state_dict(torch.load("pretrained/tversky.pt", map_location='mps' if torch.backends.mps.is_available() else 'cpu'))

unet.eval()

with torch.no_grad():
    for i, (image, _, _) in enumerate(dataset):
        image = image.unsqueeze(0).to('mps')  # Add batch dimension
        segmentation_map = unet(image).squeeze(0)
        # Convert to CPU first, then to double precision before sigmoid for more precision
        sigmoid_segmentation_map = torch.sigmoid(segmentation_map.squeeze(0).cpu().double())
        sigmoid_segmentation_maps.append(sigmoid_segmentation_map.detach().numpy().astype(np.float64))

# Get all sigmoid values from all individual samples - use float64
all_sigmoid_values = np.concatenate([s.flatten() for s in sigmoid_segmentation_maps]).astype(np.float64)
total_pixels = len(all_sigmoid_values)
num_images = len(sigmoid_segmentation_maps)
pixels_per_image = 128 * 128

# Calculate mean sigmoid map
mean_sigmoid_map = np.mean(sigmoid_segmentation_maps, axis=0)

print(f"Dataset info (no targets):")
print(f"Total images: {num_images}")
print(f"Pixels per image: {pixels_per_image}")
print(f"Total pixels: {total_pixels}")
print(f"Mean of all sigmoid values: {np.mean(all_sigmoid_values):.15f}")
print(f"Max sigmoid value: {np.max(all_sigmoid_values):.15f}")
print(f"Min sigmoid value: {np.min(all_sigmoid_values):.15f}")

# Sort all sigmoid values in descending order - maintain float64 precision
sorted_sigmoid_values = np.sort(all_sigmoid_values)[::-1]

# Calculate thresholds for different false alarm rates
far_rates = [1e-4, 1e-3]

print(f"\nThresholds for False Alarm Rates (pixel-wise):")

for far in far_rates:
    # Calculate the number of false alarm pixels we want
    target_false_alarm_pixels = int(far * total_pixels)
    
    if target_false_alarm_pixels == 0:
        threshold = np.max(all_sigmoid_values)
        false_alarm_pixels = 0
    elif target_false_alarm_pixels >= len(sorted_sigmoid_values):
        threshold = np.min(all_sigmoid_values)
        false_alarm_pixels = len(sorted_sigmoid_values)
    else:
        threshold = sorted_sigmoid_values[target_false_alarm_pixels - 1]
        false_alarm_pixels = np.sum(all_sigmoid_values > threshold)
    
    actual_far = false_alarm_pixels / total_pixels
    avg_false_alarm_pixels_per_image = false_alarm_pixels / num_images
    
    print(f"FAR {far}:")
    print(f"  Target false alarm pixels: {target_false_alarm_pixels}")
    print(f"  Threshold: {threshold}")
    print(f"  Actual false alarm pixels: {false_alarm_pixels}")
    print(f"  Actual FAR (pixels): {actual_far:.2e}")
    print(f"  Avg false alarm pixels per image: {avg_false_alarm_pixels_per_image:.1f}")
    print()




