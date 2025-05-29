import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
import time

from load_segmentation_data import create_data_loaders  # Your dataset file
from sklearn.metrics import precision_score, recall_score

# ---------------------------
# 1. U-Net Architecture
# ---------------------------
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
    def __init__(self, n_channels=1, n_classes=1):
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

# ---------------------------
# 2. Metrics
# ---------------------------
def dice_coeff(pred, target, smooth=1.):
    pred = torch.sigmoid(pred).detach().cpu().numpy() > 0.5
    target = target.cpu().numpy()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    dice_total, prec_total, recall_total = 0, 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            dice_total += dice_coeff(outputs, masks)
            prec_total += precision_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), zero_division=0)
            recall_total += recall_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), zero_division=0)

    n = len(loader)
    return dice_total / n, prec_total / n, recall_total / n

# ---------------------------
# 3. Training Loop
# ---------------------------
def train_model(dataset_path: str, num_epochs=25, batch_size=16, lr=1e-4, pretrained=None, model_save_path='unet_sea_clutter.pth'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        normalize=False
    )

    model = UNet().to(device)
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=device))
        print(f"Loaded pretrained model from {pretrained}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping variables
    best_dice = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation after every epoch
        dice, prec, recall = evaluate(model, val_loader, device)
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | Dice: {dice:.3f} | Precision: {prec:.3f} | Recall: {recall:.3f}")

        # Early stopping logic
        if dice > best_dice:
            best_dice = dice
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best validation Dice: {best_dice:.3f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Best model saved to {model_save_path} with Dice score: {best_dice:.3f}")

# ---------------------------
# 4. Interpret Model Results
# ---------------------------
def interpret_model_results(model_path: str, dataset_path: str, batch_size=16, num_samples=5, save_plots=True):
    """
    Interpret and visualize the results of the trained model.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        batch_size: Batch size for data loading
        num_samples: Number of sample predictions to visualize
        save_plots: Whether to save plots to disk
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        normalize=False
    )
    
    # Use test set if available, otherwise validation set
    eval_loader = test_loader if test_loader.dataset.__len__() > 0 else val_loader
    
    print("=" * 60)
    print("MODEL INTERPRETATION RESULTS")
    print("=" * 60)
    
    # 1. Overall Performance Metrics
    dice, precision, recall = evaluate(model, eval_loader, device)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nOverall Performance on {'Test' if test_loader.dataset.__len__() > 0 else 'Validation'} Set:")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # 2. Detailed per-batch analysis
    all_predictions = []
    all_targets = []
    all_images = []
    confidence_scores = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(eval_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Store for analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(masks.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            confidence_scores.extend(probabilities.cpu().numpy())
            
            if i >= num_samples - 1:  # Limit samples for visualization
                break
    
    # 3. Confusion Matrix Analysis
    flat_preds = np.array(all_predictions).flatten()
    flat_targets = np.array(all_targets).flatten()
    
    cm = confusion_matrix(flat_targets, flat_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix Analysis:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"Specificity: {tn/(tn+fp):.4f}")
    print(f"IoU (Jaccard): {tp/(tp+fp+fn):.4f}")
    
    # 4. Confidence Analysis
    conf_array = np.array(confidence_scores)
    print(f"\nPrediction Confidence Analysis:")
    print(f"Mean Confidence: {conf_array.mean():.4f}")
    print(f"Std Confidence: {conf_array.std():.4f}")
    print(f"Min Confidence: {conf_array.min():.4f}")
    print(f"Max Confidence: {conf_array.max():.4f}")
    
    # 5. Visualizations
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(all_images))):
        image = all_images[i][0]  # Remove channel dimension for visualization
        target = all_targets[i][0]
        prediction = all_predictions[i][0]
        confidence = confidence_scores[i][0]
        
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(target, cmap='Reds', alpha=0.7)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(prediction, cmap='Blues', alpha=0.7)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Confidence map
        im = axes[i, 3].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[i, 3].set_title('Confidence Map')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # Calculate sample-specific metrics
        sample_dice = dice_coeff(torch.tensor(prediction).unsqueeze(0).unsqueeze(0), 
                                torch.tensor(target).unsqueeze(0).unsqueeze(0))
        print(f"Sample {i+1} Dice: {sample_dice:.4f}")
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('model_interpretation_results.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as 'model_interpretation_results.png'")
    plt.show()
    
    # 6. Error Analysis
    print(f"\nError Analysis:")
    errors = np.abs(flat_preds - flat_targets)
    error_rate = errors.mean()
    print(f"Pixel-wise Error Rate: {error_rate:.4f}")
    
    # Find worst performing samples
    sample_dice_scores = []
    for i in range(len(all_predictions)):
        sample_dice = dice_coeff(torch.tensor(all_predictions[i]).unsqueeze(0), 
                                torch.tensor(all_targets[i]).unsqueeze(0))
        sample_dice_scores.append(sample_dice)
    
    worst_idx = np.argmin(sample_dice_scores)
    best_idx = np.argmax(sample_dice_scores)
    
    print(f"Best sample Dice: {sample_dice_scores[best_idx]:.4f} (Sample {best_idx})")
    print(f"Worst sample Dice: {sample_dice_scores[worst_idx]:.4f} (Sample {worst_idx})")
    
    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm,
        'confidence_stats': {
            'mean': conf_array.mean(),
            'std': conf_array.std(),
            'min': conf_array.min(),
            'max': conf_array.max()
        }
    }

# ---------------------------
# 5. Analyze Specific Sample
# ---------------------------
def plot_specific_sample(model_path: str, dataset_path: str, sample_idx: int, batch_size=16):
    """
    Plot results for a specific sample index.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        sample_idx: Index of the sample to visualize
        batch_size: Batch size for data loading
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        normalize=False
    )
    
    eval_loader = test_loader if test_loader.dataset.__len__() > 0 else val_loader
    
    # Collect all samples to find the specific one
    all_images = []
    all_targets = []
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for images, masks in eval_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_images.extend(images.cpu().numpy())
            all_targets.extend(masks.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(probabilities.cpu().numpy())
    
    # Check if sample_idx is valid
    if sample_idx >= len(all_images):
        print(f"Sample index {sample_idx} is out of range. Total samples: {len(all_images)}")
        return
    
    # Get the specific sample
    image = all_images[sample_idx][0]
    target = all_targets[sample_idx][0]
    prediction = all_predictions[sample_idx][0]
    confidence = all_confidences[sample_idx][0]
    
    # Calculate metrics for this sample
    sample_dice = dice_coeff(torch.tensor(prediction).unsqueeze(0).unsqueeze(0), 
                            torch.tensor(target).unsqueeze(0).unsqueeze(0))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Original Image (Sample {sample_idx})')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(target, cmap='Reds', alpha=0.7)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[0, 2].imshow(prediction, cmap='Blues', alpha=0.7)
    axes[0, 2].set_title('Prediction')
    axes[0, 2].axis('off')
    
    # Confidence map
    im1 = axes[1, 0].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Confidence Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay: Original + Ground Truth
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(target, cmap='Reds', alpha=0.5)
    axes[1, 1].set_title('Original + Ground Truth')
    axes[1, 1].axis('off')
    
    # Overlay: Original + Prediction
    axes[1, 2].imshow(image)
    axes[1, 2].imshow(prediction, cmap='Blues', alpha=0.5)
    axes[1, 2].set_title('Original + Prediction')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Sample {sample_idx} Analysis - Dice Score: {sample_dice:.4f}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'sample_{sample_idx}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} analysis saved as 'sample_{sample_idx}_analysis.png'")
    plt.show()
    
    # Print detailed metrics for this sample
    print(f"\nDetailed metrics for Sample {sample_idx}:")
    print(f"Dice Coefficient: {sample_dice:.4f}")
    print(f"Mean Confidence: {confidence.mean():.4f}")
    print(f"Std Confidence: {confidence.std():.4f}")
    print(f"Min Confidence: {confidence.min():.4f}")
    print(f"Max Confidence: {confidence.max():.4f}")
    
    # Pixel-wise analysis
    flat_pred = prediction.flatten()
    flat_target = target.flatten()
    tp = np.sum((flat_pred == 1) & (flat_target == 1))
    fp = np.sum((flat_pred == 1) & (flat_target == 0))
    fn = np.sum((flat_pred == 0) & (flat_target == 1))
    tn = np.sum((flat_pred == 0) & (flat_target == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")

# ---------------------------
# 6. Inference Time Measurement
# ---------------------------
def measure_inference_time(model_path: str, dataset_path: str, batch_size=16, num_batches=10, warmup_batches=5):
    """
    Measure inference time of the U-Net model.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        batch_size: Batch size for inference
        num_batches: Number of batches to time
        warmup_batches: Number of warmup batches (excluded from timing)
    
    Returns:
        dict: Timing statistics
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        normalize=False
    )
    
    eval_loader = test_loader if test_loader.dataset.__len__() > 0 else val_loader
    
    print("=" * 60)
    print("INFERENCE TIME MEASUREMENT")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup batches: {warmup_batches}")
    print(f"Measurement batches: {num_batches}")
    
    batch_times = []
    sample_times = []
    
    with torch.no_grad():
        batch_count = 0
        
        for i, (images, masks) in enumerate(eval_loader):
            images = images.to(device)
            
            # Warmup phase
            if i < warmup_batches:
                _ = model(images)
                print(f"Warmup batch {i+1}/{warmup_batches}")
                continue
            
            # Timing phase
            if batch_count >= num_batches:
                break
                
            # Measure batch inference time
            start_time = time.time()
            outputs = model(images)
            
            # For MPS, we need to synchronize
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            batch_time = end_time - start_time
            sample_time = batch_time / images.shape[0]  # Time per sample
            
            batch_times.append(batch_time)
            sample_times.append(sample_time)
            
            print(f"Batch {batch_count+1}/{num_batches}: {batch_time*1000:.2f}ms ({sample_time*1000:.2f}ms per sample)")
            batch_count += 1
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    sample_times = np.array(sample_times)
    
    stats = {
        'batch_times': {
            'mean': batch_times.mean(),
            'std': batch_times.std(),
            'min': batch_times.min(),
            'max': batch_times.max(),
            'median': np.median(batch_times)
        },
        'sample_times': {
            'mean': sample_times.mean(),
            'std': sample_times.std(),
            'min': sample_times.min(),
            'max': sample_times.max(),
            'median': np.median(sample_times)
        },
        'throughput': {
            'samples_per_second': 1.0 / sample_times.mean(),
            'batches_per_second': 1.0 / batch_times.mean()
        }
    }
    
    print("\n" + "=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"\nBatch Timing (batch_size={batch_size}):")
    print(f"  Mean: {stats['batch_times']['mean']*1000:.2f}ms")
    print(f"  Std:  {stats['batch_times']['std']*1000:.2f}ms")
    print(f"  Min:  {stats['batch_times']['min']*1000:.2f}ms")
    print(f"  Max:  {stats['batch_times']['max']*1000:.2f}ms")
    print(f"  Median: {stats['batch_times']['median']*1000:.2f}ms")
    
    print(f"\nPer-Sample Timing:")
    print(f"  Mean: {stats['sample_times']['mean']*1000:.2f}ms")
    print(f"  Std:  {stats['sample_times']['std']*1000:.2f}ms")
    print(f"  Min:  {stats['sample_times']['min']*1000:.2f}ms")
    print(f"  Max:  {stats['sample_times']['max']*1000:.2f}ms")
    print(f"  Median: {stats['sample_times']['median']*1000:.2f}ms")
    
    print(f"\nThroughput:")
    print(f"  Samples per second: {stats['throughput']['samples_per_second']:.2f}")
    print(f"  Batches per second: {stats['throughput']['batches_per_second']:.2f}")
    
    return stats

def compare_batch_sizes(model_path: str, dataset_path: str, batch_sizes=[1, 4, 8, 16, 32], num_batches=10):
    """
    Compare inference times across different batch sizes.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        batch_sizes: List of batch sizes to test
        num_batches: Number of batches to time for each size
    """
    print("=" * 60)
    print("BATCH SIZE COMPARISON")
    print("=" * 60)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        try:
            stats = measure_inference_time(
                model_path=model_path,
                dataset_path=dataset_path,
                batch_size=batch_size,
                num_batches=num_batches,
                warmup_batches=3
            )
            results[batch_size] = stats
        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            continue
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("BATCH SIZE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Avg Batch Time':<15} {'Avg Sample Time':<17} {'Samples/sec':<12}")
    print("-" * 80)
    
    for batch_size, stats in results.items():
        batch_time = stats['batch_times']['mean'] * 1000
        sample_time = stats['sample_times']['mean'] * 1000
        throughput = stats['throughput']['samples_per_second']
        print(f"{batch_size:<12} {batch_time:<15.2f}ms {sample_time:<17.2f}ms {throughput:<12.2f}")
    
    return results

# ---------------------------
# 6. Entry Point (Updated)
# ---------------------------
if __name__ == "__main__":
    dataset_file = "/Users/pepijnlens/Documents/transformers/data/sea_clutter_segmentation_lowSCR.pt"
    model_file = "unet_sea_clutter.pth"
    
    # # Train the model
    # train_model(dataset_file, pretrained='unet_sea_clutter-high.pth')
    
    # # Interpret the results
    # print("\n" + "="*60)
    # print("INTERPRETING MODEL RESULTS...")
    # print("="*60)
    # interpret_model_results(model_file, dataset_file, num_samples=5)
    
    # Plot specific sample (sample 9)
    print("\n" + "="*60)
    print("ANALYZING SAMPLE 9...")
    print("="*60)
    for i in range(10):
        print(f"Analyzing sample {i}...")
        plot_specific_sample(model_file, dataset_file, sample_idx=i+233)

    # plot_specific_sample(model_file, dataset_file, sample_idx=987)
    
    # # Measure inference time
    # print("\n" + "="*60)
    # print("MEASURING INFERENCE TIME...")
    # print("="*60)
    
    # # Single batch size measurement
    # measure_inference_time(model_file, dataset_file, batch_size=16, num_batches=20)
    
    # # Compare different batch sizes
    # compare_batch_sizes(model_file, dataset_file, batch_sizes=[1, 4, 8, 16, 32])
