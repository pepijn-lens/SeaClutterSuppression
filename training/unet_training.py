import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
import time
import logging

import os
import seaborn as sns
from scipy import stats

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
def train_model(dataset_path: str, num_epochs=30, batch_size=16, lr=1e-4, pretrained=None, model_save_path='unet_sea_clutter.pt'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        mask_strategy='last',
    )

    model = UNet(n_channels=3).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params:,} trainable parameters')
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=device))
        print(f"Loaded pretrained model from {pretrained}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping variables
    best_dice = 0.0
    patience = 8
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
        if epoch % 5 == 0 or epoch == num_epochs - 1:
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
def interpret_model_results(model_path: str, dataset_path: str, batch_size=16, num_samples=5, save_plots=None, log_file=None):
    """
    Interpret and visualize the results of the trained model.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        batch_size: Batch size for data loading
        num_samples: Number of sample predictions to visualize
        save_plots: Whether to save plots to disk
        log_file: Path to log file (if None, prints to console)
    """
    # Setup logging
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()  # Also print to console
            ]
        )
        logger = logging.getLogger(__name__)
        log_func = logger.info
    else:
        log_func = print
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_channels=3).to(device)  # Add n_channels=3 here
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
    )
    
    # Use test set if available, otherwise validation set
    eval_loader = test_loader if test_loader.dataset.__len__() > 0 else val_loader
    
    log_func("=" * 60)
    log_func("MODEL INTERPRETATION RESULTS")
    log_func("=" * 60)
    
    # 1. Overall Performance Metrics
    dice, precision, recall = evaluate(model, eval_loader, device)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    log_func(f"\nOverall Performance on {'Test' if test_loader.dataset.__len__() > 0 else 'Validation'} Set:")
    log_func(f"Dice Coefficient: {dice:.4f}")
    log_func(f"Precision: {precision:.4f}")
    log_func(f"Recall: {recall:.4f}")
    log_func(f"F1-Score: {f1_score:.4f}")
    
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
    
    log_func(f"\nConfusion Matrix Analysis:")
    log_func(f"True Negatives: {tn}")
    log_func(f"False Positives: {fp}")
    log_func(f"False Negatives: {fn}")
    log_func(f"True Positives: {tp}")
    log_func(f"Specificity: {tn/(tn+fp):.4f}")
    log_func(f"IoU (Jaccard): {tp/(tp+fp+fn):.4f}")
    
    # 4. Confidence Analysis
    conf_array = np.array(confidence_scores)
    log_func(f"\nPrediction Confidence Analysis:")
    log_func(f"Mean Confidence: {conf_array.mean():.4f}")
    log_func(f"Std Confidence: {conf_array.std():.4f}")
    log_func(f"Min Confidence: {conf_array.min():.4f}")
    log_func(f"Max Confidence: {conf_array.max():.4f}")
    
    # 5. Visualizations
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(all_images))):
        # For sequence data, we need to handle the 3-channel input differently
        # Show the middle frame for visualization
        if all_images[i].shape[0] == 3:  # 3 frames as channels
            image = all_images[i][1]  # Use middle frame (index 1)
        else:
            image = all_images[i][0]  # Single channel
            
        target = all_targets[i][0]
        prediction = all_predictions[i][0]
        confidence = confidence_scores[i][0]
        
        # Original image (middle frame)
        axes[i, 0].imshow(image, cmap='viridis', aspect='auto', origin='lower')
        axes[i, 0].set_title(f'Input Frame {i+1} (Last Frame)')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(target, cmap='Reds', alpha=0.7, aspect='auto', origin='lower')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(prediction, cmap='Blues', alpha=0.7, aspect='auto', origin='lower')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Confidence map
        im = axes[i, 3].imshow(confidence, cmap='viridis', vmin=0, vmax=1, aspect='auto', origin='lower')
        axes[i, 3].set_title('Confidence Map')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # Calculate sample-specific metrics
        sample_dice = dice_coeff(torch.tensor(prediction).unsqueeze(0).unsqueeze(0), 
                                torch.tensor(target).unsqueeze(0).unsqueeze(0))
        log_func(f"Sample {i+1} Dice: {sample_dice:.4f}")
    
    plt.tight_layout()
    if save_plots is not None:
        plt.savefig(save_plots, dpi=300, bbox_inches='tight')
        log_func(f"\nVisualization saved as {save_plots}")
    else:
        plt.show()
    
    # 6. Error Analysis
    log_func(f"\nError Analysis:")
    errors = np.abs(flat_preds - flat_targets)
    error_rate = errors.mean()
    log_func(f"Pixel-wise Error Rate: {error_rate:.4f}")
    
    # Find worst performing samples
    sample_dice_scores = []
    for i in range(len(all_predictions)):
        sample_dice = dice_coeff(torch.tensor(all_predictions[i]).unsqueeze(0), 
                                torch.tensor(all_targets[i]).unsqueeze(0))
        sample_dice_scores.append(sample_dice)
    
    worst_idx = np.argmin(sample_dice_scores)
    best_idx = np.argmax(sample_dice_scores)
    
    log_func(f"Best sample Dice: {sample_dice_scores[best_idx]:.4f} (Sample {best_idx})")
    log_func(f"Worst sample Dice: {sample_dice_scores[worst_idx]:.4f} (Sample {worst_idx})")
    
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


def save_worst_samples(model_path: str, dataset_path: str, n_worst: int = 10, 
                        batch_size: int = 16, save_dir: str = "worst_samples"):
    """
    Find and save the worst N samples from the dataset based on Dice coefficient.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        n_worst: Number of worst samples to save
        batch_size: Batch size for data loading
        save_dir: Directory to save the worst samples
    """
    import os
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
    )
    
    eval_loader = test_loader if test_loader.dataset.__len__() > 0 else val_loader
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Evaluating all samples to find {n_worst} worst performers...")
    
    # Collect all samples with their metrics
    sample_data = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(eval_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Process each sample in the batch
            for i in range(images.shape[0]):
                image = images[i].cpu().numpy()
                target = masks[i].cpu().numpy()
                prediction = predictions[i].cpu().numpy()
                confidence = probabilities[i].cpu().numpy()
                
                # Calculate Dice coefficient for this sample
                sample_dice = dice_coeff(outputs[i:i+1], masks[i:i+1])
                
                # Store sample data
                sample_data.append({
                    'sample_idx': sample_idx,
                    'batch_idx': batch_idx,
                    'within_batch_idx': i,
                    'dice_score': sample_dice,
                    'image': image,
                    'target': target,
                    'prediction': prediction,
                    'confidence': confidence
                })
                
                sample_idx += 1
    
    # Sort by Dice score (ascending - worst first)
    sample_data.sort(key=lambda x: x['dice_score'])
    
    # Take the worst N samples
    worst_samples = sample_data[:n_worst]
    
    print(f"Found {len(worst_samples)} worst samples. Saving visualizations...")
    
    # Create a summary plot with all worst samples
    fig, axes = plt.subplots(n_worst, 4, figsize=(16, 4*n_worst))
    if n_worst == 1:
        axes = axes.reshape(1, -1)
    
    # Save individual samples and create summary
    summary_data = []
    
    for i, sample in enumerate(worst_samples):
        # Extract data
        if sample['image'].shape[0] == 3:  # 3 frames as channels
            display_image = sample['image'][1]  # Use middle frame
        else:
            display_image = sample['image'][0]
            
        target = sample['target'][0]
        prediction = sample['prediction'][0]
        confidence = sample['confidence'][0]
        
        # Calculate detailed metrics for this sample
        flat_pred = prediction.flatten()
        flat_target = target.flatten()
        tp = np.sum((flat_pred == 1) & (flat_target == 1))
        fp = np.sum((flat_pred == 1) & (flat_target == 0))
        fn = np.sum((flat_pred == 0) & (flat_target == 1))
        tn = np.sum((flat_pred == 0) & (flat_target == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store summary data
        summary_data.append({
            'rank': i + 1,
            'sample_idx': sample['sample_idx'],
            'dice_score': sample['dice_score'],
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'mean_confidence': confidence.mean(),
            'std_confidence': confidence.std(),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
        
        # Create individual sample plot
        fig_single, axes_single = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes_single[0, 0].imshow(display_image, cmap='viridis', aspect='auto', origin='lower')
        axes_single[0, 0].set_title(f'Input (Sample {sample["sample_idx"]})')
        axes_single[0, 0].axis('off')
        
        # Ground truth
        axes_single[0, 1].imshow(target, cmap='Reds', alpha=0.7, aspect='auto', origin='lower')
        axes_single[0, 1].set_title('Ground Truth')
        axes_single[0, 1].axis('off')
        
        # Prediction
        axes_single[0, 2].imshow(prediction, cmap='Blues', alpha=0.7, aspect='auto', origin='lower')
        axes_single[0, 2].set_title('Prediction')
        axes_single[0, 2].axis('off')
        
        # Confidence map
        im1 = axes_single[1, 0].imshow(confidence, cmap='viridis', vmin=0, vmax=1, aspect='auto', origin='lower')
        axes_single[1, 0].set_title('Confidence Map')
        axes_single[1, 0].axis('off')
        plt.colorbar(im1, ax=axes_single[1, 0], fraction=0.046, pad=0.04)
        
        # Overlay: Original + Ground Truth
        axes_single[1, 1].imshow(display_image, cmap='viridis', aspect='auto', origin='lower')
        axes_single[1, 1].imshow(target, cmap='Reds', alpha=0.5, aspect='auto', origin='lower')
        axes_single[1, 1].set_title('Input + Ground Truth')
        axes_single[1, 1].axis('off')
        
        # Overlay: Original + Prediction
        axes_single[1, 2].imshow(display_image, cmap='viridis', aspect='auto', origin='lower')
        axes_single[1, 2].imshow(prediction, cmap='Blues', alpha=0.5, aspect='auto', origin='lower')
        axes_single[1, 2].set_title('Input + Prediction')
        axes_single[1, 2].axis('off')
        
        plt.suptitle(f'Worst Sample #{i+1} (Global Index: {sample["sample_idx"]}) - Dice: {sample["dice_score"]:.4f}', fontsize=16)
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(save_dir, f'worst_sample_{i+1:02d}_idx_{sample["sample_idx"]}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        
        # Add to summary plot
        axes[i, 0].imshow(display_image, cmap='viridis', aspect='auto', origin='lower')
        axes[i, 0].set_title(f'#{i+1}: Sample {sample["sample_idx"]}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target, cmap='Reds', alpha=0.7, aspect='auto', origin='lower')
        axes[i, 1].set_title(f'GT (Dice: {sample["dice_score"]:.3f})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(prediction, cmap='Blues', alpha=0.7, aspect='auto', origin='lower')
        axes[i, 2].set_title(f'Pred (Conf: {confidence.mean():.3f})')
        axes[i, 2].axis('off')
        
        im = axes[i, 3].imshow(confidence, cmap='viridis', vmin=0, vmax=1, aspect='auto', origin='lower')
        axes[i, 3].set_title('Confidence')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    # Save summary plot
    plt.suptitle(f'Top {n_worst} Worst Performing Samples', fontsize=16)
    plt.tight_layout()
    summary_path = os.path.join(save_dir, f'worst_{n_worst}_samples_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save detailed metrics to text file
    metrics_path = os.path.join(save_dir, f'worst_{n_worst}_samples_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"WORST {n_worst} SAMPLES ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for data in summary_data:
            f.write(f"Rank {data['rank']} - Sample {data['sample_idx']}:\n")
            f.write(f"  Dice Score: {data['dice_score']:.4f}\n")
            f.write(f"  Precision: {data['precision']:.4f}\n")
            f.write(f"  Recall: {data['recall']:.4f}\n")
            f.write(f"  Specificity: {data['specificity']:.4f}\n")
            f.write(f"  Mean Confidence: {data['mean_confidence']:.4f}\n")
            f.write(f"  Std Confidence: {data['std_confidence']:.4f}\n")
            f.write(f"  TP: {data['tp']}, FP: {data['fp']}, FN: {data['fn']}, TN: {data['tn']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nWorst samples analysis complete!")
    print(f"Individual plots saved in: {save_dir}/")
    print(f"Summary plot: {summary_path}")
    print(f"Detailed metrics: {metrics_path}")
    
    # Print summary to console
    print(f"\nWorst {n_worst} samples summary:")
    print("-" * 70)
    print(f"{'Rank':<4} {'Sample':<8} {'Dice':<6} {'Precision':<9} {'Recall':<6} {'Confidence':<10}")
    print("-" * 70)
    for data in summary_data:
        print(f"{data['rank']:<4} {data['sample_idx']:<8} {data['dice_score']:<6.3f} "
                f"{data['precision']:<9.3f} {data['recall']:<6.3f} {data['mean_confidence']:<10.3f}")
    
    return worst_samples




# ---------------------------
# 5. Plot Worst Samples - Dice Scores (New)
# ---------------------------
def plot_worst_samples_dice_scores(model_path: str, dataset_path: str, n_worst: int = 100, 
                                   batch_size: int = 16, save_path: str = None):
    """
    Plot Dice scores for the worst N samples.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        n_worst: Number of worst samples to analyze
        batch_size: Batch size for data loading
        save_path: Path to save the plot (optional)
    """
    import os
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
    )
    
    eval_loader = test_loader if test_loader.dataset.__len__() > 0 else val_loader
    
    print(f"Evaluating all samples to find {n_worst} worst performers...")
    
    # Collect all samples with their Dice scores
    sample_data = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(eval_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Process each sample in the batch
            for i in range(images.shape[0]):
                # Calculate Dice coefficient for this sample
                sample_dice = dice_coeff(outputs[i:i+1], masks[i:i+1])
                
                # Store sample data
                sample_data.append({
                    'sample_idx': sample_idx,
                    'dice_score': sample_dice
                })
                
                sample_idx += 1
    
    # Sort by Dice score (ascending - worst first)
    sample_data.sort(key=lambda x: x['dice_score'])
    
    # Take the worst N samples
    worst_samples = sample_data[:n_worst]
    
    # Extract data for plotting
    ranks = list(range(1, n_worst + 1))  # Rank from 1 to n_worst
    dice_scores = [sample['dice_score'] for sample in worst_samples]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(ranks, dice_scores, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    plt.xlabel('Worst Sample Rank', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.title(f'Dice Scores for {n_worst} Worst Performing Samples', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    min_dice = min(dice_scores)
    max_dice = max(dice_scores)
    mean_dice = np.mean(dice_scores)
    
    stats_text = f'Min Dice: {min_dice:.4f}\nMax Dice: {max_dice:.4f}\nMean Dice: {mean_dice:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Highlight the very worst samples
    if n_worst >= 10:
        worst_10_ranks = ranks[:10]
        worst_10_scores = dice_scores[:10]
        plt.scatter(worst_10_ranks, worst_10_scores, color='red', s=50, zorder=5, 
                   label='Top 10 Worst', alpha=0.8)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print some statistics
    print(f"\nWorst {n_worst} samples statistics:")
    print(f"Dice score range: {min_dice:.4f} - {max_dice:.4f}")
    print(f"Mean Dice score: {mean_dice:.4f}")
    print(f"Standard deviation: {np.std(dice_scores):.4f}")
    
    return ranks, dice_scores

def comprehensive_model_analysis(model_path: str, dataset_path: str, batch_size: int = 16, save_dir: str = "model_analysis"):
    """
    Comprehensive model performance analysis with multiple visualizations.
    """
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model and data
    model = UNet(n_channels=1).to(device)  # Single channel model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    _, val_loader, test_loader = create_data_loaders(dataset_path=dataset_path, batch_size=batch_size)
    eval_loader = test_loader if test_loader.dataset.__len__() > 0 else val_loader
    
    print("Collecting comprehensive performance data...")
    
    # Collect detailed metrics for all samples
    all_metrics = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(eval_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Handle single channel input - take only one channel if multi-channel data
            if images.shape[1] > 1:
                images = images[:, -1:, :, :]  # Take last channel for single channel model
            
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            for i in range(images.shape[0]):
                # Calculate comprehensive metrics for each sample
                pred = predictions[i].cpu().numpy().flatten()
                target = masks[i].cpu().numpy().flatten()
                conf = probabilities[i].cpu().numpy().flatten()
                
                # Basic metrics
                tp = np.sum((pred == 1) & (target == 1))
                fp = np.sum((pred == 1) & (target == 0))
                fn = np.sum((pred == 0) & (target == 1))
                tn = np.sum((pred == 0) & (target == 0))
                
                dice = dice_coeff(outputs[i:i+1], masks[i:i+1])
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                
                # Confidence metrics
                mean_conf = conf.mean()
                std_conf = conf.std()
                conf_pos = conf[target == 1].mean() if np.sum(target == 1) > 0 else 0
                conf_neg = conf[target == 0].mean() if np.sum(target == 0) > 0 else 0
                
                # Area metrics
                target_area = np.sum(target)
                pred_area = np.sum(pred)
                area_ratio = pred_area / target_area if target_area > 0 else 0
                
                all_metrics.append({
                    'sample_idx': sample_idx,
                    'dice': dice, 'precision': precision, 'recall': recall,
                    'specificity': specificity, 'iou': iou,
                    'mean_conf': mean_conf, 'std_conf': std_conf,
                    'conf_pos': conf_pos, 'conf_neg': conf_neg,
                    'target_area': target_area, 'pred_area': pred_area,
                    'area_ratio': area_ratio,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                })
                
                sample_idx += 1
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_metrics)
    
    # Create comprehensive visualization dashboard
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Performance Distribution
    plt.subplot(4, 3, 1)
    plt.hist(df['dice'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dice Scores')
    plt.axvline(df['dice'].mean(), color='red', linestyle='--', label=f'Mean: {df["dice"].mean():.3f}')
    plt.legend()
    
    # 2. Metric Correlations
    plt.subplot(4, 3, 2)
    metrics_corr = df[['dice', 'precision', 'recall', 'specificity', 'iou']].corr()
    sns.heatmap(metrics_corr, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Metric Correlations')
    
    # 3. Performance vs Confidence
    plt.subplot(4, 3, 3)
    plt.scatter(df['mean_conf'], df['dice'], alpha=0.6, s=20)
    plt.xlabel('Mean Confidence')
    plt.ylabel('Dice Score')
    plt.title('Performance vs Confidence')
    # Add trend line
    z = np.polyfit(df['mean_conf'], df['dice'], 1)
    p = np.poly1d(z)
    plt.plot(df['mean_conf'].sort_values(), p(df['mean_conf'].sort_values()), "r--", alpha=0.8)
    
    # 4. Worst Samples Trend
    plt.subplot(4, 3, 4)
    worst_100 = df.nsmallest(100, 'dice')
    plt.plot(range(1, 101), worst_100['dice'].values, 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Worst Sample Rank')
    plt.ylabel('Dice Score')
    plt.title('Worst 100 Samples Performance')
    plt.grid(True, alpha=0.3)
    
    # 5. Precision vs Recall
    plt.subplot(4, 3, 5)
    plt.scatter(df['recall'], df['precision'], alpha=0.6, s=20, c=df['dice'], cmap='viridis')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall (colored by Dice)')
    cbar = plt.colorbar()
    cbar.set_label('Dice Score')
    
    # 6. Area Prediction Accuracy
    plt.subplot(4, 3, 6)
    plt.scatter(df['target_area'], df['pred_area'], alpha=0.6, s=20)
    plt.xlabel('True Area (pixels)')
    plt.ylabel('Predicted Area (pixels)')
    plt.title('Area Prediction Accuracy')
    # Perfect prediction line
    max_area = max(df['target_area'].max(), df['pred_area'].max())
    plt.plot([0, max_area], [0, max_area], 'r--', alpha=0.8, label='Perfect Prediction')
    plt.legend()
    
    # 7. Confidence Distribution by Class
    plt.subplot(4, 3, 7)
    bins = np.linspace(0, 1, 30)
    plt.hist(df['conf_pos'], bins=bins, alpha=0.7, label='Positive Class', color='red')
    plt.hist(df['conf_neg'], bins=bins, alpha=0.7, label='Negative Class', color='blue')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Class')
    plt.legend()
    
    # 8. Performance by Sample Index (time series)
    plt.subplot(4, 3, 8)
    window_size = 50
    rolling_dice = df['dice'].rolling(window=window_size, center=True).mean()
    plt.plot(df['sample_idx'], df['dice'], alpha=0.3, color='lightblue', label='Individual Samples')
    plt.plot(df['sample_idx'], rolling_dice, color='darkblue', linewidth=2, label=f'Rolling Mean (n={window_size})')
    plt.xlabel('Sample Index')
    plt.ylabel('Dice Score')
    plt.title('Performance Over Dataset')
    plt.legend()
    
    # 9. Error Analysis
    plt.subplot(4, 3, 9)
    error_types = ['False Positives', 'False Negatives', 'True Positives']
    error_means = [df['fp'].mean(), df['fn'].mean(), df['tp'].mean()]
    plt.bar(error_types, error_means, color=['red', 'orange', 'green'], alpha=0.7)
    plt.ylabel('Average Count per Sample')
    plt.title('Error Type Distribution')
    plt.xticks(rotation=45)
    
    # 10. Outlier Detection
    plt.subplot(4, 3, 10)
    Q1 = df['dice'].quantile(0.25)
    Q3 = df['dice'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['dice'] < Q1 - 1.5*IQR) | (df['dice'] > Q3 + 1.5*IQR)]
    
    plt.boxplot(df['dice'])
    plt.ylabel('Dice Score')
    plt.title(f'Outlier Detection\n({len(outliers)} outliers found)')
    
    # 11. Confidence vs Error Rate
    plt.subplot(4, 3, 11)
    df['error_rate'] = (df['fp'] + df['fn']) / (df['tp'] + df['fp'] + df['fn'] + df['tn'])
    plt.scatter(df['mean_conf'], df['error_rate'], alpha=0.6, s=20)
    plt.xlabel('Mean Confidence')
    plt.ylabel('Error Rate')
    plt.title('Confidence vs Error Rate')
    
    # 12. Model Calibration
    plt.subplot(4, 3, 12)
    # Binned confidence vs accuracy
    conf_bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_accuracies = []
    
    for i in range(len(conf_bins)-1):
        mask = (df['mean_conf'] >= conf_bins[i]) & (df['mean_conf'] < conf_bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
            bin_accuracies.append(df[mask]['dice'].mean())
    
    plt.plot(bin_centers, bin_accuracies, 'bo-', label='Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy (Dice)')
    plt.title('Model Calibration')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_analysis_single_channel.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary statistics
    summary_stats = {
        'Overall Performance': {
            'Mean Dice': df['dice'].mean(),
            'Std Dice': df['dice'].std(),
            'Min Dice': df['dice'].min(),
            'Max Dice': df['dice'].max(),
            'Median Dice': df['dice'].median(),
        },
        'Confidence Analysis': {
            'Mean Confidence': df['mean_conf'].mean(),
            'Confidence-Performance Correlation': df['mean_conf'].corr(df['dice']),
            'Positive Class Confidence': df['conf_pos'].mean(),
            'Negative Class Confidence': df['conf_neg'].mean(),
        },
        'Error Analysis': {
            'Mean False Positives': df['fp'].mean(),
            'Mean False Negatives': df['fn'].mean(),
            'Mean Error Rate': df['error_rate'].mean(),
            'Outliers': len(outliers),
        }
    }
    
    # Save summary
    with open(os.path.join(save_dir, 'summary_statistics_single_channel.txt'), 'w') as f:
        f.write("SINGLE CHANNEL MODEL ANALYSIS\n")
        f.write("=" * 40 + "\n\n")
        for category, stats in summary_stats.items():
            f.write(f"{category}:\n")
            f.write("-" * len(category) + "\n")
            for stat_name, value in stats.items():
                f.write(f"{stat_name}: {value:.4f}\n")
            f.write("\n")
    
    print(f"Single channel model analysis saved to {save_dir}/")
    return df, summary_stats


if __name__ == "__main__":
    dataset_file = f"/Users/pepijnlens/Documents/transformers/data/sea_clutter_segmentation_sequences.pt"
    model_file = f"models/unet_sequences.pt"

    intrepretation_file = f"interpretation_results_sequences.png"
    log_file = f"interpretation_results_sequences.log"
    
    train_model(dataset_file, num_epochs=50, batch_size=16, pretrained='/Users/pepijnlens/Documents/transformers/models/unet_sequences.pt', lr=1e-4, model_save_path=model_file)

    # Interpret the results
    print("\n" + "="*60)
    print("INTERPRETING MODEL RESULTS...")
    print("="*60)
    interpret_model_results(model_file, dataset_file, num_samples=5, save_plots=intrepretation_file, log_file=log_file)
    
    # Comprehensive analysis
    df, stats = comprehensive_model_analysis(model_file, dataset_file, save_dir="model_analysis_5frames")

    # # Plot Dice scores for worst 100 samples
    # plot_worst_samples_dice_scores(model_file, dataset_file, n_worst=100, 
    #                                save_path="worst_100_samples_dice_plot.png")
    

    # save_worst_samples(model_file, dataset_file, n_worst=100)

    # # Plot specific sample (sample 9)
    # print("\n" + "="*60)
    # print("ANALYZING SAMPLE 9...")
    # print("="*60)
    # for i in range(10):
    #     print(f"Analyzing sample {i}...")
    #     plot_specific_sample(model_file, dataset_file, sample_idx=i+233)

    # plot_specific_sample(model_file, dataset_file, sample_idx=987)

    # ---------------------------
    # 7. Save Worst Samples (New)
    # ---------------------------
