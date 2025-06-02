import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
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
def train_model(dataset_path: str, num_epochs=30, batch_size=16, lr=1e-4, pretrained=None, model_save_path='unet_single_frame.pt'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        mask_strategy='last',
    )

    model = UNet(n_channels=1).to(device)
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
# 5. Plot Worst Samples - Dice Scores 
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
    model = UNet(n_channels=1).to(device)
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
    plt.ylim(0, 1900)  # Set y-axis limit to 0-1800
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
    plt.ylim(0.09, 0.83)  # Set y-axis limit to 0-0.8
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
    dataset_file = f"/Users/pepijnlens/Documents/transformers/data/sea_clutter_single_frame.pt"
    model_file = f"models/unet_single_frame.pt"

    intrepretation_file = f"interpretation_results_single_frame.png"
    log_file = f"interpretation_results_single_frame.log"
    
    train_model(dataset_file, num_epochs=50, batch_size=16, pretrained='/Users/pepijnlens/Documents/transformers/models/unet_sequences.pt', lr=1e-4, model_save_path=model_file)
    
    # Comprehensive analysis
    df, stats = comprehensive_model_analysis(model_file, dataset_file, save_dir="model_analysis_roll")

    # Plot Dice scores for worst 100 samples
    plot_worst_samples_dice_scores(model_file, dataset_file, n_worst=100, 
                                   save_path="worst_100_samples_dice_plot.png")
    
