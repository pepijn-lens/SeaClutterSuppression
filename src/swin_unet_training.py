#!/usr/bin/env python3
"""
Swin U-Net Training Script for Sea Clutter Dataset
Trains a Swin Transformer U-Net on the data/data.pt multi-frame dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import logging
from datetime import datetime
import json
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.swin_unet import radar_swin_unet
from sklearn.metrics import precision_score, recall_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) if not torch.all((inputs >= 0) & (inputs <= 1)) else inputs
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice

class WeightedBCELoss(nn.Module):
    """Weighted BCE loss to handle class imbalance"""
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # Apply sigmoid if not already applied
        if not torch.all((inputs >= 0) & (inputs <= 1)):
            inputs = torch.sigmoid(inputs)
            
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy(inputs, targets, weight=None, reduction='none')
            # Apply positive class weighting
            pos_mask = targets > 0.5
            neg_mask = targets <= 0.5
            loss[pos_mask] = loss[pos_mask] * self.pos_weight
            return loss.mean()
        else:
            return F.binary_cross_entropy(inputs, targets)

def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate segmentation metrics"""
    y_pred_binary = (y_pred > threshold).float()
    y_true_binary = y_true.float()
    
    # Flatten for metric calculation
    y_pred_flat = y_pred_binary.view(-1).cpu().numpy()
    y_true_flat = y_true_binary.view(-1).cpu().numpy()
    
    # Calculate metrics
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Intersection over Union (IoU)
    intersection = (y_pred_binary * y_true_binary).sum().item()
    union = (y_pred_binary + y_true_binary).clamp(0, 1).sum().item()
    iou = intersection / (union + 1e-8)
    
    # Dice coefficient
    dice = (2 * intersection) / (y_pred_binary.sum().item() + y_true_binary.sum().item() + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

class MultiFrameDataset(torch.utils.data.Dataset):
    """Dataset for multi-frame sequences"""
    def __init__(self, sequences, masks, labels, use_frames='all'):
        """
        Args:
            sequences: [N, 3, H, W] - multi-frame sequences  
            masks: [N, 3, H, W] - corresponding masks
            labels: [N] - target count labels
            use_frames: 'all', 'single', or 'mean' - how to handle multiple frames
        """
        self.sequences = sequences
        self.masks = masks
        self.labels = labels
        self.use_frames = use_frames
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]  # [C, H, W] where C could be 1 or 3
        mask = self.masks[idx]          # [C, H, W] where C could be 1 or 3
        label = self.labels[idx]
        
        if self.use_frames == 'single':
            # Use first/only frame for single-frame data, middle frame for multi-frame
            if sequence.shape[0] == 1:
                # Already single frame
                sequence = sequence  # [1, H, W]
                mask = mask          # [1, H, W]
            else:
                # Multi-frame: use middle frame
                sequence = sequence[1:2]  # [1, H, W]
                mask = mask[1:2]          # [1, H, W]
        elif self.use_frames == 'mean':
            # Average across frames
            sequence = sequence.mean(dim=0, keepdim=True)  # [1, H, W]
            mask = mask.mean(dim=0, keepdim=True)          # [1, H, W]
        # else use_frames == 'all': keep all frames (works for both single and multi-frame)
            
        return sequence, mask, label

def load_dataset(data_path: str, use_frames: str = 'single') -> Tuple[MultiFrameDataset, MultiFrameDataset, MultiFrameDataset]:
    """Load and split the dataset"""
    logger.info(f"Loading dataset from {data_path}")
    data = torch.load(data_path)
    
    # Handle different dataset formats
    if 'sequences' in data:
        # Multi-frame dataset (data.pt)
        sequences = data['sequences']
        masks = data['mask_sequences']
        labels = data['labels']
        logger.info(f"Multi-frame dataset - Sequences shape: {sequences.shape}")
    elif 'images' in data:
        # Single-frame dataset (sea_clutter_single_frame.pt)
        images = data['images']
        masks = data['masks']
        labels = data['labels']
        
        # Add channel dimension and convert to "sequence" format for consistency
        sequences = images.unsqueeze(1)  # [N, 1, H, W] 
        masks = masks.unsqueeze(1)       # [N, 1, H, W]
        
        logger.info(f"Single-frame dataset - Images shape: {images.shape}")
        logger.info(f"Converted to sequences shape: {sequences.shape}")
    else:
        raise ValueError(f"Unknown dataset format. Expected 'sequences' or 'images' key in {data_path}")
    
    logger.info(f"Masks shape: {masks.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Using frames: {use_frames}")
    
    # Create dataset
    dataset = MultiFrameDataset(sequences, masks, labels, use_frames=use_frames)
    
    # Split dataset: 70% train, 15% val, 15% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': [], 'dice': []}
    
    for batch_idx, (sequences, masks, labels) in enumerate(dataloader):
        sequences, masks = sequences.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(outputs, masks)
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': [], 'dice': []}
    
    with torch.no_grad():
        for sequences, masks, labels in dataloader:
            sequences, masks = sequences.to(device), masks.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, masks)
            for key, value in metrics.items():
                all_metrics[key].append(value)
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def save_sample_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device, 
                           save_dir: str, num_samples: int = 8):
    """Save sample predictions for visualization"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (sequences, masks, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            sequences, masks = sequences.to(device), masks.to(device)
            outputs = model(sequences)
            
            # Convert to numpy for plotting
            if sequences.shape[1] == 1:
                # Single frame
                input_img = sequences[0, 0].cpu().numpy()
                true_mask = masks[0, 0].cpu().numpy()
            else:
                # Multi-frame - use middle frame
                input_img = sequences[0, 1].cpu().numpy()
                true_mask = masks[0, 1].cpu().numpy()
            
            pred_mask = outputs[0, 0].cpu().numpy()
            label = labels[0].item()
            
            # Create subplot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(input_img, cmap='viridis')
            axes[0].set_title(f'Input (Label: {label} targets)')
            axes[0].axis('off')
            
            axes[1].imshow(true_mask, cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title('True Mask')
            axes[1].axis('off')
            
            axes[2].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{i}_label_{label}.png'), dpi=150, bbox_inches='tight')
            plt.close()

def plot_training_history(history: Dict, save_path: str):
    """Plot training history"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # IoU
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Validation')
    axes[0, 1].set_title('IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    
    # Dice
    axes[0, 2].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[0, 2].plot(epochs, history['val_dice'], 'r-', label='Validation')
    axes[0, 2].set_title('Dice Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].legend()
    
    # Precision
    axes[1, 0].plot(epochs, history['train_precision'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(epochs, history['train_recall'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    # F1 Score
    axes[1, 2].plot(epochs, history['train_f1'], 'b-', label='Train')
    axes[1, 2].plot(epochs, history['val_f1'], 'r-', label='Validation')
    axes[1, 2].set_title('F1 Score')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_pos_weight(dataloader: DataLoader) -> float:
    """Calculate positive weight for BCE loss based on class distribution"""
    total_pixels = 0
    target_pixels = 0
    
    for _, masks, _ in dataloader:
        total_pixels += masks.numel()
        target_pixels += (masks > 0.5).sum().item()
    
    if target_pixels == 0:
        return 1.0
    
    # Weight = (total - positive) / positive
    pos_weight = (total_pixels - target_pixels) / target_pixels
    return pos_weight

def main():
    # Configuration
    config = {
        'data_path': 'data/sea_clutter_single_frame.pt',
        'use_frames': 'single',  # 'single', 'all', or 'mean'
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 5e-4,
        'warmup_epochs': 5,  # Add warmup epochs
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'num_workers': 0,
        'save_dir': 'results/swin_unet_training',
        'save_every': 10,
        'early_stopping_patience': 15,
        'use_weighted_bce': True,  # Enable weighted BCE
        'auto_pos_weight': True,   # Automatically calculate positive weight
        'patch_size': 4,  # Patch size for Swin Transformer
        'window_size': 8,  # Window size for Swin Transformer
        'embed_dim': 32,  # Embedding dimension for Swin Transformer
        'depths': [8],  # Depths for each Swin Transformer stage
        'num_heads': [2],  # Number of attention heads
        'mlp_ratio': 4.0,  # MLP ratio for Swin Transformer
        'drop_path_rate': 0.1,  # Drop path rate for Swin Transformer
        'dropout_rate': 0.1,  # Dropout rate for Swin Transformer
    }
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{config['save_dir']}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training Swin U-Net")
    logger.info(f"Config: {config}")
    logger.info(f"Save directory: {save_dir}")
    
    # Set device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(
        config['data_path'], 
        use_frames=config['use_frames']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Get input channels based on frame usage and dataset type
    if config['use_frames'] == 'all':
        # Check the actual number of frames in the dataset
        sample_seq, _, _ = train_dataset[0]
        in_chans = sample_seq.shape[0]  # Could be 1 for single-frame or 3 for multi-frame
    else:
        in_chans = 1  # Always 1 for 'single' or 'mean' modes
    
    # Create model
    model = radar_swin_unet(
        img_size=128,
        in_chans=in_chans,
        num_classes=1,
        patch_size=config['patch_size'],
        depths=config['depths'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
        mlp_ratio=config['mlp_ratio'],
        drop_path_rate=config['drop_path_rate'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss function and optimizer
    if config['use_weighted_bce']:
        if config['auto_pos_weight']:
            logger.info("Calculating positive weight from training data...")
            pos_weight = calculate_pos_weight(train_loader)
            logger.info(f"Calculated positive weight: {pos_weight:.4f}")
        else:
            pos_weight = 10.0  # Manual weight - adjust as needed
            
        criterion = WeightedBCELoss(pos_weight=pos_weight)
        logger.info(f"Using Weighted BCE Loss with pos_weight={pos_weight:.4f}")
    else:
        criterion = nn.BCELoss()
        logger.info("Using standard BCE Loss")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            # Linear warmup
            return epoch / config['warmup_epochs']
        else:
            # Cosine annealing after warmup
            progress = (epoch - config['warmup_epochs']) / (config['num_epochs'] - config['warmup_epochs'])
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': []
    }
    
    best_val_loss = float('inf')  # Track best validation loss
    patience_counter = 0
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for key in ['iou', 'dice', 'precision', 'recall', 'f1']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}, "
            f"Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'config': config
            }, os.path.join(save_dir, 'best_model.pt'))
            logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/3600:.2f} hours")
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    logger.info("Final Test Results:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    for key, value in test_metrics.items():
        logger.info(f"Test {key.capitalize()}: {value:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss,
        'training_time_hours': total_time/3600
    }
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Plot training history
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    
    # Save sample predictions
    save_sample_predictions(model, test_loader, device, 
                           os.path.join(save_dir, 'sample_predictions'))
    
    # Save final history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {save_dir}")

if __name__ == "__main__":
    main()
