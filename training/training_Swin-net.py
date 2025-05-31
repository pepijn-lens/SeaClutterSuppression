import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score


from load_segmentation_data import create_data_loaders
from Swin_Net import SwinUNET, CombinedLoss

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

def train_swin_unet(dataset_path: str, num_epochs=50, batch_size=8, lr=1e-4, model_save_path='Swin_Net.pt'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
    )
    
    # Get a sample to determine input size
    sample_batch = next(iter(train_loader))
    sample_image = sample_batch[0][0]  # First image in batch
    img_height, img_width = sample_image.shape[-2:]
    in_channels = sample_image.shape[0]
    
    print(f"Input image size: {img_height}x{img_width}")
    print(f"Input channels: {in_channels}")
    
    # Create model adapted for radar data
    model = SwinUNET(
        img_size=max(img_height, img_width),  # Use the larger dimension
        patch_size=8,  # Fixed: Use 4 for 128x128 images
        in_chans=in_channels,  # Use actual number of input channels
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],  # Deeper middle layers for complex patterns
        depths_decoder=[2, 2, 2, 1],
        num_heads=[3, 6, 12, 24],
        window_size=4,  # Increased window size
        drop_path_rate=0.2,
        final_upsample="expand_first"
    ).to(device)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Swin-UNET: {trainable_params:,} trainable parameters')

    # Use standard BCE loss to avoid size mismatch issues
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Early stopping variables
    best_dice = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Debug: Print shapes on first iteration
            if epoch == 0 and i == 0:
                print(f"Input images shape: {images.shape}")
                print(f"Target masks shape: {masks.shape}")

            optimizer.zero_grad()
            outputs = model(images)
            
            # Debug: Print output shape on first iteration
            if epoch == 0 and i == 0:
                print(f"Model output shape: {outputs.shape}")
            
            # Fix: Ensure output matches target size
            if outputs.shape != masks.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, 
                    size=masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                if epoch == 0 and i == 0:
                    print(f"Resized output shape: {outputs.shape}")
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        # Validation phase
        dice, prec, recall = evaluate(model, val_loader, device)
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / num_batches
        
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

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Best model saved to {model_save_path} with Dice score: {best_dice:.3f}")
    return model

if __name__ == "__main__":
    dataset_file = f"data/sea_clutter_segmentation_5_state.pt"
    model_file = f"Swin_Net.pt"
    
    # Train the model
    print("=" * 60)
    print("TRAINING SWIN-UNET ON SEA CLUTTER DATA")
    print("=" * 60)
    
    model = train_swin_unet(
        dataset_path=dataset_file,
        num_epochs=50,
        batch_size=32,  # Smaller batch size due to model complexity
        lr=1e-4,
        model_save_path=model_file
    )