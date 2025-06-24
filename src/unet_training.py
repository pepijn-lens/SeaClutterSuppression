import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import marimo as mo 

import os
import seaborn as sns

from sea_clutter import create_data_loaders  
from sklearn.metrics import precision_score, recall_score

import models

# ---------------------------
# 2. Metrics
# ---------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        # BCE with logits expects raw logits (no sigmoid)
        bce = self.bce_loss(pred, target)
        # Dice loss applies sigmoid internally
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

def dice_coeff(pred, target, smooth=1.):
    pred = torch.sigmoid(pred).detach().cpu().numpy() > 0.5
    target = target.cpu().numpy()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    dice_total, prec_total, recall_total = 0, 0, 0
    with torch.no_grad():
        for images, masks, labels in loader:
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
def train_model(dataset_path: str, n_channels=3, num_epochs=30, patience = 10, batch_size=16, lr=1e-4, pretrained=None, model_save_path='unet_single_frame.pt', bce_weight=1.0, dice_weight=1.0, base_filters=16):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
    )

    model = models.UNet(n_channels=n_channels, base_filters=base_filters).to(device)  # Now uses both parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params:,} trainable parameters')
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=device))
        print(f"Loaded pretrained model from {pretrained}")
    criterion = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping variables
    best_dice = 0.0
    patience_counter = 0

    print(f"Using combined loss: BCE weight={bce_weight}, Dice weight={dice_weight}")

    for epoch in mo.status.progress_bar(range(num_epochs)):
        model.train()
        epoch_loss = 0

        for i, (images, masks, labels) in enumerate(train_loader):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a U-Net segmentation model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the training dataset")
    parser.add_argument("--n-channels", type=int, default=3,
                        help="Number of input channels of the model")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model weights (optional)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--model-save-path", type=str, default="pretrained/test_model.pt",
                        help="Where to save the trained model")
    parser.add_argument("--bce-weight", type=float, default=1.0,
                        help="Weight for BCE loss in combined loss")
    parser.add_argument("--dice-weight", type=float, default=1.0,
                        help="Weight for Dice loss in combined loss")
    parser.add_argument("--base-filters", type=int, default=16,
                        help="Number of base filters for U-Net")

    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset_path,
        n_channels=args.n_channels,
        num_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
        model_save_path=args.model_save_path,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        base_filters=args.base_filters
    )
