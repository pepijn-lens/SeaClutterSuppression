import torch
import torch.nn as nn
from typing import Tuple
import marimo as mo 

import os

from sea_clutter import create_data_loaders  
from sklearn.metrics import precision_score, recall_score

import models

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, smooth=1.):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.1, tversky_weight=0.9, alpha=0.2, beta=0.8):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
    
    def forward(self, pred, target):
        # BCE with logits expects raw logits (no sigmoid)
        bce = self.bce_loss(pred, target)
        # Tversky loss applies sigmoid internally
        tversky = self.tversky_loss(pred, target)
        return self.bce_weight * bce + self.tversky_weight * tversky

def dice_coeff(pred, target, smooth=1.):
    pred = torch.sigmoid(pred).detach().cpu().numpy() > 0.5
    target = target.cpu().numpy()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate(model, loader, device, criterion):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, masks).item()

    n = len(loader)
    return loss_total / n

# ---------------------------
# 3. Training Loop
# ---------------------------
def train_model(dataset_path: str, n_channels=3, num_epochs=30, patience = 10, batch_size=16, lr=1e-4, pretrained=None, model_save_path='unet_single_frame.pt', bce_weight=0.1, tversky_weight=0.9, tversky_alpha= 0.2, tversky_beta=0.8, base_filters=16):
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
    criterion = CombinedLoss(bce_weight=bce_weight, tversky_weight=tversky_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Using combined loss: BCE weight={bce_weight}, Tversky weight={tversky_weight}")

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
        val_loss = evaluate(model, val_loader, device, criterion)
        avg_epoch_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Step the scheduler
        scheduler.step()

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Best model saved to {model_save_path} with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a U-Net segmentation model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the training dataset")
    parser.add_argument("--n-channels", type=int, default=3,
                        help="Number of input channels of the model")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model weights (optional)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--model-save-path", type=str, default="pretrained/tversky.pt",
                        help="Where to save the trained model")
    parser.add_argument("--bce-weight", type=float, default=0.1,
                        help="Weight for BCE loss in combined loss")
    parser.add_argument("--tversky-weight", type=float, default=0.9,
                        help="Weight for Tversky loss in combined loss")
    parser.add_argument("--base-filters", type=int, default=64,
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
        tversky_weight=args.tversky_weight,
        base_filters=args.base_filters
    )
