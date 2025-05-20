import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import optuna
from datetime import datetime
import logging

from swin_class import swin_t
from radar import create_dataset
from sklearn.model_selection import train_test_split

# -------------------------
# Logging Setup
# -------------------------
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/training_log_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log(msg):
    logging.info(msg)

# -------------------------
# Dataset
# -------------------------
class RadarBurstDataset(Dataset):
    def __init__(self, pt_file_path):
        data, labels = torch.load(pt_file_path)
        self.data = torch.stack(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -------------------------
# Model Function
# -------------------------
def model_fn(hidden_dim, patch_size):
    return swin_t(
        channels=1,
        num_classes=6,
        window_size=(2, 8),
        hidden_dim=hidden_dim,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        head_dim=32,
        downscaling_factors=(patch_size, 2, 2, 2),
        relative_pos_embedding=True
    )

# -------------------------
# Data Split Function
# -------------------------
def train_val_test_split(dataset, val_ratio=0.15, test_ratio=0.15, seed=42):
    indices = list(range(len(dataset)))

    # Shuffle and split: train vs temp (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        shuffle=True
    )

    # Calculate proportion of val/test within temp
    val_prop = val_ratio / (val_ratio + test_ratio)

    # Split temp into val and test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_prop,
        random_state=seed,
        shuffle=True
    )

    # Return Subsets
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx)
    )

# -------------------------
# Train Function
# -------------------------
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=12):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / 10
                log(f"Epoch {epoch+1}, Batch {batch_idx+1} - Training Loss: {avg_loss:.4f}")
                running_loss = 0.0

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100. * correct / total
        best_val_acc = max(best_val_acc, val_acc)
        log(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {val_acc:.2f}%")

    return best_val_acc

# -------------------------
# Evaluation Function
# -------------------------
def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total

# -------------------------
# Optuna Objective
# -------------------------
def optuna_objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [96])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32])
    patch_size = trial.suggest_categorical("patch_size", [4])
    num_epochs = 10  # Increase later

    dataset = RadarBurstDataset(pt_file_path="data/preprocessed_dataset.pt")
    train_set, val_set, _ = train_val_test_split(dataset, val_ratio=0.15, test_ratio=0.15)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("mps")
    print(f"Using device: {device}")

    log(f"🧪 Trial with params: hidden_dim={hidden_dim}, lr={lr:.5f}, weight_decay={weight_decay:.6f}, batch_size={batch_size}, downscaling_factors={patch_size}")

    model = model_fn(hidden_dim=hidden_dim, patch_size=patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    val_acc = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    log(f"🎯 Trial Result - Validation Accuracy: {val_acc:.2f}%")
    return val_acc

# -------------------------
# Main Entry
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    create_dataset(6, 2000)

    # Prepare dataset split for final test set usage
    full_dataset = RadarBurstDataset(pt_file_path="data/preprocessed_dataset.pt")
    train_set, val_set, test_set = train_val_test_split(full_dataset, val_ratio=0.15, test_ratio=0.15)

    # Fix train/val/test splits across all trials
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=5)  # Increase as needed

    log("✅ Optuna Search Completed")
    log(f"Best Parameters: {study.best_params}")
    log(f"Best Validation Accuracy: {study.best_value:.2f}%")

    # Save Optuna results
    with open(f"logs/optuna_best_params_{timestamp}.txt", "w") as f:
        f.write("Best Hyperparameters:\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest Validation Accuracy: {study.best_value:.2f}\n")

    # Final training on full train + val, test on test set
    log("🚀 Final training on train + val, evaluating on test set...")
    final_train_set = torch.utils.data.ConcatDataset([train_set, val_set])
    train_loader = DataLoader(final_train_set, batch_size=study.best_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=study.best_params["batch_size"], shuffle=False)

    device = torch.device("mps")
    model = model_fn(
        hidden_dim=study.best_params["hidden_dim"],
        patch_size=study.best_params["patch_size"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.best_params["lr"], weight_decay=study.best_params["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, None, criterion, optimizer, device, num_epochs=12)
    test_acc = evaluate(model, test_loader, device)

    log(f"🏁 Final Test Accuracy: {test_acc:.2f}%")

    # Save final model
    model_path = f"models/final_model_{timestamp}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    log(f"💾 Final model saved to {model_path}")
