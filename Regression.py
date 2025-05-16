from swin_regress import swin_t, swin_s  # or replace with direct class if in same file

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import time
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class RadarDataset(Dataset):
    def __init__(self, root_dir, normalize_targets=True):
        """
        Args:
            root_dir (str): Directory containing burst_XXXXX.parquet and burst_XXXXX.json files.
            normalize_targets (bool): Whether to normalize range and velocity.
        """
        self.root_dir = root_dir
        self.normalize_targets = normalize_targets

        self.samples = sorted([
            fname.replace(".parquet", "")
            for fname in os.listdir(root_dir)
            if fname.endswith(".parquet")
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        parquet_path = os.path.join(self.root_dir, f"{sample_id}.parquet")
        json_path = os.path.join(self.root_dir, f"{sample_id}.json")

        # Load burst data
        df = pd.read_parquet(parquet_path)
        burst_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 64, 512)

        # Load label data
        with open(json_path, "r") as f:
            label_data = json.load(f)
            ranges = label_data["ranges"]
            velocities = label_data["velocities"]

        range = ranges[-1]
        velocity = velocities[-1]

        # Normalize labels (optional)
        if self.normalize_targets:
            range /= 707.106         # Normalize range [0, 707.106] -> [0, 1]
            velocity = (velocity + 27.5) / 55  # Normalize velocity [-27.5, 27.5] -> [0, 1]

        target = torch.tensor([range, velocity], dtype=torch.float32)  # Shape: (2,)

        return burst_tensor, target
    
def load_dataloaders(root_dir="data/bursts_regression", batch_size=16, num_workers=0,random_seed=42):
    dataset = RadarDataset(root_dir=root_dir)

    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(random_seed)  # Set seed for reproducibility
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_path="best_model.pt"):
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        train_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1}/{num_epochs} [{elapsed:.1f}s] "
              f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model with Val Loss: {val_loss:.4f} → {save_path}")


def evaluate(model, test_loader, device, denormalize=True):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            if denormalize:
                preds = outputs.clone()
                preds[:, 0] *= 707.106          # Denormalize range
                preds[:, 1] = preds[:, 1] * 55.0 - 27.5  # Denormalize velocity

                labels[:, 0] *= 707.106
                labels[:, 1] = labels[:, 1] * 55.0 - 27.5
            else:
                preds = outputs

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)}")
                print(f"  Predicted (range, velocity): {preds[:3].cpu().numpy()}")
                print(f"  Ground Truth:               {labels[:3].cpu().numpy()}")

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"\n✅ Test MSE Loss: {avg_loss:.4f}")
    torch.save({"preds": torch.cat(all_preds), "labels": torch.cat(all_labels)}, "eval_outputs.pt")
    return torch.cat(all_preds), torch.cat(all_labels)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = swin_t(
        channels=1,
        num_classes=2,
        window_size=(2, 8),
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        head_dim=32,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    ).to(device)

    # print(model)

    train_loader, val_loader, test_loader = load_dataloaders(batch_size=32, root_dir="data/bursts_regression/num_targets_1", random_seed=42)

    # criterion = torch.nn.SmoothL1Loss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15, save_path="new_best_model.pt")

    model.load_state_dict(torch.load("new_best_model.pt"))

    evaluate(model, test_loader, device)

