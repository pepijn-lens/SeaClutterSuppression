from swin_class import swin_t, WindowAttention  # or replace with direct class if in same file

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset
import time
import matplotlib.pyplot as plt
from radar import PulsedRadar, create_dataset
from helper import plot_doppler

from collections import Counter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class RadarBurstDataset(Dataset):
    def __init__(self, pt_file_path):
        # Load data and labels from the .pt file
        data, labels = torch.load(pt_file_path)
        
        # Convert to tensors if not already
        self.data = torch.stack(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def load_dataloaders(root_dir="swin_bursts", batch_size=16, num_workers=0,random_seed=42):
    dataset = RadarBurstDataset(pt_file_path=root_dir)

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
    best_val_acc = 0.0

    total_loss = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        i = 0
        for inputs, labels in train_loader:
            i += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = 100. * correct / total
        train_loss /= total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        total_loss.append(train_loss)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100. * correct / total
        val_loss /= total

        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} [{elapsed:.1f}s] "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model with Val Acc: {val_acc:.2f}% → {save_path}")

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        i = 0
        for inputs, labels in test_loader:
            i += 1
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)} | Predicted: {predicted.cpu().numpy()}, Labels: {labels.cpu().numpy()}")

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def analyze_model(model, dataloader, device, class_names=None, max_misclassified=10, save_path="model.png"):
    dir = f"confusion_matrices/{save_path}"
    os.makedirs(dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_labels = []
    misclassified = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(inputs.size(0)):
                if preds[i] != labels[i] and len(misclassified) < max_misclassified:
                    misclassified.append((inputs[i].cpu(), preds[i].item(), labels[i].item()))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(dir + "/confusion_matrix.png")
    plt.close()

    # Visualize misclassified images
    for i, (img, pred, label) in enumerate(misclassified):
        plt.imshow(img.squeeze(0), cmap="gray")  # squeeze channel dimension
        title = f"Misclassified {i+1}: Pred={class_names[pred] if class_names else pred} | True={class_names[label] if class_names else label}"
        plt.title(title)
        plt.axis('off')
        plt.savefig(dir + f"/misclassified_{i+1}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    create_dataset(6, 2000, save_path="data/40dB_RCS_12000samples.pt")

    model = swin_t(
        channels=1,
        num_classes=6,
        window_size=(2, 8),
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        head_dim=32,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    ).to(device)

    model.load_state_dict(torch.load("models/40dB_RCS.pt"))
    train_loader, val_loader, test_loader = load_dataloaders(batch_size=32, root_dir="data/40dB_5000samples.pt", random_seed=42)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, save_path="models/40dB.pt")

    model.load_state_dict(torch.load("models/40dB_RCS.pt"))

    evaluate(model, test_loader, device)

    class_names = [f"{i} target{'s' if i != 1 else ''}" for i in range(6)]  # or set your own
    analyze_model(model, test_loader, device, class_names=class_names, max_misclassified=10, save_path="40dB_RCS.png")
