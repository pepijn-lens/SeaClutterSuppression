import models

import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import time
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class RadarDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        loaded_data = torch.load(data_path, weights_only=False)
        
        # Check if it's a dictionary with keys or a TensorDataset
        if isinstance(loaded_data, dict):
            # Dictionary format with 'sequences'/'images', 'labels', 'metadata'
            if 'sequences' in loaded_data:
                # New sequence dataset
                self.data_tensor = loaded_data['sequences']
                self.is_sequence = True
            else:
                # Old single image dataset
                self.data_tensor = loaded_data['images']  
                self.is_sequence = False
                
            self.label_tensor = loaded_data['labels']
            self.metadata = loaded_data.get('metadata', {})
        else:
            # Assume it's a TensorDataset or similar structure
            # Extract tensors from the dataset
            if hasattr(loaded_data, 'tensors'):
                self.data_tensor = loaded_data.tensors[0]  # First tensor is data
                self.label_tensor = loaded_data.tensors[1]  # Second tensor is labels
            else:
                # If it's just tensors directly
                self.data_tensor = loaded_data[0]
                self.label_tensor = loaded_data[1]
            
            # Determine if it's sequence data based on tensor shape
            # Assuming sequences have shape (N, T, H, W) where T > 1
            if len(self.data_tensor.shape) == 4 and self.data_tensor.shape[1] > 1:
                self.is_sequence = True
            else:
                self.is_sequence = False
            
            self.metadata = {}

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        if self.is_sequence:
            # For sequences: shape is (n_frames, height, width)
            # Use time dimension as channel dimension: (n_frames, height, width)
            sequence = self.data_tensor[idx]
            sequence = sequence[:-1]  # Shape: (4, 128, 128)
            
            # Apply normalization to each frame (data is stored in dB scale)
            normalized_sequence = []
            for frame_idx in range(sequence.shape[0]):
                frame = sequence[frame_idx]
                # Normalize with frame's own statistics
                normalized_frame = (frame - frame.mean()) / (frame.std() + 1e-10)
                normalized_sequence.append(normalized_frame)
            
            sequence = torch.stack(normalized_sequence, dim=0)
            label = self.label_tensor[idx]
            return sequence, label
        else:
            # For single images: add channel dimension (1, height, width)
            image = self.data_tensor[idx]
            
            # Apply normalization (data is stored in dB scale)
            if len(image.shape) == 2:  # Single frame
                normalized_image = (image - image.mean()) / (image.std() + 1e-10)
                image = normalized_image.unsqueeze(0)  # Add channel dimension
            else:  # Already has channel dimension
                normalized_image = (image - image.mean()) / (image.std() + 1e-10)
                image = normalized_image
            
            label = self.label_tensor[idx]
            return image, label
    
    def get_metadata(self):
        """Return dataset metadata for reference."""
        return self.metadata
    
def load_dataloaders(root_dir="data/20dB", batch_size=32, num_workers=0,random_seed=42):
    dataset = RadarDataset(data_path=root_dir)

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

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_path="best_model.pt", patience=20, scheduler=None):
    best_val_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100. * correct / total
        train_loss /= total
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

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
        val_losses.append(val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} [{elapsed:.1f}s] "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
              f" | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'pretrained/{save_path}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
                print(f"🔁 Best model was from epoch {best_epoch} with Val Acc: {best_val_acc:.2f}%")
                break

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
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def analyze_model(model, dataloader, device, class_names=None, max_misclassified=10, save_path="model.png", accuracy=0):
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
    disp.plot()
    plt.title(f"Confusion Matrix | Accuracy: {accuracy:.2f}%")
    plt.savefig(dir + "/confusion_matrix.png")
    plt.close()

    # Visualize misclassified images (for 4,128,128 input: show as 2x2 grid of frames)
    for i, (img, pred, label) in enumerate(misclassified):
        if img.shape[0] == 3 and img.shape[1] == 128 and img.shape[2] == 128:
            fig, axs = plt.subplots(2, 2, figsize=(6, 6))
            for j in range(3):
                ax = axs[j // 2, j % 2]
                ax.imshow(img[j])
                ax.set_title(f"Frame {j+1}")
                ax.axis('off')
            fig.suptitle(f"Misclassified {i+1}: Pred={class_names[pred] if class_names else pred} | True={class_names[label] if class_names else label}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(dir + f"/misclassified_{i+1}.png", dpi=500)
            plt.close(fig)
        else:
            # fallback for other shapes (e.g. single image)
            plt.imshow(img.squeeze())
            title = f"Misclassified {i+1}: Pred={class_names[pred] if class_names else pred} | True={class_names[label] if class_names else label}"
            plt.title(title)
            plt.axis('off')
            plt.savefig(dir + f"/misclassified_{i+1}.png", dpi=500)
            plt.close()


def objective(trial, dataset_pth="data/30dB.pt"):
    layers = trial.suggest_categorical("layers", [6, 8])
    window_size = trial.suggest_categorical("window_size", [4, 8])
    patch_size = trial.suggest_categorical("patch_size", [4, 8])
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    hidden_dim = 2 * (patch_size ** 2) * 4 
    if hidden_dim == 128:
        heads = 8
    elif hidden_dim == 512:
        heads = 16
    head_dim = hidden_dim // heads

    model = models.radar_swin_t(
        in_channels=4,
        num_classes=6,
        hidden_dim=hidden_dim,
        window_size=window_size,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        patch_size=patch_size,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=weight_decay)

    train_loader, val_loader, _ = load_dataloaders(batch_size=32, root_dir=dataset_pth, random_seed=trial.number)
    total_steps = 100 * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.01
    )
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, save_path=f"optuna_trial_{trial.number}.pt", patience=8, scheduler=scheduler)

    model.load_state_dict(torch.load(f"pretrained/optuna_trial_{trial.number}.pt"))
    val_acc = evaluate(model, val_loader, device)
    return val_acc

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """
    Creates a schedule with a linear warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as a ratio of base LR at the end of cosine decay
    Returns:
        A LambdaLR scheduler
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # model_name = "8patch-medium.pt"
    model_name = "tiny_swin_no_clutter_20dB.pt"
    dataset_pt = "data/20dB.pt"
    
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial, dataset_pt), n_trials=20)

    # logger.info(f"Best trial: {study.best_trial.number}")
    # logger.info(f"Best value: {study.best_trial.value}")
    # logger.info(f"Best params: {study.best_trial.params}")

    model = models.radar_swin_t(
        num_classes=6,
        hidden_dim=128,  # 2 * (patch_size ** 2) * 4
        window_size=(2, 8),  # (2, 8) for 4 frames
        layers=6,
        heads=4,  # (3, 6, 12, 24) for 4 layers
        head_dim=32,  # hidden_dim // heads
        patch_size=8,  # 8 for 4 frames
    ).to(device)

    # print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params:,} trainable parameters')

    model.load_state_dict(torch.load(f'pretrained/{model_name}'))  # Load pre-trained model if available

    train_loader, val_loader, test_loader = load_dataloaders(batch_size=64, root_dir=dataset_pt, random_seed=6)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.002)

    # total_steps = 100 * len(train_loader)
    # warmup_steps = int(0.1 * total_steps)  # 10% warmup

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.01
    # )

    # train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, save_path=model_name, scheduler=scheduler)

    # dataset = RadarDataset(data_path=dataset_pt)
    # test_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    accuracy = evaluate(model, test_loader, device)

    class_names = [f"{i} target{'s' if i != 1 else ''}" for i in range(6)]  # or set your own
    analyze_model(model, test_loader, device, max_misclassified=10, save_path=f"{model_name[:-3]}/{dataset_pt[4:-3]}", accuracy=accuracy)

