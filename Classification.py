from new_swin import radar_swin_t
import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import time
import matplotlib.pyplot as plt
import optuna
import logging
import math
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class RadarDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = torch.load(data_path, weights_only=False)
        
        # Access the dictionary structure we created
        self.data_tensor = self.dataset['images']
        self.label_tensor = self.dataset['labels']
        self.metadata = self.dataset['metadata']

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        # Add channel dimension for CNN compatibility (batch_size, 1, height, width)
        image = self.data_tensor[idx].unsqueeze(0)  # Add channel dimension
        label = self.label_tensor[idx]
        return image, label
    
    def get_metadata(self):
        """Return dataset metadata for reference."""
        return self.metadata
    
    
def load_dataloaders(root_dir="swin_bursts", batch_size=16, num_workers=0,random_seed=42):
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
            torch.save(model.state_dict(), f'models/{save_path}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
                print(f"🔁 Best model was from epoch {best_epoch} with Val Acc: {best_val_acc:.2f}%")
                break

    # Plot and save training/validation loss
    os.makedirs(f"training_loss/{save_path[:-3]}", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_loss/{save_path[:-3]}/training_loss.png")
    plt.close()
    print("📈 Saved training loss plot to training_loss.png")


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

def analyze_model(model, dataloader, device, class_names=None, max_misclassified=10, save_path="model.png", accruacy=0):
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
    plt.title(f"Confusion Matrix | Accuracy: {accruacy:.2f}%")
    plt.savefig(dir + "/confusion_matrix.png")
    plt.close()

    # Visualize misclassified images
    for i, (img, pred, label) in enumerate(misclassified):
        plt.imshow(img.squeeze(0))  # squeeze channel dimension
        title = f"Misclassified {i+1}: Pred={class_names[pred] if class_names else pred} | True={class_names[label] if class_names else label}"
        plt.title(title)
        plt.axis('off')
        plt.savefig(dir + f"/misclassified_{i+1}.png", dpi=500)

# Set up logging
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("optuna_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optuna_search")

def objective(trial, dataset_pth="data/30dB.pt"):
    layers = trial.suggest_categorical("layers", [2, 4, 6, 8])
    window_size = trial.suggest_categorical("window_size", [2, 4, 8])
    patch_size = trial.suggest_categorical("patch_size", [4, 8])
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    hidden_dim = 2 * (patch_size ** 2)
    if hidden_dim == 32:
        heads = 2
    elif hidden_dim == 128:
        heads = 4
    head_dim = hidden_dim // heads

    logger.info(f"Trial {trial.number}: layers={layers}, window_size={window_size}, patch_size={patch_size}, heads={heads}, head_dim={head_dim}, weight decay={weight_decay}, dataset={dataset_pth}")

    model = radar_swin_t(
        in_channels=1,
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
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, save_path=f"optuna_trial_{trial.number}.pt", patience=15, scheduler=scheduler)

    model.load_state_dict(torch.load(f"models/optuna_trial_{trial.number}.pt"))
    val_acc = evaluate(model, val_loader, device)
    logger.info(f"Trial {trial.number} finished with val_acc={val_acc:.4f}")
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
    model_name = "sea_monster-25dB.pt"
    dataset_pt = "data/sea_clutter_classification_SCR25.pt"
    
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial, dataset_pt), n_trials=20)

    # logger.info(f"Best trial: {study.best_trial.number}")
    # logger.info(f"Best value: {study.best_trial.value}")
    # logger.info(f"Best params: {study.best_trial.params}")
    
    model = radar_swin_t(
        in_channels=1,
        num_classes=6,
        hidden_dim=32,
        window_size=8,
        layers=8,
        heads=2,
        head_dim=16,
        patch_size=4
    ).to(device)

    # print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params:,} trainable parameters')

    model.load_state_dict(torch.load(f'models/{model_name}'))  # Load pre-trained model if available

    train_loader, val_loader, test_loader = load_dataloaders(batch_size=32, root_dir=dataset_pt, random_seed=8)

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
    analyze_model(model, test_loader, device, max_misclassified=10, save_path=f"{model_name[:-3]}/{dataset_pt[4:-3]}", accruacy=accuracy)

