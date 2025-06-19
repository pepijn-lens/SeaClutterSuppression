import torch
import time
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .swin_training import load_dataloaders, get_cosine_schedule_with_warmup
import models

def train_cnn(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, save_path="cnn_model.pt", patience=20, scheduler=None):
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

        for i, (inputs, masks, labels) in enumerate(train_loader):
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
            for inputs, masks, labels in val_loader:
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

    # Plot and save training/validation loss
    os.makedirs(f"training_loss/{save_path[:-3]}", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_loss/{save_path[:-3]}/training_loss.png")
    plt.close()
    print("📈 Saved CNN training loss plot to training_loss.png")

def evaluate_cnn(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, masks, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"CNN Test Accuracy: {acc:.2f}%")
    return acc

def analyze_cnn(model, dataloader, device, class_names=None, max_misclassified=10, save_path="cnn_model.png", accuracy=0):
    dir = f"confusion_matrices/{save_path}"
    os.makedirs(dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_labels = []
    misclassified = []

    with torch.no_grad():
        for inputs, masks, labels in dataloader:
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
    plt.title(f"CNN Confusion Matrix | Accuracy: {accuracy:.2f}%")
    plt.savefig(dir + "/confusion_matrix.png")
    plt.close()

    # Visualize misclassified images
    for i, (img, pred, label) in enumerate(misclassified):
        plt.imshow(img.squeeze(0))
        title = f"CNN Misclassified {i+1}: Pred={class_names[pred] if class_names else pred} | True={class_names[label] if class_names else label}"
        plt.title(title)
        plt.axis('off')
        plt.savefig(dir + f"/misclassified_{i+1}.png", dpi=500)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_name = "CNN-no_clutter-20dB.pt"
    dataset_pt = "data/20dB.pt"
    
    # Initialize CNN model with radar-optimized parameters
    model = models.RadarCNNMedium(in_channels=1, num_classes=6, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load('pretrained/CNN-no_clutter-20dB.pt', map_location=device))
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'CNN: {trainable_params:,} trainable parameters')
    
    # Load data
    train_loader, val_loader, test_loader = load_dataloaders(batch_size=32, root_dir=dataset_pt, random_seed=6)
    
    # Training setup optimized for radar data
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    total_steps = 100 * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.01
    )
    
    # Train the model
    train_cnn(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, save_path=model_name, scheduler=scheduler)
    
    # Evaluate the model
    accuracy = evaluate_cnn(model, test_loader, device)
    
    # Analyze results
    class_names = [f"{i} target{'s' if i != 1 else ''}" for i in range(6)]
    analyze_cnn(model, test_loader, device, class_names=class_names, max_misclassified=10, save_path=f"{model_name[:-3]}/{dataset_pt[4:-3]}", accuracy=accuracy)