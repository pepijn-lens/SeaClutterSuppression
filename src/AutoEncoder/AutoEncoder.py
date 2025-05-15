import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm

from radar import generate_bursts

# ====== Radar Dataset Class ======
class RadarDataset(Dataset):
    def __init__(self, burst_tuples, root_dir='bursts'):
        """
        burst_tuples: list of (num_targets, burst_id)
        """
        self.burst_tuples = burst_tuples
        self.root_dir = root_dir

    def __len__(self):
        return len(self.burst_tuples)

    def __getitem__(self, idx):
        num_targets, burst_id = self.burst_tuples[idx]
        dir_path = f"{self.root_dir}/num_targets_{num_targets}"
        file_path = f"{dir_path}/burst_{burst_id}.parquet"

        data = pd.read_parquet(file_path)
        magnitude = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)  # (1, 64, 512)
        return magnitude


# ==========================
# 3. Radar Conv Autoencoder Model
# ==========================
class RadarConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(RadarConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(512 * 1 * 64, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 1 * 64)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_encode(x)
        return z

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 1, 64)
        x = self.decoder(x)
        
        x = (x + 1) * 10 - 10  # Scale tanh output to [-10, 10]
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


# ==========================
# 4. Autoencoder Loss
# ==========================
def autoencoder_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')

# ==========================
# 5. Training and Latent Saving
# ==========================
def train_and_save(
    latent_dim=128,
    batch_size=16,
    num_epochs=20,
    lr=1e-4,
    root_dir='bursts',
    num_targets=1,
    num_bursts=4000,
    save_path='autoencoder_final.pt'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    os.makedirs('models', exist_ok=True)

    # Load all burst tuples (1–5 targets, 100 bursts each)
    all_bursts = [(num_targets, burst_id) for num_targets in range(1,num_targets + 1) for burst_id in range(num_bursts)]

    # Create dataset and dataloader
    dataset = RadarDataset(burst_tuples=all_bursts, root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model and optimizer
    model = RadarConvAutoencoder(latent_dim=latent_dim).to(device)
    # model.load_state_dict(torch.load('models/autoencoder_final.pt'))
    optimizer = optim.Adam(model.parameters(), lr=lr)\
    
    all_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            recon = model(batch)
            loss = autoencoder_loss(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.6f}")
        all_losses.append(avg_loss)

        # Save model every epoch
        torch.save(model.state_dict(), f"models/autoencoder_epoch_{epoch+1}.pt")


    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save loss history
    loss_df = pd.DataFrame(all_losses, columns=['Loss'])
    plt.figure(figsize=(10, 5))
    plt.plot(loss_df['Loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(f"{os.path.dirname(save_path)}/loss_history.png")
    plt.close()

def evaluate_on_validation(
    model_path,
    latent_dim=128,
    batch_size=16,
    root_dir='validation_bursts',
    num_targets=5,
    num_bursts=500,
    device=None
):
    """
    Evaluates the trained autoencoder model on a validation set and returns the average MSE loss.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Prepare validation burst tuples
    val_bursts = [(num_targets, burst_id) for burst_id in range(num_bursts)]

    # Dataset and loader
    val_dataset = RadarDataset(burst_tuples=val_bursts, root_dir=root_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load model
    model = RadarConvAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = batch.to(device)
            recon = model(batch)
            loss = autoencoder_loss(recon, batch)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Validation MSE Loss: {avg_loss:.6f}")

    return avg_loss


def visualize_reconstruction(model, burst_tuple, root_dir='bursts', device=None):
    """
    Visualizes input vs reconstructed output for a single burst.
    
    Parameters:
    - model: trained RadarConvAutoencoder
    - burst_tuple: (num_targets, burst_id)
    - root_dir: base directory where bursts are stored
    - device: 'cuda' or 'cpu'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model.eval()
    
    # Load burst
    num_targets, burst_id = burst_tuple
    file_path = f"{root_dir}/num_targets_{num_targets}/burst_{burst_id}.parquet"
    data = pd.read_parquet(file_path)
    
    # Prepare tensor
    input_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 64, 4096)
    
    with torch.no_grad():
        recon_tensor = model(input_tensor)
    
    # Convert to numpy
    input_np = input_tensor.squeeze().cpu().numpy()     # (64, 4096)
    recon_np = recon_tensor.squeeze().cpu().numpy()     # (64, 4096)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_np, aspect='auto', cmap='viridis')
    plt.title('Original Magnitude')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(recon_np, aspect='auto', cmap='viridis')
    plt.title('Reconstructed Magnitude')
    plt.colorbar()

    plt.tight_layout()
    os.makedirs('reconstructions', exist_ok=True)
    plt.savefig(f"reconstructions/burst_{num_targets}_{burst_id}.png")
    plt.close()



if __name__ == "__main__":
    # Set device internally
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_targets = 5
    num_bursts = 2000

    for i in range(1, num_targets + 1):
        # Generate bursts
        generate_bursts('cpu', n_bursts=num_bursts, num_targets=i)


    # for i in range(1, num_targets + 1):
    #     # Generate bursts
    #     generate_bursts('cpu', n_bursts=num_bursts, num_targets=i)

    # train_and_save(
    #     latent_dim=128,
    #     batch_size=16,
    #     num_epochs=30,
    #     lr=2e-4,
    #     root_dir='bursts',
    #     num_targets=num_targets,
    #     num_bursts=num_bursts,
    #     save_path='models/autoencoder_final.pt'
    # )

    # num_bursts_reconstruction = 10
    # for i in range(1, num_targets + 1):
    #     # Generate bursts
    #     generate_bursts('cpu', n_bursts=200, num_targets=i, dir="validation_bursts")

    # val_loss = evaluate_on_validation(
    #     model_path="models/autoencoder_final.pt",
    #     latent_dim=128,
    #     batch_size=16,
    #     root_dir='validation_bursts',
    #     num_targets=4 ,       # evaluate on specific target count
    #     num_bursts=200       # e.g., use 500 bursts as validation set
    # )

    # num_bursts_reconstruction = 10
    # for i in range(1, num_targets + 1):
    #     # Generate bursts
    #     generate_bursts('cpu', n_bursts=num_bursts_reconstruction, num_targets=i, dir="reconstruction_bursts")

    # # Load model
    # model = RadarConvAutoencoder(latent_dim=128)
    # model.load_state_dict(torch.load('models/autoencoder_final.pt'))
    # model.to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # for target in range(1, num_targets + 1):
    #     for burst in range(num_bursts_reconstruction):
    #         # Visualize one example (e.g., 3 targets, burst 42)
    #         visualize_reconstruction(model, (target, burst), root_dir="reconstruction_bursts")
