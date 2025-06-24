import torch
import time
import torch.nn as nn
# ---------------------------
# 1. U-Net Architecture
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_filters=16):
        super().__init__()
        self.enc1 = DoubleConv(n_channels, base_filters)
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)

        self.out_conv = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out_conv(x)
    
if __name__ == "__main__":
    # Load dataset
    data = torch.load("local_data/12SNR_clutter.pt")
    # Extract input and target
    sequences = data['sequences'][:30]  # shape: (batch, 3, H, W)
    masks = data['masks'][:30]          # shape: (batch, num_frames, H, W)
    # Use the last mask as target
    targets = masks[:, -1, :, :].unsqueeze(1)  # shape: (batch, 1, H, W)

    device = torch.device("mps" if torch.backends.mps() else "cpu")
    model = UNet(n_channels=3, n_classes=1, base_filters=64)
    model.load_state_dict(torch.load("pretrained/12SNR_clutter_64.pt", map_location=device))
    model.to(device)
    model.eval()

    sequences = sequences.to(device)

    # Warm-up
    with torch.no_grad():
        _ = model(sequences)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        _ = model(sequences)
    end_time = time.time()

    print(f"Inference time for batch of 30: {(end_time - start_time) * 1000:.2f} ms")
