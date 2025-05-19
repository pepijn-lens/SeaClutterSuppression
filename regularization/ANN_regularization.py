import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Generate synthetic data
n_samples = 50
X_np = np.linspace(-7, 7, n_samples)
y_true_np = 1 - 0.1 * X_np + 0.1 * X_np**2 - 0.01 * X_np**3 + 0.001 * X_np**4 + 0.001 * X_np**5
noise = np.random.normal(0, 2, n_samples)
y_np = y_true_np + noise

# Dense grid for plotting
X_dense_np = np.linspace(-7, 7, 300)
y_true_dense = 1 - 0.1 * X_dense_np + 0.1 * X_dense_np**2 - 0.01 * X_dense_np**3 + 0.001 * X_dense_np**4 + 0.001 * X_dense_np**5

# Convert to PyTorch tensors
X = torch.tensor(X_np, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
X_dense = torch.tensor(X_dense_np, dtype=torch.float32).view(-1, 1)

# Hyperparameters
lambda_reg = 0.1 # Regularization strength on second derivative
lr = 1e-3
epochs = 3000

hidden_size = 128

# Neural network model
model = nn.Sequential(
    nn.Linear(1, hidden_size),
    nn.Tanh(),
    nn.Dropout(p=0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.Tanh(),
    nn.Dropout(p=0.2),
    nn.Linear(hidden_size, 1)
)

optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=lambda_reg)
mse_loss = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = mse_loss(outputs, y)  # Weight decay is handled by optimizer

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X).squeeze().numpy()
    y_pred_dense = model(X_dense).squeeze().numpy()

# Plotting
os.makedirs('regularization', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.scatter(X_np, y_np, label='Noisy Data', color='blue', alpha=0.6)
plt.plot(X_dense_np, y_true_dense, label='True Function (Degree 5)', color='green', linewidth=2)
plt.plot(X_dense_np, y_pred_dense, label='NN Prediction', color='red', linewidth=2)
plt.title(f'Neural Network Fit with dropout (λ={lambda_reg})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.ylim(min(y_np) - 5, max(y_np) + 5)
plt.savefig('regularization/nn_dropout_fit.png')
