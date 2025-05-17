import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error, r2_score
import os

# Seed for reproducibility
np.random.seed(0)

# Generate synthetic data
n_samples = 34
X = np.linspace(-7, 7, n_samples)
y_true = 1 - 0.1 * X + 0.1 * X**2 - 0.01 * X**3 + 0.001 * X**4 + 0.001 * X**5
noise = np.random.normal(0, 1.5, n_samples)
y = y_true + noise

# Dense grid for prediction and plotting
X_dense = np.linspace(-7, 7, 300)
y_true_dense = 1 - 0.1 * X_dense + 0.1 * X_dense**2 - 0.01 * X_dense**3 + 0.001 * X_dense**4 + 0.001 * X_dense**5

# Settings
degree = 30
lambda_reg = 2e-2 # Regularization strength on second derivative

# Build design matrix manually (Vandermonde)
def design_matrix(x, deg):
    return np.vstack([x**i for i in range(deg + 1)]).T

Φ = design_matrix(X, degree)
Φ_dense = design_matrix(X_dense, degree)

# Approximate second derivative operator (D2 @ w gives second derivative at sample points)
def second_derivative_matrix(x, deg):
    dx = x[1] - x[0]
    D2 = np.zeros((len(x), deg + 1))
    for i in range(2, deg + 1):
        D2[:, i] = i * (i - 1) * x ** (i - 2)
    return D2

D2 = second_derivative_matrix(X_dense, degree)

# Solve the penalized least squares problem:
# w = argmin ||Φw - y||^2 + λ * ||D2w||^2
A = Φ.T @ Φ + lambda_reg * (D2.T @ D2)
b = Φ.T @ y
w = np.linalg.solve(A, b)

# Predict
y_pred = Φ @ w
y_pred_dense = Φ_dense @ w

# Metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Plot
os.makedirs('regularization', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Noisy Data', color='blue', alpha=0.6)
plt.plot(X_dense, y_true_dense, label='True Function (Degree 5)', color='green', linewidth=2)
plt.plot(X_dense, y_pred_dense, label=f'Predicted (Degree {degree})', color='red', linewidth=2)
plt.title(f'Manual Polynomial Fit (Degree {degree})\n2nd Derivative Penalty λ={lambda_reg} | RMSE={rmse:.3f} | R²={r2:.3f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.ylim(min(y) - 5, max(y) + 5)
plt.savefig('regularization/manual_regularized_fit.png')
plt.close()


