import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from numpy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D

# Seed
np.random.seed(0)

# Generate synthetic data
n_samples = 300
X = np.linspace(-7, 7, n_samples)
y_true = 1 - 0.1 * X + 0.1 * X**2 - 0.01 * X**3 + 0.001 * X**4 + 0.001 * X**5
noise = np.random.normal(0, 1.5, n_samples)
y = y_true + noise

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Design matrix
def design_matrix(x, deg):
    return np.vstack([x**i for i in range(deg + 1)]).T

# Second derivative regularization matrix
def second_derivative_matrix(x, deg):
    D2 = np.zeros((len(x), deg + 1))
    for i in range(2, deg + 1):
        D2[:, i] = i * (i - 1) * x ** (i - 2)
    return D2

# Exact lambda values of interest
lambdas = np.linspace(0, 1e-1, 20)  # Regularization parameters
degrees = np.arange(1, 50)  # Polynomial degrees 1 to 30

# Grid to store validation errors (shape: [#λs, #degrees])
rmse_grid = np.zeros((len(lambdas), len(degrees)))

# Loop over λ and degree
for i, lambda_reg in enumerate(lambdas):
    for j, deg in enumerate(degrees):
        Φ_train = design_matrix(X_train, deg)
        Φ_val = design_matrix(X_val, deg)
        D2 = second_derivative_matrix(X_train, deg)

        A = Φ_train.T @ Φ_train + lambda_reg * (D2.T @ D2)
        b = Φ_train.T @ y_train
        w = solve(A, b)

        y_pred_val = Φ_val @ w
        rmse_grid[i, j] = np.sqrt(mean_squared_error(y_val, y_pred_val))


# 3D plot (with exact lambda values)
X_grid, Y_grid = np.meshgrid(degrees, lambdas)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_grid, Y_grid, rmse_grid, cmap='viridis', edgecolor='k')

# Axis labels (showing exact λ values)
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('λ')
ax.set_zlabel('Validation RMSE')
ax.set_title('Validation Error Surface: Degree vs λ')
fig.colorbar(surf, shrink=0.5, aspect=10)

# Optional: adjust view angle or tick formatting
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.savefig('regularization/validation_surface_exact_lambda.png')
plt.show()
