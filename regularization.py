import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Seed for reproducibility
np.random.seed(0)

# Generate synthetic data
n_samples = 10
X = np.linspace(-1000, 1000, n_samples)
y_true = 1 - X + X**2 - X**3 + X**4 - X**5
noise = np.random.normal(0, 2000, n_samples)
y = y_true + noise

# Reshape X for sklearn
X_reshaped = X.reshape(-1, 1)

# Overfitting polynomial regression with high order (e.g., 25)
high_order = 25
model = make_pipeline(PolynomialFeatures(high_order), LinearRegression())
model.fit(X_reshaped, y)

y_pred = model.predict(X_reshaped)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Noisy Data', color='blue', alpha=0.6)
plt.plot(X, y_true, label='True Function (Degree 5)', color='green', linewidth=2)
plt.plot(X, y_pred, label=f'Overfit Model (Degree {high_order})', color='red', linewidth=2)
plt.title('Polynomial Regression: Overfitting Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()