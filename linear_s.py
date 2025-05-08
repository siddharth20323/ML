# %%
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (with Gaussian noise)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Random feature
y = 2 * X + 5 + np.random.randn(100, 1)  # Linear relation with noise

# Add a bias term (x0 = 1)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Gradient Descent for Linear Regression
def linear_regression(X, y, lr=0.01, epochs=1000):
    m = len(X)
    theta = np.random.randn(2, 1)  # Random initialization of weights
    for epoch in range(epochs):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradients
    return theta

# Train model
theta = linear_regression(X_b, y)

# Predict
y_pred = X_b.dot(theta)

# Compute MSE and R² score
mse = np.mean((y - y_pred) ** 2)
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gaussian Noise')
plt.legend()
plt.show()

# Print MSE and R² score
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")



# Mean Squared Error: 0.8066895146098613
# R² Score: 0.9764537349721645