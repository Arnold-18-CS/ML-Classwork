# Task

# Using NumPy, generate 200 samples of x and corresponding samples of y.
# Use these values to create a linear regression model and obtain the slope 
# and bias coefficients using the Gradient Descent technique with a learning rate of your choice.
# Store the corresponding MSE, Slope, and Bias values in three arrays.
# Return the optimal values of Slope and Bias that have the lowest cost.

# Required

# Submit a Python 3 script file which when run, prints outs the optimal values of Slope and Bias on the Terminal.

# You are only allowed to use the NumPy library for this task.
# Use of any other library especially Sklearn (or similar variants) will lead to an automatic 0 score.
# Feel free to use Notebooks generated/shared in class for reference. 
# Submitting a file type of any other kind (anything that doesn't have a .py extension) will also lead
# to a zero score. 

# Submission Link:

# https://docs.google.com/forms/d/e/1FAIpQLSc9UL_WQw_dtIsKKeU6n7byPOV9Mh5SOhEzysxPwxH71uqaqg/viewform?usp=preview

import numpy as np
import matplotlib.pyplot as plt

# Seed to reproduce my personal results
np.random.seed(42)

# Generate 200 samples
num_samples = 200
X = np.random.uniform(0, 10, num_samples)

# Create y with some linear relationship plus random noise
true_slope = 2.5
true_bias = 4.0
noise = np.random.normal(0, 1, num_samples)
y = true_slope * X + true_bias + noise

# Reshape x to work with our model
X_reshaped = X.reshape(-1, 1)

# My implementation for Linear Regression
class LinearRegression:
    def __init__(self, lr=0.01, n_it=1000):
        self.lr = lr
        self.n_it = n_it
        self.weights = None
        self.bias = None

        # Arrays to store MSE, slope and bias values during training
        self.mse_history = np.zeros(n_it)
        self.slope_history = np.zeros(n_it)
        self.bias_history = np.zeros(n_it)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias as zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent the specified iterations
        for i in range(self.n_it):
            # Linear model predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
            # Calculate and store MSE, slope and bias for this iteration
            self.mse_history[i] = np.mean((y_pred - y) ** 2)
            self.slope_history[i] = self.weights[0]
            self.bias_history[i] = self.bias
            
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def get_optimal_parameters(self):
        min_mse_idx = np.argmin(self.mse_history)
        return self.slope_history[min_mse_idx], self.bias_history[min_mse_idx], self.mse_history[min_mse_idx]

# Create and train the model
lr_model = LinearRegression(lr=0.01, n_it=1000)
lr_model.fit(X_reshaped, y)

# Get optimal parameters
optimal_slope, optimal_bias, optimal_mse = lr_model.get_optimal_parameters()

# Print the results
print(f"Optimal Slope: {optimal_slope}")
print(f"Optimal Bias: {optimal_bias}")
print(f"Optimal MSE: {optimal_mse}")

# Visualize the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data Points')
plt.plot(X, optimal_slope * X + optimal_bias, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression with Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig("./plot.png")
plt.show()

# Visualize MSE over iterations
plt.figure(figsize=(10, 6))
plt.plot(lr_model.mse_history)
plt.title('Mean Squared Error vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig("./mse_plot.png")
plt.show()