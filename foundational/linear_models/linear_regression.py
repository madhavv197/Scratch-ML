"""
Linear Regression is a supervised learning algorithm used to model linear relationships between input features and a continuous target variable.
It works by finding optimal weights (coefficients) and bias (intercept) that minimize the difference between predicted and actual values.

How it works:
1. Initialize model parameters:
  - Weights (w): Coefficients for each feature
  - Bias (b): Y-intercept term
  - Together they form the hypothesis function h(x) = wx + b

2. Training process (Gradient Descent):
  - Make predictions using current parameters: y_pred = wx + b
  - Calculate error (difference between predictions and actual values)
  - Compute gradients (direction of steepest increase in error):
    * dw = (1/m)∑(y_pred - y)x  (for weights)
    * db = (1/m)∑(y_pred - y)   (for bias)
    where m is number of samples
  - Update parameters in opposite direction of gradients:
    * w = w - α * dw
    * b = b - α * db
    where α is learning rate
  - Repeat until convergence or maximum iterations reached

3. Cost function (Mean Squared Error):
  J(w,b) = (1/2m)∑(y_pred - y)²
  The (1/2) term makes derivatives cleaner

4. Prediction:
  Once trained, predict new values using h(x) = wx + b

Key hyperparameters:
- Learning rate (α): Controls step size in gradient descent
- Number of epochs: How many times to iterate through the data

"""

import numpy as np

class LinearRegressor():
    def __init__(self,lr, epochs) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = self.predict(X)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
