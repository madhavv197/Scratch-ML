"""
Logistic Regression is a supervised learning algorithm used for binary classification problems. Despite its name, it's a classification algorithm, not regression. 
It predicts the probability of an instance belonging to a particular class using the logistic (sigmoid) function.

How it works:
1. The Model:
   - Similar to linear regression, but wraps the linear combination in a sigmoid function
   - h(x) = σ(wx + b) where σ(z) = 1/(1 + e^(-z))
   - Sigmoid function maps any real number to [0,1], making it perfect for probabilities
   - If h(x) ≥ 0.5, predict class 1; else predict class 0

2. Training Process (Gradient Descent):
   - Initialize weights (w) and bias (b)
   - For each epoch:
     a) Forward pass: Calculate predicted probabilities using sigmoid
     b) Calculate gradients:
        * dw = (1/m)∑(h(x) - y)x
        * db = (1/m)∑(h(x) - y)
        where m is number of samples
     c) Update parameters:
        * w = w - α * dw
        * b = b - α * db
        where α is learning rate

3. Cost Function:
   - Unlike linear regression, uses Binary Cross-Entropy Loss:
   - J(w,b) = -(1/m)∑[y log(h(x)) + (1-y)log(1-h(x))]
   - This cost function is convex, ensuring a global minimum
   - Each term considers:
     * When y=1: -log(h(x)) → penalizes low probabilities for positive class
     * When y=0: -log(1-h(x)) → penalizes high probabilities for negative class

4. Decision Boundary:
   - The line/surface where h(x) = 0.5
   - Points on one side classify as 1, other side as 0
   - Linear boundary in feature space (can be made non-linear with feature engineering)

Key Differences from Linear Regression:
1. Output is probability between [0,1] (using sigmoid)
2. Uses cross-entropy loss instead of MSE
3. Decision boundary creates binary classification
4. Assumes data is linearly separable

"""

import numpy as np


class LogisticRegressor():
    def __init__(self, lr = 1e-3, epochs = 100) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
