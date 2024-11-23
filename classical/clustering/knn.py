"""

KNN is a simple, instance-based learning algorithm used for classification (and regression) tasks. 
It works by finding the k-nearest data points to a given test point and making predictions based on the labels of these neighbors. Curretly implemented is a majority vote.

How it works:
1. Store the training data points and their corresponding labels.
2. For each test data point, calculate the distance to all training data points.
3. Sort the distances and find the k-nearest neighbors.
4. Determine the most common label (for classification) or average label (for regression) among the k-nearest neighbors.
5. Return the predicted label.

Minkowski distance is used to compute the distance between data points. By default, this implementation uses Euclidean distance (p=2).

"""

from utils.metrics.distance_functions import minkowski_distance
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5, problem_type='classification'):
        """
        Initialize KNN classifier/regressor
        
        Args:
            k (int): Number of neighbors to use
            problem_type (str): Either 'classification' or 'regression'
        """
        self.k = k
        self.problem_type = problem_type.lower()
        
    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X, p=2):
        """Predict for multiple samples"""
        X = np.array(X)
        predictions = [self._predict_single_datapoint(x, p=p) for x in X]
        return np.array(predictions)
    
    def _predict_single_datapoint(self, x, p):
        """Predict for a single sample"""
        distances = [minkowski_distance(x, y, p=p) for y in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        
        if self.problem_type == 'classification':
            return Counter(k_labels).most_common(1)[0][0]
        elif self.problem_type == 'regression':
            return np.mean(k_labels)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")