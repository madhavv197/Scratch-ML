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

from ...utils.metrics.distance_functions import minkowski_distance
import numpy as np
from collections import Counter

class KNN():
    def __init__(self, use, k=5) -> None:
        self.use = use
        self.k = k
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_single_datapoint(x) for x in X]
    
    def _predict_single_datapoint(self, x, method=2):
        distance = [minkowski_distance(x=x, y=y, p=method) for y in self.X_train]
        indices = np.argsort(distance)[:self.k]
        labels = [self.y_train[i] for i in indices]
        if self.use == 'classification'.upper():
            return Counter(labels).most_common()
        elif self.use == 'regression'.upper():
            return np.mean(labels)
