"""
Decision Trees are supervised learning algorithms used for both classification and regression. They create a model that predicts the target by learning simple decision rules from the features in a tree-like structure.

How it works:
1. The Model:
   - A hierarchical structure of nodes and edges
   - Each internal node tests a feature against a threshold
   - Each leaf node contains a prediction (class label or value)
   - Predictions made by traversing from root to leaf based on feature values
   - Binary tree structure: each node splits into two children (left ≤ threshold, right > threshold)

2. Training Process (Recursive Partitioning):
   - Start at root with all data
   - For each node, while stopping criteria not met:
     a) Find best feature and threshold for splitting
     b) Split data into left (≤ threshold) and right (> threshold) subsets
     c) Create child nodes with respective subsets
     d) Recursively continue process on child nodes
   - Create leaf node when stopping criteria met:
     * Maximum depth reached
     * Minimum samples for split not met
     * All samples have same target value
     * No more features to split on

3. Finding Best Split:
   - For classification, uses metrics like:
     * Information Gain (using entropy)
     * Gini Impurity
   - Process:
     * Try each feature and possible threshold
     * Calculate metric before and after split
     * Choose split maximizing improvement
     * Information Gain = Entropy(parent) - WeightedAvg(Entropy(children))

4. Prediction Process:
   - Start at root node
   - At each internal node:
     * Compare feature value with threshold
     * Go left if value ≤ threshold, right otherwise
   - Return prediction when leaf node reached

Key Features:
1. Non-parametric (no assumptions about data distribution)
2. Can handle both numerical and categorical data
3. Naturally handles multi-class problems
4. Can capture non-linear relationships
5. Easy to interpret and visualize

Hyperparameters:
1. max_depth: Controls tree depth
2. min_samples_split: Minimum samples needed for split
3. min_samples_leaf: Minimum samples in leaf nodes
4. max_features: Number of features to consider for best split

"""

import numpy as np
from collections import Counter
from utils.metrics.information_gain import information_gain

class Node():
    def __init__(self, feature = None, threshold = None, left=None, right = None, val = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val
    
    def is_leaf_node(self):
        return self.val is not None
    
class DecisionTree:
    def __init__(self, min_samples = 2, max_depth = 50, n_features = None) -> None:
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self,X,y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

        self.root = self._growtree(X,y)
    
    def _growtree(self,X,y, depth =0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth>=self.max_depth or n_labels == 1 or n_samples < self.min_samples:
            return Node(val=self._getlabel(y))
        
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_thresh = self._bestsplit(X, y, feat_idxs)
        left_idx, right_idx = np.argwhere(X[:, best_feature] <= best_thresh).flatten(), np.argwhere(X[:, best_feature] > best_thresh).flatten()
        left, right = self._growtree(X[left_idx, :], y[left_idx ], depth+1), self._growtree(X[right_idx, :], y[right_idx ], depth+1)

        return Node(best_feature, best_thresh, left,right)

    def _bestsplit(self,  X,y, feat_idx):
        best_gain = -1
        split_idx, split_threshold = None, None

        for idx in feat_idx:
            X_col = X[: , idx]
            thresholds = np.unique(X_col)
            for thresh in thresholds:
                gain = information_gain(X_col, y,thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = thresh

        return split_idx, split_threshold

    def _getlabel(self,y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traversetree(x, self.root) for x in X])

    def _traversetree(self,  x, node):
        if node.is_leaf_node():
            return node.val
        feature_val = x[node.feature]
        if feature_val <= node.threshold:
            return self._traversetree(x, node.left)
        return self._traversetree(x, node.right)