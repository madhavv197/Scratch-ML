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