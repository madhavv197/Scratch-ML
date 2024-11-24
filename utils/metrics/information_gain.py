import numpy as np
from utils.metrics.entropy import entropy

def information_gain(X_col, y, threshold):
    parent_entropy = entropy(y)

    left_idx, right_idx = np.argwhere(X_col <= threshold).flatten(), np.argwhere(X_col > threshold).flatten()

    if len(left_idx) ==0 or len(right_idx) == 0:
        return 0

    n = len(y)
    n_left, n_right = len(left_idx), len(right_idx)
    entropy_left, entropy_right = entropy(y[left_idx]), entropy(y[right_idx])
    child_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right

    return parent_entropy - child_entropy