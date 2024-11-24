import numpy as np
def entropy(y):
    counts = np.bincount(y)
    p_counts = counts/len(y)
    return -np.sum([p * np.log(p) for p in p_counts if p > 0])

