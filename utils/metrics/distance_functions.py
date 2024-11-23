import numpy as np

## minkowski distance can be both manhattan or euclidean distance depending on required use case!

def minkowski_distance(x, y, p):
    if x_1.shape != y_1.shape:
        raise ValueError("Points must have same shape")
    return np.power(np.sum(np.power(np.abs(x - y), p)), 1/p)

def manhattan_distance(x,y):
    if x.shape != y.shape:
        raise ValueError("Points must have same shape")
    return np.sum(np.abs(x - y))  # Same as minkowski_distance(x, y, p=1)


def euclidean_distance(x,y):
    if x.shape != y.shape:
        raise ValueError("Points must have same shape")
    return np.sqrt(np.sum((x - y)**2)) # Same as minkowski_distance(x , y, p=2)

