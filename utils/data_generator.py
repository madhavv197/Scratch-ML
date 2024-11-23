# scratch-ml/utils/data_generator.py

import numpy as np
from typing import Tuple, Dict, Union, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class DatasetInfo:
    """Class to hold dataset information and metadata"""
    name: str
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    noise: float = 0.1
    random_state: int = 42

class MLDataGenerator:
    """
    Data generator for various machine learning problems.
    Generates synthetic datasets for:
    - Classification (Linear and Non-linear)
    - Regression (Linear and Non-linear)
    - Clustering
    - Binary and Multi-class problems
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def make_blobs(self, info: DatasetInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Generate isotropic Gaussian blobs for clustering/classification"""
        centers = np.random.uniform(-10, 10, (info.n_classes, info.n_features))
        X = np.zeros((info.n_samples, info.n_features))
        y = np.zeros(info.n_samples)
        
        samples_per_class = info.n_samples // info.n_classes
        
        for i in range(info.n_classes):
            start_idx = i * samples_per_class
            end_idx = start_idx + samples_per_class
            
            # Generate samples for each cluster
            X[start_idx:end_idx] = (np.random.randn(samples_per_class, info.n_features) * info.noise + 
                                  centers[i])
            y[start_idx:end_idx] = i
            
        # Shuffle the data
        idx = np.random.permutation(info.n_samples)
        return X[idx], y[idx]
    
    def make_moons(self, info: DatasetInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two interleaving half circles"""
        n_samples_out = info.n_samples // 2
        n_samples_in = info.n_samples - n_samples_out

        # Outer circle
        linspace_out = np.linspace(0, np.pi, n_samples_out)
        X_out = np.vstack([np.cos(linspace_out), np.sin(linspace_out)]).T
        
        # Inner circle
        linspace_in = np.linspace(0, np.pi, n_samples_in)
        X_in = np.vstack([1 - np.cos(linspace_in), 1 - np.sin(linspace_in) - 0.5]).T
        
        X = np.vstack([X_out, X_in])
        y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])
        
        # Add noise
        X += np.random.normal(0, info.noise, X.shape)
        
        # Shuffle
        idx = np.random.permutation(info.n_samples)
        return X[idx], y[idx]
    
    def make_regression(self, info: DatasetInfo, 
                       non_linear: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regression dataset with optional non-linearity"""
        X = np.random.uniform(-10, 10, (info.n_samples, info.n_features))
        
        if non_linear:
            y = np.sum(np.sin(X) + X**2, axis=1)
        else:
            coefficients = np.random.randn(info.n_features)
            y = np.dot(X, coefficients)
            
        # Add noise
        y += np.random.normal(0, info.noise * np.std(y), info.n_samples)
        return X, y
    
    def make_classification(self, info: DatasetInfo, 
                          non_linear: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate classification dataset with optional non-linearity"""
        if non_linear:
            return self.make_moons(info)
        else:
            return self.make_blobs(info)
    
    def make_svm_data(self, info: DatasetInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Generate linearly separable data with clear margin"""
        # Generate random hyperplane
        w = np.random.randn(info.n_features)
        w = w / np.linalg.norm(w)
        b = np.random.randn()
        
        # Generate points
        X = np.random.randn(info.n_samples, info.n_features)
        y = np.sign(np.dot(X, w) + b)
        
        # Add margin
        margin_mask = np.abs(np.dot(X, w) + b) < 1
        X[margin_mask] += 2 * y[margin_mask].reshape(-1, 1) * w
        
        return X, y
    
    def generate_all_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate all types of datasets for testing different models"""
        datasets = {}
        
        # Classification datasets
        datasets['binary_classification'] = self.make_classification(
            DatasetInfo(name='binary_classification', n_samples=1000, n_features=2, n_classes=2))
        
        datasets['multi_classification'] = self.make_classification(
            DatasetInfo(name='multi_classification', n_samples=1000, n_features=2, n_classes=4))
        
        datasets['nonlinear_classification'] = self.make_classification(
            DatasetInfo(name='nonlinear_classification', n_samples=1000, n_features=2, n_classes=2),
            non_linear=True)
        
        # Regression datasets
        datasets['linear_regression'] = self.make_regression(
            DatasetInfo(name='linear_regression', n_samples=1000, n_features=2))
        
        datasets['nonlinear_regression'] = self.make_regression(
            DatasetInfo(name='nonlinear_regression', n_samples=1000, n_features=2),
            non_linear=True)
        
        # Clustering dataset
        datasets['clustering'] = self.make_blobs(
            DatasetInfo(name='clustering', n_samples=1000, n_features=2, n_classes=4))[0]
        
        # SVM dataset
        datasets['svm'] = self.make_svm_data(
            DatasetInfo(name='svm', n_samples=1000, n_features=2))
        
        return datasets

def visualize_dataset(X: np.ndarray, y: Optional[np.ndarray] = None, 
                     title: str = "Dataset Visualization"):
    """Utility function to visualize generated datasets"""
    
    plt.figure(figsize=(10, 6))
    
    if y is None:  # For clustering data
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5)
        
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Example usage of the data generator"""
    generator = MLDataGenerator(random_state=42)
    datasets = generator.generate_all_datasets()
    
    # Visualize each dataset
    for name, data in datasets.items():
        if name == 'clustering':
            visualize_dataset(data, title=f"{name} Dataset")
        else:
            X, y = data
            visualize_dataset(X, y, title=f"{name} Dataset")

if __name__ == "__main__":
    main()