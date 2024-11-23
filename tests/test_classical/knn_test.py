# tests/test_classical/test_knn.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from ...scratch-ml.classical.clustering.knn import KNN
from scratch_ml.utils.distance_functions import minkowski_distance

def generate_classification_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for classification with three classes in a 2D space
    Returns: X_train, X_test, y_train, y_test
    """
    np.random.seed(42)
    
    # Generate three clusters for training
    n_samples = 100
    
    # Class 1: Cluster around (0, 0)
    X1 = np.random.randn(n_samples, 2) * 0.5 + np.array([0, 0])
    y1 = np.zeros(n_samples)
    
    # Class 2: Cluster around (2, 2)
    X2 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    y2 = np.ones(n_samples)
    
    # Class 3: Cluster around (-1, 2)
    X3 = np.random.randn(n_samples, 2) * 0.5 + np.array([-1, 2])
    y3 = np.ones(n_samples) * 2
    
    # Combine training data
    X_train = np.vstack([X1, X2, X3])
    y_train = np.hstack([y1, y2, y3])
    
    # Generate test data
    n_test = 50
    X_test = np.random.uniform(low=-2, high=3, size=(n_test, 2))
    
    # Get true labels for test data using minkowski distance (p=2 for Euclidean)
    centers = np.array([[0, 0], [2, 2], [-1, 2]])
    distances = np.array([np.array([minkowski_distance(x, center, p=2) 
                                  for x in X_test]) for center in centers])
    y_test = np.argmin(distances, axis=0)
    
    return X_train, X_test, y_train, y_test

def generate_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for regression using a noisy sine wave
    Returns: X_train, X_test, y_train, y_test
    """
    np.random.seed(42)
    
    # Generate training data
    X_train = np.random.uniform(0, 10, (200, 1))
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape)
    
    # Generate test data
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_test = np.sin(X_test)  # True values without noise
    
    return X_train, X_test, y_train, y_test

def visualize_classification(knn, X_train, X_test, y_train, y_test):
    """Visualize classification results with decision boundaries"""
    plt.figure(figsize=(15, 5))
    
    # Plot training data
    plt.subplot(121)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', 
                alpha=0.6, label='Training Data')
    plt.colorbar(scatter)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot test data and decision boundary
    plt.subplot(122)
    
    # Create mesh grid for decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Predict for each point in the mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    # Plot test points
    scatter_test = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                cmap='viridis', alpha=1, marker='x', s=100, 
                label='Test Data')
    plt.colorbar(scatter_test)
    
    plt.title('Decision Boundary and Test Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_regression(knn, X_train, X_test, y_train, y_test):
    """Visualize regression results"""
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
    
    # Plot true function
    plt.plot(X_test, y_test, color='green', label='True Function', linewidth=2)
    
    # Get predictions
    y_pred = knn.predict(X_test)
    
    # Plot predictions
    plt.plot(X_test, y_pred, color='red', label='KNN Predictions', linewidth=2)
    
    plt.title('KNN Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_classification():
    """Test KNN classification"""
    print("Testing KNN Classification...")
    X_train, X_test, y_train, y_test = generate_classification_data()
    
    for k in [3, 5, 7]:
        # Initialize and train KNN classifier
        knn_classifier = KNN(k=k, problem_type='classification')
        knn_classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = knn_classifier.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"Classification Accuracy (k={k}): {accuracy:.3f}")
        
        # Visualize classification results
        visualize_classification(knn_classifier, X_train, X_test, y_train, y_test)

def test_regression():
    """Test KNN regression"""
    print("\nTesting KNN Regression...")
    X_train, X_test, y_train, y_test = generate_regression_data()
    
    for k in [3, 5, 7]:
        # Initialize and train KNN regressor
        knn_regressor = KNN(k=k, problem_type='regression')
        knn_regressor.fit(X_train, y_train)
        
        # Make predictions
        y_pred = knn_regressor.predict(X_test)
        
        # Calculate MSE
        mse = np.mean((y_pred - y_test)**2)
        print(f"Regression MSE (k={k}): {mse:.3f}")
        
        # Visualize regression results
        visualize_regression(knn_regressor, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    test_classification()
    test_regression()