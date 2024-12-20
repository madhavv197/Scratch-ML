import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import MLDataGenerator, DatasetInfo
from foundational.clustering.knn import KNN

def demonstrate_knn():
    """
    Comprehensive demonstration of KNN for both classification and regression
    using generated datasets.
    """
    generator = MLDataGenerator()
    
    def demonstrate_classification():
        print("=== KNN Classification Demonstration ===")
        
        info = DatasetInfo(
            name='nonlinear_classification',
            n_samples=300,
            n_features=2,
            n_classes=2,
            noise=0.1
        )
        X, y = generator.make_moons(info)
        
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        train_size = int(0.7 * len(X))
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        k_values = [1, 3, 7, 15]
        plt.figure(figsize=(20, 5))
        
        for i, k in enumerate(k_values, 1):
            knn = KNN(k=k, problem_type='classification')
            knn.fit(X_train, y_train)
            
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()], p=1)
            Z = np.array(Z, dtype=float)
            Z = Z.reshape(xx.shape)
            
            plt.subplot(1, 4, i)
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                       cmap='viridis', alpha=1, marker='o', label='Train')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                       cmap='viridis', alpha=1, marker='s', label='Test')
            
            accuracy = np.mean(knn.predict(X_test) == y_test)
            plt.title(f'KNN (k={k})\nAccuracy: {accuracy:.3f}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def demonstrate_regression():
        print("\n=== KNN Regression Demonstration ===")
        
        info = DatasetInfo(
            name='nonlinear_regression',
            n_samples=200,
            n_features=1,
            noise=0.1
        )
        
        X = np.linspace(0, 10, info.n_samples).reshape(-1, 1)
        y = np.sin(X.ravel()) + np.random.normal(0, info.noise, info.n_samples)
        
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        train_size = int(0.7 * len(X))
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        k_values = [1, 3, 7, 15]
        plt.figure(figsize=(20, 5))
        
        for i, k in enumerate(k_values, 1):
            knn = KNN(k=k, problem_type='regression')
            knn.fit(X_train, y_train)
            
            X_plot = np.linspace(0, 10, 500).reshape(-1, 1)
            y_pred = knn.predict(X_plot, p=1)
            
            plt.subplot(1, 4, i)
            plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train')
            plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test')
            plt.plot(X_plot, y_pred, color='red', label='Predictions')
            
            mse = np.mean((knn.predict(X_test) - y_test) ** 2)
            plt.title(f'KNN (k={k})\nMSE: {mse:.3f}')
            plt.legend()
            
        plt.tight_layout()
        plt.show()

    demonstrate_classification()
    demonstrate_regression()

if __name__ == "__main__":
    demonstrate_knn()