import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import MLDataGenerator, DatasetInfo
from foundational.linear_models.logistic_regression import LogisticRegressor

def visualize_decision_boundary(X, y, model, ax):
    """Plot decision boundary and data points"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu_r')
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.7)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.7)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

def visualize_training():
    generator = MLDataGenerator(random_state=42)
    info = DatasetInfo(
        name='binary_classification',
        n_samples=100,
        n_features=2,
        n_classes=2,
        noise=0.2
    )
    X, y = generator.make_blobs(info)
    
    snapshots = [0, 1, 3, 5, 10, 20]  # Epochs to visualize
    
    
    fig = plt.figure(figsize=(20, 10))
        
    for epochs in snapshots:
        model = LogisticRegressor(lr=0.1, epochs=epochs)

        model.fit(X,y)
        
        ax = fig.add_subplot(2, 3, snapshots.index(epochs) + 1)
        visualize_decision_boundary(X, y, model, ax)
        ax.set_title(f'Epoch {epochs}')
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_training()