import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import MLDataGenerator, DatasetInfo
from foundational.trees.decision_tree import DecisionTree

def plot_decision_surface(X, y, tree, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = tree.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.8)

def visualize_tree_learning():
    generator = MLDataGenerator(random_state=42)
    info = DatasetInfo(
        name='binary_classification',
        n_samples=1000,
        n_features=2,
        n_classes=5,
        noise=1.2
    )
    X, y = generator.make_blobs(info)
    
    max_depths = [1, 2, 3, 5]
    fig, axes = plt.subplots(1, len(max_depths), figsize=(20, 5))
    
    for i, depth in enumerate(max_depths):
        tree = DecisionTree(max_depth=depth)
        tree.fit(X, y)
        
        plot_decision_surface(X, y, tree, axes[i])
        axes[i].set_title(f'Decision Surface (Max Depth={depth})')
        
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Visualizing decision tree learning process...")
    visualize_tree_learning()