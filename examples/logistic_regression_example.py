import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import MLDataGenerator, DatasetInfo
from foundational.linear_models.logistic_regression import LogisticRegressor

def visualize_decision_boundary(X, y, model, ax):
    """Plot decision boundary and data points"""
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Make predictions on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu_r')
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.7)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.7)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

def visualize_training():
    # Generate binary classification data
    generator = MLDataGenerator(random_state=42)
    info = DatasetInfo(
        name='binary_classification',
        n_samples=100,
        n_features=2,
        n_classes=2,
        noise=0.2
    )
    X, y = generator.make_blobs(info)
    
    # Training parameters
    epochs = 1000
    snapshots = [0, 1, 3, 5, 10, 20]  # Epochs to visualize
    
    # Create and train model while visualizing
    model = LogisticRegressor(lr=0.1, epochs=epochs)
    
    # Setup plotting
    fig = plt.figure(figsize=(20, 10))
    
    # Initialize model parameters for snapshot visualization
    n_samples, n_features = X.shape
    model.weights = np.random.uniform(low=-1.0, high=1.0, size=n_features)
    model.bias = 0
    
    # Plot initial decision boundary
    for epoch in range(epochs):
        # Make predictions
        linear_pred = np.dot(X, model.weights) + model.bias
        y_pred = model.sigmoid(linear_pred)
        
        # Compute gradients
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)
        
        # Update parameters
        model.weights -= model.lr * dw
        model.bias -= model.lr * db
        
        # Compute and store cost
        cost = -(1/n_samples) * np.sum(y * np.log(y_pred + 1e-15) + 
                                     (1-y) * np.log(1 - y_pred + 1e-15))
        model.cost_history.append(cost)
        
        # Visualize at snapshot epochs
        if epoch in snapshots:
            ax = fig.add_subplot(2, 3, snapshots.index(epoch) + 1)
            visualize_decision_boundary(X, y, model, ax)
            ax.set_title(f'Epoch {epoch+1}')
            
    plt.tight_layout()
    plt.show()
    
    # Plot cost history
    plt.figure(figsize=(10, 5))
    plt.plot(model.cost_history)
    plt.title('Training Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Print final accuracy
    final_predictions = model.predict(X)
    accuracy = np.mean(final_predictions == y)
    print(f"\nFinal Training Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    visualize_training()