import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import MLDataGenerator, DatasetInfo
from classical.linear_models.linear_regression import LinearRegressor

def visualize_learning_rates():
    # Generate dataset
    generator = MLDataGenerator(random_state=42)
    info = DatasetInfo(
        name='linear_regression',
        n_samples=100,
        n_features=1,
        noise=0.3
    )
    X, y = generator.make_regression(info)
    
    # Different learning rates to try
    learning_rates = [0.00001, 0.0001, 0.001, 0.05]
    epochs = 100
    
    # Create subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Plot 1: Data fitting process
    for i, lr in enumerate(learning_rates, 1):
        # Train model
        model = LinearRegressor(lr=lr, epochs=epochs)
        model.fit(X, y)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Plot data and prediction line
        plt.subplot(2, len(learning_rates), i)
        plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
        plt.plot(sorted(X), sorted(y_pred), color='red', label='Prediction')
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        
        # Plot cost history
        plt.subplot(2, len(learning_rates), i + len(learning_rates))
        plt.plot(model.cost_history)
        plt.title(f'Cost History (lr={lr})')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.yscale('log')  # Log scale to better see the changes
    
    plt.tight_layout()
    plt.show()

def visualize_training_progression():
    # Generate dataset
    generator = MLDataGenerator(random_state=42)
    info = DatasetInfo(
        name='linear_regression',
        n_samples=100,
        n_features=1,
        noise=0.3
    )
    X, y = generator.make_regression(info)
    
    # Training parameters
    lr = 0.01
    epochs = 100
    snapshots = [0, 4, 9, 24, 49, 99]  # Epochs at which to show the fit
    
    # Train model and save states
    model = LinearRegressor(lr=lr, epochs=epochs)
    predictions = []
    
    # Initialize
    n_samples, n_features = X.shape
    model.weights = np.random.uniform(low=-1.0, high=1.0, size=n_features)
    model.bias = 0
    
    # Collect predictions at different epochs
    for epoch in range(epochs):
        y_pred = model.predict(X)
        
        if epoch in snapshots:
            predictions.append((epoch, y_pred.copy()))
        
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)
        
        model.weights -= lr * dw
        model.bias -= lr * db
        
        cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
        model.cost_history.append(cost)
    
    # Plotting
    fig = plt.figure(figsize=(20, 10))
    
    # Plot progression of fit
    for i, (epoch, y_pred) in enumerate(predictions, 1):
        plt.subplot(2, 3, i)
        plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
        plt.plot(sorted(X), sorted(y_pred), color='red', label='Prediction')
        plt.title(f'Epoch {epoch}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot cost history
    plt.figure(figsize=(10, 5))
    plt.plot(model.cost_history)
    plt.title('Cost History')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Visualizing effect of different learning rates...")
    visualize_learning_rates()
    
    print("\nVisualizing training progression...")
    visualize_training_progression()