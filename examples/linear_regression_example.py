import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import MLDataGenerator, DatasetInfo
from foundational.linear_models.linear_regression import LinearRegressor

def visualize_learning_rates():
    generator = MLDataGenerator(random_state=42)
    info = DatasetInfo(
        name='linear_regression',
        n_samples=100,
        n_features=1,
        noise=0.3
    )
    X, y = generator.make_regression(info)
    
    learning_rates = [0.00001, 0.0001, 0.001, 0.05]
    epochs = 100
    
    fig = plt.figure(figsize=(20, 10))
    
    for i, lr in enumerate(learning_rates, 1):

        model = LinearRegressor(lr=lr, epochs=epochs)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        plt.subplot(2, len(learning_rates), i)
        plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
        plt.plot(sorted(X), sorted(y_pred), color='red', label='Prediction')
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Visualizing effect of different learning rates...")
    visualize_learning_rates()
