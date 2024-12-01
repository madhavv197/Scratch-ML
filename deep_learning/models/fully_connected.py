import numpy as np
from utils.data_generator import MLDataGenerator
from tqdm import tqdm

class FullyConnected():
   def __init__(self, input_size, hidden_sizes, output_size):
       self.input_size = input_size
       self.hidden_sizes = hidden_sizes
       self.output_size = output_size
       self.weights = self._init_weights()
   
   def _init_weights(self):
       weights = []
       layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
       
       for i in range(len(layer_sizes) - 1):
           scale = np.sqrt(2.0 / layer_sizes[i])
           weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]) * scale)
       
       return weights
   
   def _forward(self, X):
       current_activation = X
       activations = [current_activation]
       
       for weight_matrix in self.weights[:-1]:
           z = np.dot(current_activation, weight_matrix.T)
           current_activation = np.maximum(0, z)
           activations.append(current_activation)
       
       z_final = np.dot(current_activation, self.weights[-1].T)
       
       z_final = z_final - np.max(z_final, axis=1, keepdims=True)
       exp_scores = np.exp(z_final)
       output_activation = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
       
       activations.append(output_activation)
       return activations
   
   def _backward(self, X, y, activations):
       batch_size = X.shape[0]
       gradients = []
       
       y_encoded = np.zeros((batch_size, self.output_size))
       y_encoded[np.arange(batch_size), y] = 1
       
       error = activations[-1] - y_encoded
       
       for layer_index in reversed(range(len(self.weights))):
           current_activations = activations[layer_index]
           
           layer_gradient = np.dot(error.T, current_activations)
           gradients.insert(0, layer_gradient / batch_size)
           
           if layer_index > 0:
               error = np.dot(error, self.weights[layer_index])
               error = error * (activations[layer_index] > 0)
       
       return gradients
   
   def _update_weights(self, gradients, learning_rate):
       for i in range(len(self.weights)):
           self.weights[i] -= learning_rate * gradients[i]
   
   def train(self, X, y, learning_rate=0.01, num_epochs=1, batch_size=128):
       n_samples = X.shape[0]
       
       for epoch in tqdm(range(num_epochs)):
           for i in range(0, n_samples, batch_size):
               batch_X = X[i:i + batch_size]
               batch_y = y[i:i + batch_size]
               
               activations = self._forward(batch_X)
               gradients = self._backward(batch_X, batch_y, activations)
               self._update_weights(gradients, learning_rate)
   
   def predict(self, X):
       activations = self._forward(X)
       return np.argmax(activations[-1], axis=1)

generator = MLDataGenerator()
X_train, y_train, X_test, y_test = generator.get_mnist()

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

neural_network = FullyConnected(input_size=784, hidden_sizes=[100, 100], output_size=10)
print(f"Training data shape: {X_train_reshaped.shape}, Labels shape: {y_train.shape}")

neural_network.train(X_train_reshaped, y_train, learning_rate=0.1, num_epochs=1, batch_size=1)

predictions = neural_network.predict(X_test_reshaped)
accuracy = np.mean(predictions == y_test)
print(f"Test accuracy: {accuracy}")