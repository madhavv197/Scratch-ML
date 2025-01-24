import numpy as np
from torchvision import datasets, transforms
import torch


class MLP():
    def __init__(self, input_size, output_size, n_hidden, n_neurons) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.weights, self.biases = self._init_weights()

    def _init_weights(self):
        # initialise weights for input layer
        input_weights = np.random.rand(1, self.n_neurons, self.input_size)
        input_bias = np.random.rand(1, self.n_neurons, 1)
        

        # initialise hidden layers
        hidden_weights = np.random.rand(self.n_hidden, self.n_neurons, self.n_neurons)
        hidden_bias = np.random.rand(self.n_hidden, self.n_neurons, 1)

        # initialise output_layer
        output_weights = np.random.rand(1, self.output_size, self.n_neurons)
        output_bias = np.random.rand(1, self.output_size, 1)

        print(input_weights.shape, input_bias.shape)
        print(hidden_weights.shape, hidden_bias.shape)
        print(output_weights.shape, output_bias.shape)

        weights = {
            'input': input_weights,
            'hidden': hidden_weights,
            'output': output_weights
        }

        biases = {
            'input': input_bias,
            'hidden': hidden_bias,
            'output': output_bias
        }
        
        return weights, biases
    
    def _activation_fn(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss_fn(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        
        # Compute the cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def _softmax(self, z):
        z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    
        softmax_output = z_exp / np.sum(z_exp, axis=0, keepdims=True)
        return softmax_output

    def _forward(self, input):
        current_activation = input
        pre_activations = [current_activation]
        activations = [current_activation]

        # input layers
        z = np.dot(self.weights['input'][0], current_activation) + self.biases['input'][0]
        pre_activations.append(z)
        current_activation = self._activation_fn(z)
        activations.append(current_activation)

        # hidden layers
        for i in range(self.n_hidden):
            z = np.dot(self.weights['hidden'][i], current_activation) + self.biases['hidden'][i]
            pre_activations.append(z)
            current_activation = self._activation_fn(z)
            activations.append(current_activation)

        # output layers
        z = np.dot(self.weights['output'][0], current_activation) + self.biases['output'][0]
        pre_activations.append(z)
        output = self._softmax(z)
        activations.append(output)

        return output, activations

    def _backward(self, output, labels, activations, pre_activations):
        dlda_out = output - labels
        print(dlda_out.shape)
        dadz_out = output*(1-output)
        print(dadz_out.shape)
        exit()
        dldz_out = dlda_out*dadz_out
        dzdw_out = output
        dwdl_out = dzdw_out*dldz_out



    def _update_weights():
        pass

if __name__ == "__main__":
    mlp = MLP(784, 10, 1, 2)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    data, labels = next(iter(train_loader))

        # Flatten the data and convert to NumPy
    flattened_data = data.view(data.shape[0], -1).numpy()  # Shape: (batch_size, 784)

    # Transpose the data to match the input format of the MLP (input_size, batch_size)
    input_data = flattened_data.T  # Shape: (784, batch_size)
    # Print the shapes
    output, activations = mlp._forward(input_data)
    torch.nn.

    # Print the output and activations
    print("Activations of the layers:")
    for i, activation in enumerate(activations):
        print(f"Layer {i} activation shape: {activation.shape}")
    print('lolz')