import numpy as np
from torchvision import datasets, transforms
import torch
from tqdm import tqdm

class Linear():
    def __init__(self, in_features, out_features) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weights = self._init_weights()
        self.bias = self._init_biases()

    def _init_weights(self):
        limit = np.sqrt(6 / (self.in_features + self.out_features))
        weights = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
        return weights
    
    def _init_biases(self):
        bias = np.zeros((1, self.out_features))
        return bias
    
    def __call__(self, X):
        return np.dot(X,self.weights) + self.bias
    
class MLP():
    def __init__(self, in_features, hidden_size, out_features):
        self.input = Linear(in_features, hidden_size)
        self.hidden = Linear(hidden_size, hidden_size)
        self.output = Linear(hidden_size, out_features)

    def _activation_fn(self, z):
        return np.where(z > 0, z, 0.01 * z)

    def _activation_fn_derivative(self, z):
        return np.where(z > 0, 1, 0.01)

    def _softmax(self, z):
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    
        softmax_output = z_exp / np.sum(z_exp, axis=1, keepdims=True)
        return softmax_output

    def _loss_fn(self, y_true, y_pred):
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def _forward(self, X):
        activations = []
        pre_activations = []
        activations.append(X)
        pre_activations.append(X)
        z1 = self.input(X)
        pre_activations.append(z1)
        a1 = self._activation_fn(z1)
        activations.append(a1)

        z2 = self.hidden(a1)
        pre_activations.append(z2)
        a2 = self._activation_fn(z2)
        activations.append(a2)

        z3 = self.output(a2)
        pre_activations.append(z3)
        a3 = self._softmax(z3)
        activations.append(a3)


        return a3, activations, pre_activations
    
    def _backward(self, output, y_encoded, activations, pre_activations):
        grads = []
        biases = []
        m = y_encoded.shape[0]
        dcdz_out = output-y_encoded
        biases.append(dcdz_out)
        dzdw_out = activations[-2]
        dcdw_out = np.dot(dzdw_out.T, dcdz_out) / m
        grads.append(dcdw_out)
        
        dcdz_hidden = np.dot(dcdz_out, self.output.weights.T) * self._activation_fn_derivative(pre_activations[-2])
        biases.append(dcdz_hidden)
        dzdw_hidden = activations[-3]
        dcdw_hidden = np.dot(dzdw_hidden.T, dcdz_hidden) / m
        grads.append(dcdw_hidden)


        dcdz_input = np.dot(dcdz_hidden, self.hidden.weights.T) * self._activation_fn_derivative(pre_activations[-3])
        biases.append(dcdz_input)
        dzdw_input = activations[-4]
        dcdw_input = np.dot(dzdw_input.T, dcdz_input) / m
        grads.append(dcdw_input)

        # print('dcdz', dcdz_out.shape)
        # print('dzdw',dzdw_out.shape)
        # print('dcdw',dcdw_out.shape)
        # print('dcdz',dcdz_hidden.shape)
        # print('dzdw',dzdw_hidden.shape)
        # print('dcdw',dcdw_hidden.shape)
        # print('dcdz',dcdz_input.shape)
        # print('dzdw',dzdw_input.shape)
        # print('dcdw',dcdw_input.shape)

        #print(grads)

        #print(f"Weight Gradients (Input): {grads[-1].mean()}, Hidden: {grads[-2].mean()}, Output: {grads[-3].mean()}")


        return grads, biases

    def _update_weights(self, lr, grads):
        self.input.weights -= lr*grads[-1]
        self.hidden.weights -= lr*grads[-2]
        self.output.weights -= lr*grads[-3]
    
    def _update_biases(self, lr, grads):
        self.output.bias -= lr * np.sum(grads[-3], axis=0, keepdims=True)
        self.hidden.bias -= lr * np.sum(grads[-2], axis=0, keepdims=True)
        self.input.bias -= lr* np.sum(grads[-1], axis=0, keepdims=True)

    def _compute_accuracy(self, predictions, labels):
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def train(self, X, labels, epochs, learning_rate, val_data=None, val_labels=None, validate_every=50):
        for epoch in tqdm(range(epochs), desc="Training"):
            output, activations, pre_activations = self._forward(X)
            grads, biases = self._backward(output, labels, activations, pre_activations)
            self._update_weights(lr=learning_rate, grads=grads)
            self._update_biases(lr=learning_rate, grads=biases)

            # Compute training accuracy
            train_accuracy = self._compute_accuracy(output, labels)

            if val_data is not None and epoch % validate_every == 0:
                val_output, _, _ = self._forward(val_data)
                val_loss = self._loss_fn(val_labels, val_output)
                val_accuracy = self._compute_accuracy(val_output, val_labels)
                print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}")

        print("Training Complete.")



if __name__ == "__main__":
    mlp = MLP(784, 128, 10)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    data, labels = next(iter(train_loader))

    flattened_data = data.view(data.shape[0], -1).numpy()
    labels = labels.numpy()

    num_classes = 10

    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1

    val_data, val_labels = next(iter(test_loader))
    flattened_val_data = val_data.view(val_data.shape[0], -1).numpy()
    val_labels = val_labels.numpy()

    one_hot_val_labels = np.zeros((val_labels.size, num_classes))
    one_hot_val_labels[np.arange(val_labels.size), val_labels] = 1

    mlp.train(flattened_data, one_hot_labels, 10000, learning_rate=3e-4, val_data=flattened_val_data, val_labels=one_hot_val_labels, validate_every=100)
    print('lolz')