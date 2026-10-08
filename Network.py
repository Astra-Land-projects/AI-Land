"""
Neural Network Built From Scratch (NumPy only, no frameworks)
Implements a simple feedforward neural network with backpropagation
by hand - to understand exactly what's happening inside a neural
network, instead of relying on TensorFlow/PyTorch to do it for you.

Trains the network to solve the XOR problem, a classic example that
a single-layer model CANNOT solve, but a network with a hidden layer can.
"""

import numpy as np

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # Initialize weights randomly (small values work best to start)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
        self.bias_output = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    def forward(self, X):
        # Input -> hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Hidden -> output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output):
        # Calculate the error at the output
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Propagate the error back to the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases using the calculated gradients
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000, print_every=1000):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            loss = np.mean((y - output) ** 2)
            losses.append(loss)

            if epoch % print_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        return self.forward(X)


def main():
    # The XOR problem: output is 1 only if exactly one input is 1
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ])

    print("=" * 50)
    print("Training a neural network to solve XOR")
    print("(a problem that a simple linear model cannot solve)")
    print("=" * 50)

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    nn.train(X, y, epochs=10000, print_every=1000)

    print("\nFinal predictions:")
    predictions = nn.predict(X)
    for inputs, true_value, pred in zip(X, y, predictions):
        rounded = round(pred[0])
        correct = "✅" if rounded == true_value[0] else "❌"
        print(f"  Input: {inputs} -> Predicted: {pred[0]:.4f} (rounded: {rounded}, true: {true_value[0]}) {correct}")


if __name__ == "__main__":
    main()