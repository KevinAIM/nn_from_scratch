import numpy as np
from activations import cost_derivative, sigmoid_derivative

class Network:
    def __init__(self, sizes):
        #sizes is a list
        self.weights = []
        self.biases = []
        for i in range(1,len(sizes)):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]) * np.sqrt(1 / sizes[i-1]))
            self.biases.append(np.random.randn(sizes[i]))

    def forward(self, input):
        activations = []
        z_values = []

        for weight, bias in zip(self.weights, self.biases):
            x = np.dot(input, weight.T)
            z_values.append(x)
            activation = 1 / (1 + np.exp(-(x + bias)))
            activations.append(activation)
            input = activation

        softmax = np.exp(input) / np.sum(np.exp(input))
        return softmax, activations, z_values
    
    def backward(self, input, actual, learning_rate):
        original_input = np.array(input)
        output, activations, z_values = self.forward(input)
        delta = cost_derivative(output, actual)

        for i in reversed(range(len(self.weights))):
            delta *= sigmoid_derivative(z_values[i])
            prev_activation = original_input if i == 0 else activations[i-1]
            weight_gradient = np.outer(delta, prev_activation)
            bias_gradient = delta

            self.weights[i] -= learning_rate * weight_gradient
            self.biases[i] -= learning_rate * bias_gradient

            delta = np.dot(self.weights[i].T, delta)



if __name__ == "__main__":
    net = Network([784, 128, 64, 10])
    actual = np.zeros(10)
    actual[3] = 1  # correct answer is digit 3

    before = net.forward(np.random.rand(784))
    print("before training:", before[0])

    for i in range(1000):
        net.backward(np.random.rand(784), actual, 0.1)

    after = net.forward(np.random.rand(784))
    print("after training:", after[0])
    prediction = np.argmax(after[0])
    print("predicted digit:", prediction)