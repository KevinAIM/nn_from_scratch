import numpy as np

class Network:
    def __init__(self, sizes):
        #sizes is a list
        self.weights = []
        self.biases = []
        for i in range(1,len(sizes)):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]))
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

if __name__ == "__main__":
    net = Network([784, 16, 16, 10])
    output = net.forward(np.random.rand(784))
    print(output)