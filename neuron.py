import numpy as np
from activations import softmax, sigmoid, sigmoid_derivative, cost, cost_derivative

def activate(input, weight, bias):
    x = np.dot(input, weight)
    sigmoid = 1 / (1 + np.exp(-(x + bias)))
    return sigmoid

def layer(input, weight, bias):
    weight = np.array(weight)
    x = np.dot(input, weight.T)
    sigmoid = 1 / (1 + np.exp(-(x + bias)))
    return sigmoid

def main():

    result = layer([1, 2, 3], [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5], [0.5, 0.2, 0.1], [0.3, 0.3, 0.3]], [-1, 0, 1, -0.5])
    result2 = layer(result, [[0.2, 0.4, 0.6, 0.1], [0.1, 0.3, 0.5, 0.2], [0.5, 0.2, 0.1, 0.3], [0.3, 0.3, 0.3, 0.4]], [-1, 0, 1, -0.5])
    squish = softmax(result2)
    print(result)
    print(result2)
    print(squish)

if __name__ == "__main__":
    main()