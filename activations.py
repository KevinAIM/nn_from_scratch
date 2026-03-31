import numpy as np

def softmax(input):
    summation = np.sum(np.exp(input))
    softmax = np.exp(input) / summation
    return softmax

def sigmoid_derivative(input):
    sigmoid = 1 / (1 + np.exp(-input))
    return sigmoid * (1 - sigmoid)

def cost(predicted, actual):
    return np.mean((predicted - actual) ** 2)

def cost_derivative(predicted, actual):
    return 2 * (predicted - actual) / len(predicted)

def relu(input):
    return np.maximum(0, input)

def relu_derivative(input):
    return (input > 0).astype(float)