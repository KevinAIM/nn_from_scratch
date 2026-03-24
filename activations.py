import numpy as np

def activate(input, weight, bias):
    x = np.dot(input, weight)
    sigmoid = 1 / (1 + np.exp(-(x + bias)))
    return sigmoid

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