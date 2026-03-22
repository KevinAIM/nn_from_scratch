import numpy as np

#currently redudant. Just learning for single neuron 
def activate(input, weight, bias):
    x = np.dot(input, weight)
    sigmoid = 1 / (1 + np.exp(-(x + bias)))
    return sigmoid

def layer(input, weight, bias):
    weight = np.array(weight)
    x = np.dot(input, weight.T)
    sigmoid = 1 / (1 + np.exp(-(x + bias)))
    return sigmoid

def softmax(input):
    summation = np.sum(np.exp(input))
    softmax = np.exp(input) / summation
    return softmax

def main():
    result = layer([1, 2, 3], [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5], [0.5, 0.2, 0.1], [0.3, 0.3, 0.3]], [-1, 0, 1, -0.5])
    squish = softmax(result)
    print(result)
    print(squish)

if __name__ == "__main__":
    main()