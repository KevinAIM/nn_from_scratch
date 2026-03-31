from network import Network
from data import x_train, y_train, one_hot, x_test, y_test
from activations import cost
import numpy as np

def evaluation(network, x_test, y_test):
    correct = 0
    for i in range(len(x_test)):
        output, _, _ = network.forward(x_test[i])
        prediction = np.argmax(output)
        if prediction == y_test[i]:
            correct += 1
    accuracy = correct / len(x_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    net = Network([784, 128, 64, 10])
    for epoch in range(5):
        for i in range(len(x_train)):
            net.backward(x_train[i], one_hot(y_train[i]), 0.5)

            if i % 1000 == 0:
                output, _, _ = net.forward(x_train[i])
                c = cost(output, one_hot(y_train[i]))
                print(f"image {i}, cost: {c}")
    
        evaluation(net, x_test, y_test)
