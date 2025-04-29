import numpy as np

class MLP:
    def __init__(self, layers ,bias = True):
        self.layers = layers
        self.bias = bias
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            weight = np.random.uniform(-0.5, 0.5, (layers[i + 1], layers[i]))
            self.weights.append(weight)
            bias = np.random.uniform(-0.5, 0.5, (layers[i + 1], 1)) if bias else None
            self.biases.append(bias)


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        sx = sigmoid(x)
        return sx * (1 - sx)
