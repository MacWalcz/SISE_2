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
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        sx = MLP.sigmoid(x)
        return sx * (1 - sx)
    
    def forward(self, x):
        activations = [x.reshape(-1, 1)]
        zs = []
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1])
            if self.use_bias:
                z += self.biases[i]
            zs.append(z)
            a = MLP.sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward(self, x, y, learning_rate):
        activations, zs = self.forward(x)
        y = y.reshape(-1, 1)
        delta = (activations[-1] - y) * MLP.sigmoid_derivative(zs[-1])

        deltas = [delta]
        for i in reversed(range(len(zs) - 1)):
            delta = np.dot(self.weights[i + 1].T, deltas[-1]) * MLP.sigmoid_derivative(zs[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(deltas[i], activations[i].T)
            if self.use_bias:
                self.biases[i] -= learning_rate * deltas[i]

    def train(self, X, Y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                self.backward(x, y, learning_rate)

    
