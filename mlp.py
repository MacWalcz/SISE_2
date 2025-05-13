import numpy as np
import pickle


class MLP:
    def __init__(self, layers ,use_bias = True):
        self.layers = layers
        self.use_bias = use_bias
        self.weights = []
        self.biases = []
      

        for i in range(len(layers) - 1):
            weight = np.random.uniform(-1, 1, (layers[i + 1], layers[i]))
            self.weights.append(weight)
            bias = np.random.uniform(-1, 1, (layers[i + 1], 1)) if use_bias else None
            self.biases.append(bias)

        self.v_w= [np.zeros_like(W) for W in self.weights]
        if self.use_bias:
            self.v_b = [np.zeros_like(b) for b in self.biases]
    
    @staticmethod
    def sigmoid(x:float) -> float:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x:float) -> float:
        sx = MLP.sigmoid(x)
        return sx * (1 - sx)
    
    def forward(self, x:np.array):
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

    def backward(self, x, y):
        activations, zs = self.forward(x)
        y = y.reshape(-1, 1)
        delta = (activations[-1] - y) * MLP.sigmoid_derivative(zs[-1])

        deltas = [delta]
        for i in reversed(range(len(zs) - 1)):
            delta = np.dot(self.weights[i + 1].T, deltas[-1]) * MLP.sigmoid_derivative(zs[i])
            deltas.append(delta)

        deltas.reverse()

        dW, dB = [], []

        for i in range(len(self.weights)):
           a_previous = activations[i] if i > 0 else x.reshape(-1, 1)
           dW.append(deltas[i] @ a_previous.T)
           dB.append(deltas[i])

        return dW, dB

    def train(self, X, Y, epochs=1000, learning_rate=0.1, momentum=0.9):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                dw, db = self.backward(x, y)
                for i in range(len(self.weights)):
                    self.v_w[i] = momentum * self.v_w[i] - learning_rate * dw[i]
                    self.weights[i] += self.v_w[i]
                
                    if self.use_bias:
                        self.v_b[i] = momentum * self.v_b[i] - learning_rate * db[i]
                        self.biases[i] += self.v_b[i]

            
                    

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases, self.layers, self.use_bias), f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            weights, biases, layers, use_bias = pickle.load(f)
            mlp = MLP(layers, use_bias)
            mlp.weights = weights
            mlp.biases = biases
            return mlp

    @staticmethod
    def load_dataset(filename:str) -> tuple[np.array,np.array]:
        X = []
        Y = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                try:
                    x_part, y_part = line.split()
                    x_vals = [float(v) for v in x_part.split(',')]
                    y_vals = [float(v) for v in y_part.split(',')]
                    X.append(x_vals)
                    Y.append(y_vals)
                except ValueError as e:
                    raise ValueError(f"Nieprawidłowy format wiersza: {line}") from e
        return np.array(X), np.array(Y)
    
    
