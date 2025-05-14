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

    def train(self, X, Y, epochs=1000, learning_rate=0.1, momentum=0.9, shuffle=False, error_threshold=None,filename="training_error_log.txt"):
        for epoch in range(epochs):
            # Shuffle dataset if requested
            if shuffle:
                perm = np.random.permutation(len(X))
                X = X[perm]
                Y = Y[perm]

            total_loss = 0

            for x, y in zip(X, Y):
                dw, db = self.backward(x, y)
                for i in range(len(self.weights)):
                    self.v_w[i] = momentum * self.v_w[i] - learning_rate * dw[i]
                    self.weights[i] += self.v_w[i]

                    if self.use_bias:
                        self.v_b[i] = momentum * self.v_b[i] - learning_rate * db[i]
                        self.biases[i] += self.v_b[i]

                # Calculate per-pattern loss (MSE) for early stopping
                out, _ = self.forward(x)
                error = out[-1] - y.reshape(-1, 1)
                total_loss += np.mean(error ** 2)

            avg_loss = total_loss / len(X)
            if epoch % 10 == 0:
                with open(filename, "a") as f:
                    f.write(f"{epoch},{avg_loss:.6f}\n")
        
            if error_threshold is not None and avg_loss <= error_threshold:
                break

    

    def test(self, X, Y):
        total_loss = 0
        correct = 0
        predictions = []
        test_log = []

        for x, y_true in zip(X, Y):
            activations, z_values = self.forward(x) 
            y_pred = activations[-1]
            y_true = y_true.reshape(-1, 1)

            
            error = y_pred - y_true
            mse = np.mean(error ** 2)
            total_loss += mse

            # Klasyfikacja – sprawdzenie poprawności
            pred_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            if pred_label == true_label:
                correct += 1

        
            entry = {
                "input": x.tolist(),
                "mse": mse,
                "desired_output": y_true.flatten().tolist(),
                "output_error": error.flatten().tolist(),
                "output_values": y_pred.flatten().tolist(),
                "output_weights": [w.tolist() for w in self.weights[-1]],  
                "hidden_outputs": [a.flatten().tolist() for a in activations[1:-1]],  
                "hidden_weights": [w.tolist() for w in self.weights[:-1]][::-1]  
        }

        test_log.append(entry)
        predictions.append((x, y_true.flatten(), y_pred.flatten(), error.flatten()))

        avg_loss = total_loss / len(X)
        accuracy = correct / len(X)

        return {
            "mse": avg_loss,
            "accuracy": accuracy,
            "predictions": predictions,
            "log": test_log
        }

                    
    def save_log(self, data, filename):
        # Funkcja zapisująca dane do pliku
        with open(filename, 'w') as f:
            for line in data:
                f.write(f"{line}\n")

    def save_log_dict(self, data: dict, filename: str):
        with open(filename, 'w') as f:
            for key, value in data.items():
                f.write(f"{key}: {value}\n")

    
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
    
    

    def training_mode(self):
        print("\n=== Tryb nauki ===")
    
        # Wczytywanie danych
        
        filename = input("Podaj nazwe pliku z danymi do nauki ")
        X,Y = MLP.load_dataset(filename)
        # Parametry użytkownika
        epochs = int(input("Podaj liczbę epok: "))
        learning_rate = float(input("Podaj współczynnik uczenia (learning rate): "))
        momentum = float(input("Podaj momentum: "))
        shuffle = input("Czy prezentować wzorce w losowej kolejności (tak/nie)? ").strip().lower() == 'tak'
        error_threshold = input("Podaj poziom błędu (opcjonalne, naciśnij Enter, aby pominąć): ")
        train_filename = input("Podaj nazwe pliku gdzie zostaną zapisane wartosći MSE w kolejnych epokach ")
        error_threshold = float(error_threshold) if error_threshold else None

        # Tworzenie i trenowanie modelu
        
        self.train(X, Y, epochs, learning_rate, momentum, shuffle, error_threshold,train_filename)

        # Zapis wyników do pliku
        
        print(f"Trening zakończony")

    def testing_mode(self):
        print("\n=== Tryb testowania ===")
    
        filename = input("Podaj nazwe pliku z danymi do testowania ")
        X, Y = MLP.load_dataset(filename)  # Przykład ładowania danych
    
        test_data = self.test(X,Y)
    
        # Zapis wyników do pliku
        log_filename = input("Podaj nazwę pliku do zapisania wyników testu: ")
        self.save_log_dict(test_data, log_filename)
        print(f"Wyniki testu zapisane w pliku {log_filename}")
