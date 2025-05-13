from mlp import *
import numpy as np
import pandas as pd

def main():
    neat = MLP((4,4,4), use_bias= True)
    X , Y = MLP.load_dataset( "SISE_2/autoenkoder.txt")
   
    neat.train(X,Y,epochs=1000,learning_rate=0.1, momentum=0.95)
    cos, innecos = neat.forward(np.array([1,0,0,0]))
    print(f"wynik dla (1,0,0,0): \n {cos[-1]} ")
    cos, innecos = neat.forward(np.array([0,1,0,0]))
    print(f"wynik dla (0,1,0,0): \n  {cos[-1]} ")
    cos, innecos = neat.forward(np.array([0,0,1,0]))
    print(f"wynik dla (0,0,1,0): \n  {cos[-1]} ")
    cos, innecos = neat.forward(np.array([0,0,0,1]))
    print(f"wynik dla (0,0,0,1): \n  {cos[-1]} ")


    model = MLP(layers=(4, 5, 3), use_bias=True)
   

    X, Y = MLP.load_iris('SISE_2/iris.data')  # X:(150,4), Y:(150,3)

    # 2) Shuffle + split 80/20
    np.random.seed(32)
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # 3) Normalizacja według X_train
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    model.train(X_train, Y_train,epochs=1000,learning_rate=0.01,momentum=0.9)

    # 5) Predykcje na zbiorze testowym
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    y_true = []
    y_pred = []
    for x, y_vec in zip(X_test, Y_test):
        activations, _ = model.forward(x)
        probs    = activations[-1].flatten()
        pred_idx = np.argmax(probs)      # <-- obliczamy tutaj
        true_idx = np.argmax(y_vec)      # <-- i tutaj

        y_pred.append(pred_idx)
        y_true.append(true_idx)

        print(f"Data:      {x.tolist()}")
        print(f"Predicted: {class_names[pred_idx]}  ({pred_idx})")
        print(f"Actual:    {class_names[true_idx]}  ({true_idx})")
        print("---")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    conf_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    print("\nConfusion Matrix:")
    print(conf_df)

    # Liczba poprawnych sklasyfikacji
    total_correct = (y_true == y_pred).sum()
    total = len(y_true)
    print(f"\nTotal correct: {total_correct}/{total} ({100 * total_correct/total:.2f}%)")

    print("\nCorrect by class:")
    for i, cname in enumerate(class_names):
        correct_i = confusion[i, i]
        total_i   = confusion[i, :].sum()
        print(f"  {cname}: {correct_i}/{total_i}")

    # Precision, Recall, F-measure
    precision = []
    recall    = []
    f_score  = []
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f_score.append(f1)

    metrics_df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F-measure": f_score
    }, index=class_names)
    print("\nPer-Class Metrics:")
    print(metrics_df)

if __name__ == '__main__':
    main()

