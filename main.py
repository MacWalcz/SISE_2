from mlp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_loss_from_file(filename):
    epochs = []
    losses = []

    # Wczytanie danych z pliku
    with open(filename, 'r') as f:
        for line in f:
            if ',' in line:
                epoch_str, loss_str = line.strip().split(',')
                epochs.append(int(epoch_str))
                losses.append(float(loss_str))

    
    plt.figure(figsize=(10, 6))
    plt.ylim(bottom=0)
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.title('MSE w czasie epok')
    plt.xlabel('Epoka')
    plt.ylabel('MSE (Błąd średniokwadratowy)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

def main():
    parser = argparse.ArgumentParser(description="SISE_2")
    parser.add_argument("--shape", 
                        type=lambda s: tuple(map(int, s.split(","))),
                        help="Podaj ilości neuronóww każdej wartswie w formie np. 4,2,4")
    parser.add_argument("--bias", action="store_true",help="Czy używać biasu")
    parser.add_argument("--load", type=str, help="Czy załadować z pliku")
    parser.add_argument("--train", type=lambda s: s.split(","),
                         help="Lista argumentów do trenignu: " \
                         "nazwa pliku do czytania zbioru treningowego" \
                         "ilość epok" \
                         "learning rate" \
                         "momentum" \
                         "czy prezentować wzorce w losowej kolejności tak/nie" \
                         "nazwa pliku gdzie będą zapisane wartości MSE w kolejnych epokach" )
    parser.add_argument("--error", type=float, help="czy podczas uczenia ustawić limit błędu")
    parser.add_argument("--test", type=lambda s: s.split(","),
                        help="Lista argumentów do testowania sieci:" \
                        "nazwa pliku do wczytania zbioru testowego" \
                        "nazwa pliku do zapisania logów testów")
    parser.add_argument("--save", type=str, help="nazwa pliku do zapisania sieci")
    args = parser.parse_args()

    if (args.shape or args.bias) and args.load:
        parser.error("Można albo stworzyć nową albo załadowac z pliku sieć")
    elif args.shape:
        mlp = MLP(args.shape,args.bias)
    elif args.load:
        mlp = MLP.load(args.load)
    else:
        parser.error("Stwórz/Wczytaj sieć")
    
    if args.train:
        
        X,Y = MLP.load_dataset(args.train[0])  
        mlp.train(X,Y,int(args.train[1]),float(args.train[2]),float(args.train[3]),args.train[4].strip().lower() == 'tak',args.error,args.train[5])
        plot_loss_from_file(args.train[5])
    if args.test:
        X,Y = MLP.load_dataset(args.test[0])
        test_data = mlp.test(X,Y)
        mlp.save_log_dict(test_data,args.test[1])
    if args.save:
        mlp.save(args.save)

if __name__ == '__main__':
    main()

