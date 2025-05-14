from mlp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.title('MSE w czasie epok')
    plt.xlabel('Epoka')
    plt.ylabel('MSE (Błąd średniokwadratowy)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def print_menu():
        print("\n=== MENU ===")
        print("1. Stwórz nową sieć")
        print("2. Wczytaj sieć z pliku")
        print("3. Wyjście ")
        print("==============")

def use_menu(mlp:MLP):
    while True:
        print("\n=== MENU ===")
        print("1. Tryb nauki ")
        print("2. Tryb testowania ")
        print("3. Zapisz sieć")
        print("4. Wyświetl wykres z pliku otrzymanego z nauki sieci")
        print("5. Wyjście ")
        print("==============")
        choice = input("Wybierz opcję: ").strip()

        if choice == '1':
            mlp.training_mode()
        elif choice == '2':
            mlp.testing_mode()
        elif choice == '3':
            filename = input("Podaj nazwe pliku do zapisu sieci... ")
            mlp.save(filename)
        elif choice =='4':
            filename = input("Podaj nazwe pliku do wyświetlenia ")
            plot_loss_from_file(filename)
        elif choice == '5':
            print("Zakończono program.")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")
    

def main():

    while True:
        print_menu()
        choice = input("Wybierz opcję: ").strip()

        if choice == '1':
            layers:int = int(input("Ile warstw ma mieć sieć? ").strip())
            neurons = []
            for i in range(layers):
                count:int = int(input(f"Podaj liczbe neuronów w warstwie nr {i+1} (1 to wejściowa) "))
                neurons.append(count)
            print("Czy zaanicjować Biasy?")
            print("1. Tak")
            print("2. Nie")
            bias =int( input(f"Wybierz... "))
            if bias == 1:
                bias = True
            elif bias == 2:
                bias = False
            else:
                print("zly wybor")
            mlp = MLP(tuple(neurons),bias)
            use_menu(mlp)

        elif choice == '2':
            filename =  input("Podaj nazwe pliku ")
            mlp = MLP.load(filename)
            use_menu(mlp)
        elif choice == '3':
            print("Zakończono program.")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")




if __name__ == '__main__':
    main()

