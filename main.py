from mlp import *

def main():
    neat = MLP((4,2,4), use_bias= False)
    X , Y = MLP.load_dataset("autoenkoder.txt")
   
    neat.train(X,Y,epochs=1000,learning_rate=0.1)
    cos, innecos = neat.forward(np.array([1,0,0,0]))
    print(f"wynik dla (1,0,0,0): {cos[-1]} ")
    cos, innecos = neat.forward(np.array([0,1,0,0]))
    print(f"wynik dla (0,1,0,0): {cos[-1]} ")
    cos, innecos = neat.forward(np.array([0,0,1,0]))
    print(f"wynik dla (0,0,1,0): {cos[-1]} ")
    cos, innecos = neat.forward(np.array([0,0,0,1]))
    print(f"wynik dla (0,0,0,1): {cos[-1]} ")
    

if __name__ == '__main__':
    main()

