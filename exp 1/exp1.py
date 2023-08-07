import numpy as np
import matplotlib.pyplot as plt

def identity(x):
    return x

def threshold(x, threshold_value=0):
    return 1 if x >= threshold_value else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

def binary_step(x):
    return np.where(x < 0, 0, 1)

def plot_activation_function(func, func_name):
    x = np.linspace(-5, 5, 100)
    y = [func(i) for i in x]

    plt.plot(x, y)
    plt.title(func_name + " Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

def main():
    print("Choose an activation function:")
    print("1. Threshold")
    print("2. Sigmoid")
    print("3. ReLU")
    print("4. Identity")
    print("5. Tanh")
    print("6. Softmax")
    print("7. Bipolar Sigmoid")
    print("8. Binary Step")

    choice = int(input("Enter the number corresponding to the activation function: "))

    if choice == 1:
        threshold_value = float(input("Enter the threshold value: "))
        func = lambda x: threshold(x, threshold_value)
        func_name = "Threshold"
    elif choice == 2:
        func = sigmoid
        func_name = "Sigmoid"
    elif choice == 3:
        func = relu
        func_name = "ReLU"
    elif choice == 4:
        func = identity
        func_name = "Identity"
    elif choice == 5:
        func = tanh
        func_name = "Tanh"
    elif choice == 6:
        func = softmax
        func_name = "Softmax"
    elif choice == 7:
        func = bipolar_sigmoid
        func_name = "Bipolar Sigmoid"
    elif choice == 8:
        func = binary_step
        func_name = "Binary Step"
    else:
        print("Invalid choice.")
        return

    plot_activation_function(func, func_name)

if __name__ == "__main__":
    main()
