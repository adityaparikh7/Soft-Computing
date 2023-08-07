import numpy as np
import matplotlib.pyplot as plt

def identity(x):
    return x

def binary_step(x):
    return np.where(x < 0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

def ramp(x):
    return max(0, x)

def plot_activation_function(func, func_name):
    user_input = get_user_input()
    x = np.array(user_input)
    y = [func(i) for i in x]

    plt.plot(x, y)
    plt.title(func_name + " Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

def get_user_input():
    try:
        input_str = input("Enter input values separated by spaces: ")
        input_values = [float(val) for val in input_str.split()]
        return input_values
    except ValueError:
        print("Invalid input! Please enter valid numbers separated by spaces.")
        return get_user_input()

def main():
    print("Choose an activation function: ")
    print("1. Identity")
    print("2. Binary Step")
    print("3. Sigmoid")
    print("4. Bipolar Sigmoid")

    choice = int(input("Enter the number corresponding to the activation function: "))

    if choice == 1:
        func = identity
        func_name = "Identity"
    elif choice == 2:
        func = binary_step
        func_name = "Binary Step"
    elif choice == 3:
        func = sigmoid
        func_name = "Sigmoid"
    elif choice == 4:
        func = bipolar_sigmoid
        func_name = "Bipolar Sigmoid"
    else:
        print("Invalid choice.")
        return

    plot_activation_function(func, func_name)

if __name__ == "__main__":
    main()
