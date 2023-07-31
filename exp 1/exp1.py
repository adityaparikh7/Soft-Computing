# import numpy as np

# def threshold(x, threshold_value=0):
#     return 1 if x >= threshold_value else 0

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def relu(x):
#     return max(0, x)

# def main():
#     print("Choose an activation function:")
#     print("1. Threshold")
#     print("2. Sigmoid")
#     print("3. ReLU")

#     choice = int(input("Enter the number corresponding to the activation function: "))

#     if choice == 1:
#         threshold_value = float(input("Enter the threshold value: "))
#         x = float(input("Enter the input value: "))
#         result = threshold(x, threshold_value)
#     elif choice == 2:
#         x = float(input("Enter the input value: "))
#         result = sigmoid(x)
#     elif choice == 3:
#         x = float(input("Enter the input value: "))
#         result = relu(x)
#     else:
#         print("Invalid choice.")
#         return

#     print("Result:", result)

# if __name__ == "__main__":
#     main()


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
    print("4.Identity")

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
    else:
        print("Invalid choice.")
        return

    plot_activation_function(func, func_name)

if __name__ == "__main__":
    main()
