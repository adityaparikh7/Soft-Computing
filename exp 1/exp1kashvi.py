# import numpy as np

# def identity(x):
#     return x

# def binary_step(x):
#     return np.where(x < 0, 0, 1)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def bipolar_sigmoid(x):
#     return (2 / (1 + np.exp(-x))) - 1

# def relu(x):
#     return np.maximum(0, x)

# # Function to get user input
# def get_user_input():
#     try:
#         input_str = input("Enter input values separated by spaces: ")
#         input_values = [float(val) for val in input_str.split()]
#         return input_values
#     except ValueError:
#         print("Invalid input! Please enter valid numbers separated by spaces.")
#         return get_user_input()

# # Test the activation functions with user input
# if __name__ == "__main__":
#     user_input = get_user_input()
#     x = np.array(user_input)

#     print("Identity function:")
#     print(identity(x))

#     print("Binary Step function:")
#     binary_step_output = binary_step(x)
#     print(binary_step_output)

#     print("Sigmoid function:")
#     print(sigmoid(x))

#     print("Bipolar Sigmoid function:")
#     print(bipolar_sigmoid(x))

#     print("ReLU function:")
#     print(relu(x))

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

def relu(x):
    return np.maximum(0, x)

# Function to get user input
def get_user_input():
    try:
        input_str = input("Enter input values separated by spaces: ")
        input_values = [float(val) for val in input_str.split()]
        return input_values
    except ValueError:
        print("Invalid input! Please enter valid numbers separated by spaces.")
        return get_user_input()

# Test the activation functions with user input
if __name__ == "__main__":
    user_input = get_user_input()
    x = np.array(user_input)

    print("Identity function:")
    print(identity(x))

    print("Binary Step function:")
    binary_step_output = binary_step(x)
    print(binary_step_output)

    print("Sigmoid function:")
    print(sigmoid(x))

    print("Bipolar Sigmoid function:")
    print(bipolar_sigmoid(x))

    print("ReLU function:")
    print(relu(x))

    # Plotting the activation functions
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, identity(x))
    plt.title('Identity Function')

    plt.subplot(2, 2, 2)
    plt.plot(x, binary_step_output)
    plt.title('Binary Step Function')

    plt.subplot(2, 2, 3)
    plt.plot(x, sigmoid(x))
    plt.title('Sigmoid Function')

    plt.subplot(2, 2, 4)
    plt.plot(x, bipolar_sigmoid(x))
    plt.title('Bipolar Sigmoid Function')

    plt.figure(figsize=(8, 6))
    plt.plot(x, relu(x))
    plt.title('ReLU Function')

    plt.tight_layout()
    plt.show()

    # ReLU is a special case and doesn't require separate subplot
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, relu(x))
    # plt.title('ReLU Function')
    # plt.show()
