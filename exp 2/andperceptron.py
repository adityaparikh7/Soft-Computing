import numpy as np
from tabulate import tabulate

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# AND logic gate data (input and target)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# User input for the number of neurons in the hidden layer
hidden_layer_size = int(input("Enter the number of neurons in the hidden layer: "))

# Set the random seed for reproducibility
np.random.seed(1)

# Initialize the weights randomly with mean 0
input_layer_size = 2
output_layer_size = 1

# Weights and biases for the first layer (input layer to hidden layer)
synapse_0 = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
bias_0 = np.zeros((1, hidden_layer_size))

# Weights and biases for the second layer (hidden layer to output layer)
synapse_1 = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1
bias_1 = np.zeros((1, output_layer_size))

# Training the Neural Network
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0) + bias_0)
    layer_2 = sigmoid(np.dot(layer_1, synapse_1) + bias_1)

    # Calculate the error (difference between predicted and target)
    layer_2_error = y - layer_2

    # Backpropagation
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights and biases
    synapse_1_update = learning_rate * layer_1.T.dot(layer_2_delta)
    synapse_0_update = learning_rate * layer_0.T.dot(layer_1_delta)
    bias_1_update = learning_rate * np.sum(layer_2_delta, axis=0, keepdims=True)
    bias_0_update = learning_rate * np.sum(layer_1_delta, axis=0, keepdims=True)

    synapse_1 += synapse_1_update
    synapse_0 += synapse_0_update
    bias_1 += bias_1_update
    bias_0 += bias_0_update

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}")
        print("Output after forward propagation:")
        table_data = [["Input 1", "Input 2", "Output"]]
        for i in range(len(X)):
            row = np.append(X[i], layer_2[i])
            table_data.append(row)
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

        print("Weights and Biases:")
        print("Synapse_0 (weights 0 to hidden layer):")
        print(tabulate(synapse_0, headers=["Weight 1", "Weight 2"], tablefmt="grid"))
        print("Bias_0 (biases at hidden layer):")
        print(tabulate(bias_0, headers=["Bias"], tablefmt="grid"))
        print("Synapse_1 (weights at hidden layer to output):")
        print(tabulate(synapse_1, headers=["Weight"], tablefmt="grid"))
        print("Bias_1 (bias at output layer):")
        print(tabulate(bias_1, headers=["Bias"], tablefmt="grid"))

# Testing the Neural Network
print("Final output after training:")
table_data = [["Input 1", "Input 2", "Output"]]
for i in range(len(X)):
    row = np.append(X[i], layer_2[i])
    table_data.append(row)
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

# Rounding the output to 0 or 1 to match the AND gate output
rounded_output = np.round(layer_2)
print("Rounded output:")
table_data = [["Input 1", "Input 2", "Rounded Output"]]
for i in range(len(X)):
    row = np.append(X[i], rounded_output[i])
    table_data.append(row)
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
