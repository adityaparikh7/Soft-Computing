import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize a simple feedforward neural network
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = int(input("Enter the number of epochs: "))

# Initialize random weights and biases
np.random.seed(0)
input_layer = np.random.rand(input_size, 1)
hidden_layer_weights = np.random.rand(hidden_size, input_size)
output_layer_weights = np.random.rand(output_size, hidden_size)
hidden_layer_bias = np.random.rand(hidden_size, 1)
output_layer_bias = np.random.rand(output_size, 1)

# Training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training loop
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        # Forward pass
        input_layer = X[i].reshape(input_size, 1)
        hidden_layer_input = np.dot(hidden_layer_weights, input_layer) + hidden_layer_bias
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(output_layer_weights, hidden_layer_output) + output_layer_bias
        output_layer_output = sigmoid(output_layer_input)

        # Compute the error
        error = y[i] - output_layer_output
        total_error += np.abs(error)

        # Backpropagation
        delta_output = error * sigmoid_derivative(output_layer_output)
        error_hidden_layer = np.dot(output_layer_weights.T, delta_output)
        delta_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        output_layer_weights += learning_rate * np.dot(delta_output, hidden_layer_output.T)
        output_layer_bias += learning_rate * delta_output
        hidden_layer_weights += learning_rate * np.dot(delta_hidden_layer, input_layer.T)
        hidden_layer_bias += learning_rate * delta_hidden_layer

    # Print the average error for this epoch
    average_error = total_error / len(X)
    print(f"Epoch {epoch + 1}/{epochs}, Average Error: {average_error[0]}")

# Display the final trained weights and biases
print("\nTrained Weights and Biases:")
print("Hidden Layer Weights:")
print(hidden_layer_weights)
print("Hidden Layer Bias:")
print(hidden_layer_bias)
print("Output Layer Weights:")
print(output_layer_weights)
print("Output Layer Bias:")
print(output_layer_bias) 