import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the weights and bias for a simple XOR perceptron
# These values are set manually to mimic the XOR behavior
weights = np.array([1, 1])
bias = -1

# Calculate the output of the perceptron
outputs = sigmoid(np.dot(X, weights) + bias)

# Print the results
for i in range(len(X)):
    input_data = X[i]
    output = outputs[i]
    print(f"Input: {input_data}, Predicted Output: {output:.2f}")
