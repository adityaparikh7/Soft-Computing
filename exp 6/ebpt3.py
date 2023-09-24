import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

x = np.array(([2,9],[1,5],[3,6]), dtype=float)
x = x/np.amax(x,axis=0)
y = np.array(([92],[86],[89]), dtype=float)
y = y/100

epochs = 10000
learning_rate = 0.1

input_layer_neurons = 2
hidden_layer_neurons = 3
output_neurons = 1

np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

for epoch in range(epochs):
    hidden_layer_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    error = y - output
    slope_output_layer = sigmoid_derivative(output)
    slope_hidden_layer = sigmoid_derivative(hidden_layer_output)
    delta_output = error * slope_output_layer
    error_hidden_layer = delta_output.dot(weights_hidden_output.T)
    delta_hidden_layer = error_hidden_layer * slope_hidden_layer

    weights_hidden_output += hidden_layer_output.T.dot(delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += x.T.dot(delta_hidden_layer) * learning_rate
    bias_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True) * learning_rate

print("Input: \n" + str(x))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)