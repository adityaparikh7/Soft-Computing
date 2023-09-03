import numpy as np

# Initialize an empty list for training data
training_data = []

# Collect user input for vectors and desired outputs
for i in range(3):
    print(f"Enter data for vector x{i + 1}:")
    x = []
    for j in range(4):
        x_j = float(input(f"Enter x{i + 1}{j + 1}: "))
        x.append(x_j)
    d = float(input(f"Enter desired output d{i + 1}: "))
    training_data.append((np.array([1] + x), d))

# Initialize weight vector
weights = np.array([1, -1, 0, 0.5])

# Learning rate
learning_rate = 0.1

# Number of epochs
epochs = 100

# Delta learning rule
for epoch in range(epochs):
    total_error = 0
    for data_point, target in training_data:
        output = np.dot(weights, data_point)
        error = target - output
        weights += learning_rate * error * data_point
        total_error += error**2
    if total_error == 0:
        break

# Verify the trained weights
print("Trained Weights:", weights)

# Test the perceptron with the training data
print("\nTesting the trained perceptron:")
for data_point, _ in training_data:
    output = np.dot(weights, data_point)
    print("Input:", data_point[1:], "Output:", output)

# Verify if the perceptron learned correctly
print("\nDesired Outputs:")
for _, target in training_data:
    print(target)
