import numpy as np

def signum(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

# Define the number of neurons and the learning rate
num_neurons = 4  # Update to match the size of your input patterns
learning_rate = 0.1

# Initialize the weight vector with zeros
weights = np.array([1, -1, 0, 0.5])

# Define a list of input patterns (binary vectors)
input_patterns = [
    np.array([1, -2, 1.5, 0]),
    np.array([1, -0.5, -2, -1.5]),
    np.array([0, 1, -1, 1.5])
]

# Iterate through input patterns and update weights using Hebb's rule
for pattern in input_patterns:
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j:
                weights[i] += learning_rate * signum(pattern[i]) * signum(pattern[j])

# Print the learned weight vector
print("Learned Weight Vector:")
print(weights)
