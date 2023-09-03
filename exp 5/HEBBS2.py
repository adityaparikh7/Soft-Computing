import numpy as np

# Get the number of neurons from the user
num_neurons = int(input("Enter the number of neurons: "))

# Get the learning rate from the user
learning_rate = float(input("Enter the learning rate: "))

# Initialize the weight matrix with zeros
weights = np.zeros((num_neurons, num_neurons))

# Get input patterns from the user
input_patterns = []
print("Enter the input patterns (each pattern as a space-separated binary vector):")
for i in range(num_neurons):
    pattern_str = input(f"Enter input pattern {i + 1}: ")
    pattern = [int(x) for x in pattern_str.split()]
    input_patterns.append(pattern)

# Iterate through input patterns and update weights using Hebb's rule
for pattern in input_patterns:
    pattern = np.array(pattern)
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j:
                weights[i][j] += learning_rate * pattern[i] * pattern[j]

# Print the learned weight matrix
print("\nLearned Weight Matrix:")
print(weights)
