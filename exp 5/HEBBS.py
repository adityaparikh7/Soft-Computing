import numpy as np

def signum(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else:
        return -1
    

# Define the number of neurons and the learning rate
num_neurons = 3
learning_rate = 0.1

# Initialize the weight matrix with zeros
weights = np.zeros((num_neurons, num_neurons))

# Define a list of input patterns (binary vectors)
input_patterns = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]

# Iterate through input patterns and update weights using Hebb's rule
for pattern in input_patterns:
    pattern = np.array(pattern)
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j:
                weights[i][j] += learning_rate * signum(pattern[i]) * signum(pattern[j])

# Print the learned weight matrix
print("Learned Weight Matrix:")
print(weights)
