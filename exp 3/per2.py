import numpy as np

# Define the bipolar activation function
def bipolar_activation(x):
    return 1 if x >= 0 else -1

# Define the perceptron training algorithm
def perceptron_train(inputs, targets, learning_rate=0.1, epochs=100):
    num_samples, num_features = inputs.shape
    weights = np.random.rand(num_features)
    bias = np.random.rand()

    for epoch in range(epochs):
        errors = 0
        for i in range(num_samples):
            net_input = np.dot(inputs[i], weights) + bias
            output = bipolar_activation(net_input)
            error = targets[i] - output
            weights += learning_rate * error * inputs[i]
            bias += learning_rate * error
            errors += int(error != 0)

        print(f"Epoch {epoch + 1}: Errors = {errors}")

        # If all examples are correctly classified, stop training
        if errors == 0:
            break

    return weights, bias

# Define the logic functions
logic_functions = {
    "AND": (np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), np.array([-1, -1, -1, 1])),
    "OR": (np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), np.array([-1, 1, 1, 1])),
    "NOR": (np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), np.array([1, -1, -1, -1])),
    "NAND": (np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), np.array([1, 1, 1, -1]))
}

# User input to select a logic function
print("Select a logic function: ")
print("1. AND")
print("2. OR")
print("3. NOR")
print("4. NAND")
choice = int(input("Enter your choice (1/2/3/4): "))

if choice in [1, 2, 3, 4]:
    function_name = list(logic_functions.keys())[choice - 1]
    inputs, targets = logic_functions[function_name]

    weights, bias = perceptron_train(inputs, targets)

    print(f"\nTraining completed. Weights: {weights}, Bias: {bias}")

    while True:
        test_input = input("Enter test input (e.g., -1 -1): ")
        test_input = np.array([int(x) for x in test_input.split()])
        net_input = np.dot(test_input, weights) + bias
        output = bipolar_activation(net_input)
        print(f"Output for input {test_input}: {output}")
else:
    print("Invalid choice. Please select 1, 2, 3, or 4.")
