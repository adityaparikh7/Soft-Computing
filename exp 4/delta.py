import numpy as np

def delta_learning(training_data, learning_rate, epochs):
    num_inputs = len(training_data[0]) - 1
    weights = np.random.rand(num_inputs)
    bias = np.random.rand()
    errors = []

    for epoch in range(epochs):
        total_error = 0
        for data_point in training_data:
            inputs = np.array(data_point[:-1])
            target = data_point[-1]
            output = np.dot(weights, inputs) + bias
            error = target - output
            total_error += error**2

            weights += learning_rate * error * inputs
            bias += learning_rate * error

        errors.append(total_error)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Error = {total_error}")

    return weights, bias, errors

def main():
    num_inputs = int(input("Enter the number of inputs: "))
    training_data = []
    num_samples = int(input("Enter the number of training samples: "))

    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        inputs = [float(input(f"Enter input {j + 1}: ")) for j in range(num_inputs)]
        target = float(input("Enter target output: "))
        training_data.append(inputs + [target])

    learning_rate = float(input("Enter the learning rate: "))
    epochs = int(input("Enter the number of epochs: "))

    weights, bias, errors = delta_learning(training_data, learning_rate, epochs)

    print("\nTraining complete. Final weights and bias:")
    print("Weights:", weights)
    print("Bias:", bias)

if __name__ == "__main__":
    main()
