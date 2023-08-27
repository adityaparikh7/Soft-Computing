import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0
    
    def activation(self,x):
        return 1 if x >= 0 else -1
    
    def predict(self,inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)
    
    def train(self, inputs, targets, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            error = False
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                delta = targets[i] - prediction
                if delta != 0:
                    self.weights += learning_rate * delta * inputs[i]
                    self.bias += learning_rate * delta
                    error = True

            print(f"Epoch {epoch + 1}:")
            print(f"Weights: {self.weights}, Bias: {self.bias}")
            if not error:
                print("Converged")
                break


input_data = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
and_targets = np.array([-1, -1, -1, 1])
or_targets = np.array([-1, 1, 1, 1])
nor_targets = np.array([1, -1, -1, -1])
nand_targets = np.array([1, 1, 1, -1])

and_perceptron = Perceptron(input_size=2)
or_perceptron = Perceptron(input_size=2)
nor_perceptron = Perceptron(input_size=2)
nand_perceptron = Perceptron(input_size=2)

print("Training AND gate:")
and_perceptron.train(input_data, and_targets)
print("\nTraining OR gate:")
or_perceptron.train(input_data, or_targets)
print("\nTraining NOR gate:")
nor_perceptron.train(input_data, nor_targets)
print("\nTraining NAND gate:")
nand_perceptron.train(input_data, nand_targets)
