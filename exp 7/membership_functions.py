import numpy as np
import matplotlib.pyplot as plt

# Define a range of x values
x = np.linspace(0, 10, 100)

# a. Triangular
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

a, b, c = 2, 4, 6
triangular_membership = triangular(x, a, b, c)
plt.plot(x, triangular_membership, label='Triangular')
plt.title('Triangular Membership Function')


# b. Trapezoidal
def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))

a, b, c, d = 2, 4, 6, 8
trapezoidal_membership = trapezoidal(x, a, b, c, d)
plt.plot(x, trapezoidal_membership, label='Trapezoidal')
plt.title('Trapezoidal Membership Function')


# c. Gaussian
def gaussian(x, mean, std_dev):
    return np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

mean, std_dev = 5, 1
gaussian_membership = gaussian(x, mean, std_dev)
plt.plot(x, gaussian_membership, label='Gaussian')
plt.title('Gaussian Membership Function')


# d. Generalized (S-shaped)
def generalized(x, a, b):
    return 1 / (1 + np.abs((x - a) / b) ** (2 * b))

a, b = 5, 2
generalized_membership = generalized(x, a, b)
plt.plot(x, generalized_membership, label='Generalized')
plt.title('Generalized Membership Function')


# e. Sigmoid
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

a, b = 1, 5
sigmoid_membership = sigmoid(x, a, b)
plt.plot(x, sigmoid_membership, label='Sigmoid')
plt.title('Sigmoid Membership Function')
plt.legend()
plt.show()
