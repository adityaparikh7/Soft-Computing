import numpy as np

def bipolar_sign(y):
    return 1 if y >= 0 else -1

def continuous_bipolar(y):
    return 2 / (1 + np.exp(-y)) - 1

W = np.array([1, -1])
X1 = np.array([1, -2])
X2 = np.array([2, 3])
X3 = np.array([1, -1])
C = 1


for X in [X1, X2, X3]:
    Y = bipolar_sign(np.dot(W, X))
    w_old = X * C
    W = W + w_old

print("Weights after one iteration (Bipolar Binary):", W)

W = np.array([1, -1])

for X in [X1, X2, X3]:
    Y = continuous_bipolar(np.dot(W, X))
    w_old = X * C
    W = W + w_old

print("Weights after one iteration (Continuous Bipolar):", W)
