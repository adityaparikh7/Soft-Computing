import numpy as np
import math

def bipolarbinsig(x):
    return ((2 * (1 + math.exp(-x))**-1) - 1)

x1 = np.array([1, -2, 0, -1])
x2 = np.array([0, 1.5, -0.5, -1])
x3 = np.array([-1, 1, 0.5, -1])

w1 = np.array([1, -1, 0, 0.5])

learning_rate = float(input("Enter the learning rate: "))

d1 = -1
d2 = -1
d3 = 1

epochs = 1000  
for epoch in range(epochs):
    net1 = np.dot(x1, w1)
    o1 = bipolarbinsig(net1)

    error1 = 0.5 * (d1 - o1)**2

    der1 = 0.5 * (1 - o1**2)

    delta_w1 = learning_rate * (d1 - o1) * der1 * x1
    w1 += delta_w1

    net2 = np.dot(x2, w1)
    o2 = bipolarbinsig(net2)
    error2 = 0.5 * (d2 - o2)**2
    der2 = 0.5 * (1 - o2**2)
    delta_w2 = learning_rate * (d2 - o2) * der2 * x2
    w1 += delta_w2

    net3 = np.dot(x3, w1)
    o3 = bipolarbinsig(net3)
    error3 = 0.5 * (d3 - o3)**2
    der3 = 0.5 * (1 - o3**2)
    delta_w3 = learning_rate * (d3 - o3) * der3 * x3
    w1 += delta_w3

    total_error = error1 + error2 + error3

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Total Error = {total_error}")

print("Final Weights:", w1)
