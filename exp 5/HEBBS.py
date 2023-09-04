import numpy as np

def Hebbian(x, w):
    v = np.dot(w, x)
    return v

def bipolar_sign(y):
    return 1 if y >= 0 else -1

def continuous_bipolar(y):
    return 2 / (1 + np.exp(-y)) - 1

def HebbianModel(x, w, desired_output):
    old_w = w
    print("Using bipolar:")
    for i in range(len(x)):
        y = Hebbian(x[i], w)
        if y != desired_output[i]:
            w = w + bipolar_sign(y) * x[i]
        print(f"Weight in step {i + 1}: {w}")
    
    print("\nUsing continuous bipolar:")
    for i in range(len(x)):
        y = Hebbian(x[i], old_w)
        if y != desired_output[i]:
            old_w = old_w + continuous_bipolar(y) * x[i]
        print(f"Weight in step {i + 1}: {old_w}")

if __name__ == "__main__":
    inputs = np.array([[1, -2, 1.5, 0], [1, -0.5, -2, -1.5], [0, 1, -1, 1.5]])
    weights = np.array([1, -1, 0, 0.5])
    desired_output = np.array([1, 1, 1, -1])
    
    HebbianModel(inputs, weights, desired_output)
