import numpy as np

x1 = np.array([1, -2, 1.5, 0])
x2 = np.array([1, -0.5, -2, -1.5])
x3 = np.array([0, 1, -1, 1.5])
w1 = np.array([1, -1, 0, 0.5])

# Define the learning rate
learning_rate = 0.1

def signum(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def sig1(net):
    sgn = []
    for i in range(len(net)):
        sgn.append(signum(net[i]))
    sgn = np.array(sgn)
    return sgn

print("Initial Weights (w1):", w1)
print("\nTraining:")
for step in range(3):
    print(f"\nStep {step + 1}:")
    print(f"Input: x{step + 1}")
    
    # Calculate net value
    net = w1 * globals()[f'x{step + 1}']
    print(f"Net: {net}")

    # Update weights
    w1 = w1 + learning_rate * sig1(net) * globals()[f'x{step + 1}']
    print(f"Updated Weights: w{step + 2}:", w1)

print("\nFinal Weights:", w1)
