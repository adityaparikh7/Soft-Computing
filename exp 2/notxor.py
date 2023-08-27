import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def perceptron(x, w, b):
  return sigmoid(np.dot(x, w) + b)

def train(x, y, epochs):
  w = np.random.rand(3)
  b = np.random.rand()

  for epoch in range(epochs):
    for i in range(len(x)):
      y_pred = perceptron(x[i], w, b)
      error = y[i] - y_pred
      w += x[i] * error
      b += error

  return w, b

def main():
  x = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
  y = np.array([0, 1, 1, 0])

  w, b = train(x, y, 1000)

  for i in range(len(x)):
    y_pred = perceptron(x[i], w, b)
    print(f"Input = {x[i]}, Expected Output = {y[i]}, Predicted Output = {round(y_pred, 2)}")

if __name__ == "__main__":
  main()
