def hebbian_learning(W, X, Y, C, activation_function):
  W_new = []
  for i in range(len(W)):
    w_new = W[i] + C * X[i] * activation_function(Y[i])
    W_new.append(w_new)
  return W_new


def bipolar_activation(y):
  if y >= 0:
    return 1
  else:
    return -1


def continuous_bipolar_activation(y):
  """
  The continuous bipolar activation function.

  Args:
    y: The output of the neuron.

  Returns:
    tanh(y).
  """
  return tanh(y)


if __name__ == "__main__":
  # Initialize the weights.
  W = [1, -1]

  # The inputs.
  X = [[1, -2], [2, 3], [1, -1]]

  # The output vectors.
  Y = [1, -1, 1]

  # The learning rate.
  C = 1

  # The activation function.
  activation_function = bipolar_activation

  # Find the weights after one iteration.
  W_new = hebbian_learning(W, X, Y, C, activation_function)

  # Print the updated weights.
  print(W_new)


  # The activation function.
  activation_function = continuous_bipolar_activation

  # Find the weights after one iteration.
  W_new = hebbian_learning(W, X, Y, C, activation_function)

  # Print the updated weights.
  print(W_new)
