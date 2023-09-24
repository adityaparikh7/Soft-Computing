import numpy as np
import matplotlib.pyplot as plt

# Define the universe of discourse (X-axis range)
x = np.linspace(0, 10, 100)

# Define two fuzzy sets as membership functions
# Let's assume two triangular fuzzy sets
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# Fuzzy Set A
a1, b1, c1 = 2, 4, 6
fuzzy_set_A = triangular(x, a1, b1, c1)

# Fuzzy Set B
a2, b2, c2 = 3, 5, 7
fuzzy_set_B = triangular(x, a2, b2, c2)

# Visualize Fuzzy Sets A and B
plt.figure(figsize=(8, 4))
plt.plot(x, fuzzy_set_A, label='Fuzzy Set A')
plt.plot(x, fuzzy_set_B, label='Fuzzy Set B')
plt.xlabel('X-axis')
plt.ylabel('Membership Value')
plt.title('Fuzzy Sets A and B')
plt.legend()
plt.grid(True)
plt.show()

# Fuzzy Union
fuzzy_union = np.maximum(fuzzy_set_A, fuzzy_set_B)

# Visualize Fuzzy Union
plt.figure(figsize=(8, 4))
plt.plot(x, fuzzy_union, label='Fuzzy Union (A ∪ B)')
plt.xlabel('X-axis')
plt.ylabel('Membership Value')
plt.title('Fuzzy Union of A and B')
plt.legend()
plt.grid(True)
plt.show()

# Fuzzy Intersection
fuzzy_intersection = np.minimum(fuzzy_set_A, fuzzy_set_B)

# Visualize Fuzzy Intersection
plt.figure(figsize=(8, 4))
plt.plot(x, fuzzy_intersection, label='Fuzzy Intersection (A ∩ B)')
plt.xlabel('X-axis')
plt.ylabel('Membership Value')
plt.title('Fuzzy Intersection of A and B')
plt.legend()
plt.grid(True)
plt.show()

# Fuzzy Complement (of A)
fuzzy_complement_A = 1 - fuzzy_set_A

# Visualize Fuzzy Complement of A
plt.figure(figsize=(8, 4))
plt.plot(x, fuzzy_complement_A, label='Fuzzy Complement of A')
plt.xlabel('X-axis')
plt.ylabel('Membership Value')
plt.title('Fuzzy Complement of A')
plt.legend()
plt.grid(True)
plt.show()
