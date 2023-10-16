# Implement the fuzzy membership functions to define the fuzzy set for the following course evaluation crisp set

# (Hint:  descriptors for fuzzy set can be bad, poor, average, good, very good, excellent)

# EX= Marks >= 90

# A = 80=< Marks <90

# B = 70=< Marks <80

# C = 60=< Marks <70

# D = 50=< Marks <60

# P = 35=< Marks <50

# F = Marks < 35

import numpy as np
import matplotlib.pyplot as plt

# Define the range of marks (universe of discourse)
x = np.arange(0, 101, 1)

# Define membership functions for each fuzzy set
def marks_bad(x):
    return np.where(x <= 20, 1, 0)

def marks_poor(x):
    return np.where((x >= 20) & (x <= 45), (x - 20) / (45 - 20), 0)

def marks_average(x):
    return np.where((x >= 30) & (x <= 60), (x - 30) / (60 - 30), 0)

def marks_good(x):
    return np.where((x >= 50) & (x <= 85), (x - 50) / (85 - 50), 0)

def marks_very_good(x):
    return np.where((x >= 75) & (x <= 100), (x - 75) / (100 - 75), 0)

def marks_excellent(x):
    return np.where(x >= 95, 1, 0)

# Visualize the membership functions for each fuzzy set
plt.figure(figsize=(10, 6))
plt.plot(x, marks_bad(x), label='Bad')
plt.plot(x, marks_poor(x), label='Poor')
plt.plot(x, marks_average(x), label='Average')
plt.plot(x, marks_good(x), label='Good')
plt.plot(x, marks_very_good(x), label='Very Good')
plt.plot(x, marks_excellent(x), label='Excellent')

plt.xlabel('Marks')
plt.ylabel('Membership Degree')
plt.title('Fuzzy Membership Functions for Course Evaluation')
plt.legend()
plt.grid(True)
plt.show()

