# design activation function for threshold 

import numpy as np
import matplotlib.pyplot as plt

def threshold(x):
    return 1 if x >= 0 else 0


def main():
    x = np.arange(-10, 10, 0.1)
    y = [threshold(i) for i in x]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()








