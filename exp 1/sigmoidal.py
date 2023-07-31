# design activation function for sigmoidal
import matplotlib.pyplot as plt
import numpy as np

def sigmoidal(x):
    return 1 / (1 + np.exp(-x))


def main():
    x = np.arange(-10, 10, 0.1)
    y = [sigmoidal(i) for i in x]
    print(y)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()


