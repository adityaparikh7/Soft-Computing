import numpy as np

def HebbianLearning(w,x_list,y_list):
    for x, y in zip(x_list, y_list):
        dw = x * y
        w = w + dw

    return w

def main():
    w = np.array([1, -1, 0, 0.5])

    x_list = [np.array([1, -2, 1.5, 0]), np.array([1, -0.5, -2, -1.5]), np.array([0, 1, -1, 1.5])]
    y_list = [np.array([0,1.25,3,-1])]
    
    w = HebbianLearning(w, x_list, y_list)
    print("Final Weights:")
    print(w)

if __name__ == "__main__":
    main()