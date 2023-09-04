import numpy as np

def Hebbian(x,w):
    v = np.dot(w,x)
    return float(v)

def sgn(y):
    if(y>=0):
        return 1
    else:
        return -1

def sigm(y):
    return 2/(1+np.exp(-y))-1

def perceptronLearning(x,w,desired_output):
    oldw = w
    print("Using bipolar: ")
    for i in range(len(x)):
        y = float(Hebbian(x[i],w))
        if(y != desired_output[i]):
            w = w + sgn(y)*x[i]
        print("Weight in step",(i+1),":",w)
    print("\nUsing continous bipolar: ")
    for i in range(len(x)):
        y = float(Hebbian(x[i],oldw))
        if(y != desired_output[i]):
            oldw = oldw + sigm(y)*x[i]
        print("Weight in step",(i+1),":",oldw)

inputs = np.array([[1,-2,1.5,0],[1,-0.5,-2,-1.5],[0,1,-1,1.5]])
weights = np.array([1,-1,0,0.5])
desired_output = np.array([1,1,1,-1])
perceptronLearning(inputs,weights,desired_output)