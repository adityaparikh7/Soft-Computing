import numpy as np
import math

def bipolarbinsig(x):
    return ((2*(1+math.exp(-x))**-1)-1)

x1 = np.array([1,-2,0,-1])
x2 = np.array([0,1.5,-0.5,-1])
x3 = np.array([-1,1,0.5,-1])
w1 = np.array([1,-1,0,0.5])

net1 = np.dot(x1,w1)
print("Net value 1: ",net1)
# print(net)
o = bipolarbinsig(net1)
print(o)
der = 0.5*(1-(o*o))
w2 = 0.1*(1-o)*der*x1 + w1
print(w2)

net2 = np.dot(x2,w2)
print("Net value 2: ",net2)
o2 = bipolarbinsig(net2)
der2 = 0.5*(1-(o2*o2))

w3 = 0.1*(1-o2)*der2*x2 + w2
print(w3)

net3 = np.dot(x3,w3)
print("Net value 3: ",net3)
o3 = bipolarbinsig(net3)
der3 = 0.5*(1-(o3*o3))

w4 = 0.1*(1-o3)*der3*x3 + w3
print(w4)