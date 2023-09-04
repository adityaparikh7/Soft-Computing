import numpy as np

x1 = np.array([1,-2,1.5,0])
x2 = np.array([1,-0.5,-2,-1.5])
x3 = np.array([0,1,-1,1.5])
w1 = np.array([1,-1,0,0.5])

def signum(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else:
        return -1
    
def sig1(net):
   if(net>=0):
        return 1
   else:
       return -1
   

print("Input 1: ", x1)
print("Weight 1: ",w1)

net1 = w1*x1
print("Net 1: ", net1)

w2 = w1 + sig1(net1)*x1
print("Input 2: ", x2)
print("Weight 2: ",w2)

net2 = w2*x2
print("Net 2: ", net2)

w3 = w2 + sig1(net2)*x2
print("Input 3: ", x3)
print("Weight 3: ",w3)

net3 = w3*x3
print("Net 3: ", net3)

w4 = w3 + sig1(net3)*x3
print("Weight 4: ",w4)
