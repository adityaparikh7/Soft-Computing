import numpy as np
import math

def bipolarBinarySigmoid(x):
    return ((2*(1+math.exp(-x))**-1)-1)

def errorPortion(target,dfyin,fyin):
    return (target+fyin)*dfyin

def backPropogation(inputs,target,lr,b1,b2,w0,v1,v2,w):
    zin1 = np.dot(inputs,v1) + b1
    zin2 = np.dot(inputs,v2) + b2
    print("zin1 :- ", zin1)
    print("zin2 :- ", zin2)
    z1 = bipolarBinarySigmoid(zin1)
    z2 = bipolarBinarySigmoid(zin2)
    print("z1 :- ", z1)
    print("z2 :- ", z2)
    z = [z1,z2]
    z = np.array(z)
    yin = np.dot(z,w) + w0
    print("yin :-", yin)
    fyin = bipolarBinarySigmoid(yin)
    print("fyin :-", fyin)
    dfyin = 0.5*(1+fyin)*(1-fyin)
    print("f'yin :-", dfyin)
    del1 = 0.5*(1+z1)*(1-z1)
    del2 = 0.5*(1+z2)*(1-z2)
    print("del1 :-", del1)
    print("del2 :-", del2)
    delta1 = errorPortion(target,dfyin,fyin)
    delta_weight = [lr*delta1*z1,lr*delta1*z2]
    delta_weight = np.array(delta_weight)
    print("delta w :- ",delta_weight)
    delta_bias = lr*delta1
    delv1 = lr*del1*inputs
    print("delv :- ",delv1)
    bias_v1 = lr*del1
    print("bias v1 :- ",bias_v1)
    delv2 = lr*del2*inputs
    print("delv :- ",delv2)
    bias_v2 = lr*del2
    print("bias v2 :- ",bias_v2)
    v1_new = v1 + delv1
    print("v1 new :- ",v1_new)
    v2_new = v2 + delv2
    print("v1 new :- ",v2_new)
    delv1_new = bias_v1 + b1
    delv2_new = bias_v2 + b2
    print("New Bias v1 :- ",delv1_new)
    print("New Bias v1 :- ",delv2_new)
    w_new = delta_weight + w
    w_bias_new = delta_bias + w0
    print("W(new) :- ",w_new)
    print("W bias (new) :- ",w_bias_new)
    
inputs = [1,-1]
inputs = np.array(inputs)
target = 1
lr = 0.1
b1 = 1
b2 = 0.5
w0 = -0.2
v1 = [0.6,0.1]
v2 = [-0.3,0.4]
v1 = np.array(v1)
v2 = np.array(v2)
w = [0.4,0.1]
w = np.array(w)
backPropogation(inputs,target,lr,b1,b2,w0,v1,v2,w)