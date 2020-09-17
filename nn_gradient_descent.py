import numpy as np
from sklearn.preprocessing import scale

X = np.array([[1,4,6,3],[5,3,2,5],[2,5,3,-2],[4,5,2,-8]])
y = np.array([0,0,1,1]).reshape((-1,1))
X = scale(X)

def loss(true,predicts, deriv = False):
    if deriv:
        return -true/predicts + (1-true)/(1-predicts)
    return -1*(true *np.log(predicts) + (1-true)*np.log(1-predicts))

def sigmoid(x, deriv = False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

neurons = 64
weight_1 = np.random.normal(size = (4,neurons))
weight_2 = np.random.normal(size = (neurons,1))

bias_1 = np.mean(weight_1,axis=0)
bias_2 = np.mean(weight_2,axis=0)
for e in range(2000):
    alpha = 0.01

    layer_1 = sigmoid(np.dot(X,weight_1)+ bias_1)
    layer_2 = sigmoid(np.dot(layer_1,weight_2)+ bias_2)

    log_loss = np.mean(loss(y, layer_2))

    error_2_1 = loss(y,layer_2,True)
    error_2_2 = error_2_1*sigmoid(layer_2,True)

    error_1_1 = np.dot(error_2_2,weight_2.transpose())
    error_1_2 = error_1_1*sigmoid(layer_1,True)

    weight_2 -= alpha*np.dot(layer_1.transpose(),error_2_2)
    weight_1 -= alpha*np.dot(X.transpose(),error_1_2)

    bias_2 -= np.mean(error_2_2,axis = 0)
    bias_1 -= np.mean(error_1_2,axis = 0)

#     if  e % 1000 == 0:
#         print(log_loss)
