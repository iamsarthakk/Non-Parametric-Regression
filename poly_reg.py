import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv

#It is the function that calculates and outputs the hypothesis value of the Target Variable,
#given theta (theta_0, theta_1, theta_2, …., theta_n),
#Feature X and Degree of the Polynomial n as input
def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        x_array = np.ones(n+1)
        for j in range(0,n+1):
            x_array[j] = pow(X[i],j)
        x_array = x_array.reshape(n+1,1)
        h[i] = float(np.matmul(theta, x_array))
    h = h.reshape(X.shape[0])
    return h

data = np.loadtxt('data1.txt', delimiter=',')

#Polynomial Regression using normal equation
def poly_reg_normal(data):
    X_train = data[:,0] #the feature_set
    y_train = data[:,1] #the labels
    x_array = np.ones((X_train.shape[0], n+1))
    for i in range(0,X_train.shape[0]):
        for j in range(0,n+1):
            x_array[i][j] = pow(X_train[i],j)
    theta = np.matmul(np.matmul(inv(np.matmul(x_array.transpose()
                      , x_array)), x_array.transpose()), y_train)
    return theta;
