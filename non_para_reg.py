import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from math import exp,sqrt,pi

def f(x):
    return 3*np.cos(x/2) + x**2/5 + 3

def fit(test_X, train_X, train_y, bandwidth=1.0, kn='box'):
    kernels = {
        'box': lambda x: 1/2 if (x<=1 and x>=-1) else 0,
        'gs': lambda x: 1/sqrt(2*pi)*exp(-x**2/2),
        'ep': lambda x: 3/4*(1-x**2) if (x<=1 and x>=-1) else 0
    }
    predict_y = []
    for entry in test_X:
        nks = [np.sum((j-entry)**2)/bandwidth for j in train_X]
        ks = [kernels['box'](i) for i in nks]
        dividend = sum([ks[i]*train_y[i] for i in range(len(ks))])
        divisor = sum(ks)
        predict = dividend/divisor
        predict_y.extend(predict)
        # print(entry)
    return np.array(predict_y)

plt.style.use('ggplot')

a = np.linspace(0, 9.9, 200)

train_a = a[:,np.newaxis]
# noise = np.random.normal(0, 0.5, 200)
# e = noise[:,np.newaxis]
b = f(train_a) + 2*np.random.randn(*train_a.shape)
test_a = np.linspace(1,4.8,20)
formed_a = test_a[:,np.newaxis]

pred_b = fit(train_a,train_a,b,0.3,'gs')
plt.scatter(train_a,b,color='black')


plt.scatter(train_a, pred_b, color='red', linewidth = 0.1)

train_a.size, b.size

pred_b = fit(train_a,train_a,b,0.3,'gs')
plt.scatter(train_a,b,color='black')


plt.scatter(train_a, pred_b, color='red', linewidth = 0.1)


# #Define our Kernel Functions
# #1.Gaussian Kernel
# def Kernel_Gauss(x):
#     return 1/np.sqrt(2*np.pi)*pow(np.e,-x*x/2)
#
# #2.Epanechnikov kernel
# def Kernel_Epanech(x):
#     if(abs(x)<=1):
#         return 3/(4*(1-x*x))
#     else:
#         return 0
#
# #Nadaray-Watson kernel regression
# #n_nearest are the n nearest neighbours of x whose values we know from data
# def hypothesis(x, n_nearest, data):
#     num = 0
#     den = 0
#     h = 0.9 #Bandwidth, we take it as low as our data allow with descennt tradeoff b/w biase and variance
#     for i in range(n_nearest):
#         num += Kernel_Gauss((x-n_nearest[i])/h)*data[i]
#         den += Kernel_Gauss((x-n_nearest[i])/h)
#     h = num/den
#     return h
#
# def nonparam_regression(data):
#     X_train = data['0.03428'] #the feature_set
#     y_train = data['96.701'] #the labels
#     x_array = np.ones((X_train.shape[0], n+1))
#     for i in range(0,X_train.shape[0]):
#         for j in range(0,n+1):
#             x_array[i][j] = pow(X_train[i],j)
#
#     W = np.ones((X_train.shape[0], X_train.shape[0])
#     for i in range(0,X_train.shape[0]):
#         for j in range(0, X_train.shape[0]):
#             if i==j:
#                 W[i,j] = Kernel_Gauss()
