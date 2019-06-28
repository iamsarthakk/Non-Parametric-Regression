import numpy as np
import matplotlib.pyplot as plt
import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods
from pyqt_fit import plot_fit



#Define our Kernel Functions
#1.Gaussian Kernel
def Kernel_Gauss(x):
    return 1/np.sqrt(2*np.pi)*pow(np.e,-x*x/2)

#2.Epanechnikov kernel
def Kernel_Epanech(x):
    if(abs(x)<=1):
        return 3/(4*(1-x*x))
    else:
        return 0

#Nadaray-Watson kernel regression
#n_nearest are the n nearest neighbours of x whose values we know from data
def hypothesis(x, n_nearest, data):
    num = 0
    den = 0
    h = 0.9 #Bandwidth, we take it as low as our data allow with descennt tradeoff b/w biase and variance
    for i in range(n_nearest):
        num += Kernel_Gauss((x-n_nearest[i])/h)*data[i]
        den += Kernel_Gauss((x-n_nearest[i])/h)
    h = num/den
    return h
