import numpy as np
import scipy
from scipy import sin,cos,exp
import scipy.linalg
from scipy import interpolate
from matplotlib import pylab as plt
from scipy import optimize
fx = lambda x: sin(x / 5.0) * exp(x / 10.0) + 5.0 * exp(-x / 2.0) #задаем функцию
apx = scipy.optimize.minimize (fx,30,method="BFGS") # минимищируем с приближением 30 можно сравнить с рпиближением 2
x = np.arange(0, 30, 0.1)  #стриом функцию
plt.plot(x, fx(x)) #выводим график
plt.show() #выводим
