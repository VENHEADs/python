import numpy as np
import scipy
from scipy import sin,cos,exp
import scipy.linalg
from scipy import interpolate
from matplotlib import pylab as plt
from scipy import optimize
fx = lambda x: sin(x / 5.0) * exp(x / 10.0) + 5.0 * exp(-x / 2.0) #задаем функцию
apx = scipy.optimize.minimize (fx,30,method="BFGS") # минимищируем с приближением 30 можно сравнить с приближеением 2
x = np.arange(0, 30, 0.1)  #стриом функцию
plt.plot(x, fx(x)) #выводим график
plt.show() #выводим
bounds = [(1,30)]
scipy.optimize.differential_evolution(fx,bounds) # алгоритм дифференциальной эволюции
def h1(x):
    return int(fx(x))  # создаем функцию новую
hx = np.vectorize(h1)  # делаем ее активной для использования с помощью более чем 1 переменной

hx1 = np.vectorize(lambda x: int(sin(x / 5.0) * exp(x / 10.0) + 5.0 * exp(-x / 2.0)))  #вариант 3
fx(x).astype(int)  #вариант 3
def z(x): #вариант 4
    return fx(x).astype(int)
