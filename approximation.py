import numpy as np
import scipy
from scipy import sin,cos,exp
import scipy.linalg
from scipy import interpolate
from matplotlib import pylab as plt
fx = lambda x: sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0) #задачем функцию
p = np.array([1,15])
b = fx(p)
k=2
a = np.zeros((len(p), len(p)))  #создаем пустой массив нужной размерности
for i in range(0,k):            # заполняем массив
    a[i,:] = np.array([p[i]**n for n in xrange(0,k)]) # заполняем массив
s1 = scipy.linalg.solve(a,b)
p = np.array([1,4,8,15])
b = fx(p)
k=4
a = np.zeros((len(p), len(p)))
for i in range(0,k):
    a[i,:] = np.array([p[i]**n for n in xrange(0,k)])
s3 = scipy.linalg.solve(a,b)
f = interpolate.interp1d(p, fx(p), kind='quadratic')
xnew = np.arange(0, 15, 0.1)
plt.plot(p, b, 'x', p, b, '----')
plt.show()
p = np.array([1,8,15])
b = fx(p)
k=3
a = np.zeros((len(p), len(p)))
for i in range(0,k):
    a[i,:] = np.array([p[i]**n for n in xrange(0,k)])
s2 = scipy.linalg.solve(a,b)
