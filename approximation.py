import numpy as np
import scipy
from scipy import sin,cos,exp
import scipy.linalg
In [183]:
fx = lambda x: sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0)
In [184]:
p1 = np.array([1,15])
b1 = fx(p1)
In [185]:
a1 = np.array([[p1[0]**n for n in xrange(0,2)], [p1[1]**n for n in xrange(0,2)]])
In [186]:
s1 = scipy.linalg.solve(a1,b1)
In [ ]:
 
In [187]:
b =  np.row_stack((p1[0], p1[1]))
In [ ]:
 
In [188]:
p3 = np.array([1,4,10,15])
In [189]:
b3 = fx(p3)
In [190]:
a3 = np.array([[p3[0]**n for n in xrange(0,4)], [p3[1]**n for n in xrange(0,4)], [p3[2]**n for n in xrange(0,4)], 
               [p3[3]**n for n in xrange(0,4)]])
