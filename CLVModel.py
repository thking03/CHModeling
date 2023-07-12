"""
Competitive Lotka-Volterra Model of cell growth
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from OptModel import *

# define the carrying capacity
ktot = 30000

def LVnprime(y, t, r, Amat):
    vlen = len(r)
    dN = list(r*y*(np.ones(vlen)-np.matmul(Amat,y)/ktot))
    return dN

# Test functions to explore alternative formulations
def LVtest2(y, t, r, Amat):
    vlen = len(r)
    dN = list(r*y*(np.ones(vlen)-np.matmul(Amat,y)/ktot)*(1-sum(y)/ktot))
    return dN

intermat = np.array([[ 1, 0.11169748, 0.15714228],
       [ -0.14614691,  1,  -0.08447383],
       [ -0.13419745, 0.07566981,  1]])

# Do a test
times = np.linspace(0,30,100000)
soln = odeint(LVnprime, [400,50,50], times, args=([0.71965564,1.18736729,1.07266845], intermat))
soln1 = odeint(LVnprime, [400,50,50], times, args=([1,1,1], intermat))

plt.plot(times, soln[:,0])
plt.plot(times, soln[:,1])
plt.plot(times, soln[:,2])
plt.show()
print(sum(soln[-1]), soln[-1])

plt.clf()
plt.plot(times, soln1[:,0])
plt.plot(times, soln1[:,1])
plt.plot(times, soln1[:,2])
plt.show()
print(sum(soln1[-1]), soln1[-1])