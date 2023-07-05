"""
Naive Model of cell growth in murine CH
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Set up naive model with prototype matrix A0 and initialize diagonal to 1
a11, a22, a33 = 1, 1, 1
A0 = np.array([[a11,0,0],[0,a22,0],[0,0,a33]])
ktot = 30000

# Define derivative, which is of the form A.y(1-Sum(y)/k)
def nprime(y, t, Amat):
    dN = list(np.matmul(Amat,y)*(1-sum(y)/ktot))
    return dN

# Test the function to make sure that it returns properly
times = np.linspace(0,10,10000)
testsoln = odeint(nprime,[350,100,50],times, args=(A0,))

print(np.shape(testsoln))

plot1 = plt.figure(num=1, clear=True)
ax = plot1.add_subplot(1,1,1)
ax.plot(times, testsoln[:,0], label="Clone 1")
ax.plot(times, testsoln[:,1], label="Clone 2")
ax.plot(times, testsoln[:,2], label="Clone 3")
ax.set(xlabel="time (days)", ylabel="population")
ax.legend()
plt.show()

# Optimize based on data
from OptModel import *

results = naiveopt(nprime, [400,50,50], times, np.identity(3), [6, 71, 23], order=1)
print(results[0])

plot2 = plt.figure(num=2, clear=True)
optax = plot2.add_subplot(1,1,1)
optax.plot(times, results[2][:,0], label="WT")
optax.plot(times, results[2][:,1], label="TP53")
optax.plot(times, results[2][:,2], label="Tet2")
optax.set(xlabel="time (days)", ylabel="n (# cells)")
optax.legend()

plot3 = plt.figure(num=3, clear=True)
lossax = plot3.add_subplot(1,1,1)
lossax.plot(range(len(results[1])),results[1], label="losses")
lossax.set(xlabel="iteration", ylabel="loss: sum of squared errors")
lossax.set_yscale("log")
plt.show()