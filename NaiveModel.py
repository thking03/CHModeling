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

# Create 