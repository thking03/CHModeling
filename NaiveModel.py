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
def nprime(y, t):
    dN = list(np.matmul(A0,y)*(1-sum(y)/ktot))
    return dN
