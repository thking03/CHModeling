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