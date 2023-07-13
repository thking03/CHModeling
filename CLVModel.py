"""
Competitive Lotka-Volterra Model of cell growth

NOTE: OptModel() contains a method to numerically fit equilibrium values given enough time in an ODE solver. This file contains a different method FitInTime() that is used to fit timed experimental data and is designed to be used with scipy.optimize.minimize() as it returns a sum of squares rather than providing its own parameter-optimization method.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from OptModel import *

# LVnprime() defines the competitive Lotka-Volterra model with carrying capacity ktot
ktot = 30000
def LVnprime(y, t, r, Amat):
    vlen = len(r)
    dN = list(r*y*(np.ones(vlen)-np.matmul(Amat,y)/ktot))
    return dN

def FitInTime(params, dfunc, chdata, samplingrate=1000):
    """
    Solves an ODE given parameters and then evaluates the results against time-associated data, returning a sum of squares. 
    Args:   
            params: 1D array of parameters to be fit. Must be flattened list of rvec and then Amat in the standard order [WT, TP53, Tet2].
            dfunc: ODE to be evaluated
            data: CHData object that contains what data should be fitted.
            samplingrate: how many points (per week) the function should use in ODE solving
    Returns: Sum of squared errors from the data.
    """
    if type(chdata) != CHData:
        raise Exception("FitInTime() currently requires data passed in as the CHData class.")
    
    if len(params) != 12:
        raise Exception("12 parameters must be passed into the function as a 1D list.")
    else:
        rvec = np.array(params[:3])
        amat = np.array(params[3:]).reshape((3,3))
        for i in range(len(amat)):
            amat[i,i] = 1
    
    if "2X" in chdata.type:
        ics = [400, 50, 50]
    elif "Tet2" in chdata.type:
        ics = [450, 0, 50]
    elif "TP53" in chdata.type:
        ics = [450, 50, 0]
    
    soln = odeint(dfunc, ics, np.linspace(0, 14, 14*samplingrate), args=(rvec, amat))
    
    totloss = 0
    for profile in chdata.data:
        nsoln = soln[samplingrate*(profile.week)-1]
        psoln = nsoln/(sum(nsoln))*100
        ptarget = profile.probs
        lvect = (ptarget - psoln)**2
        pointloss = sum(lvect)
        totloss += pointloss
    
    print(totloss)
    return totloss

"""
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
"""