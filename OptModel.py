""" 
Optimize parameters for a numerically-solved differential equation for logistic growth with multiple populations model
"""

import numpy as np
from scipy.integrate import odeint

# initial guesses for parameters must be previously defined
# target must be same length as ICs and can be a list
# params is the matrix A0 which we assume is square
def naiveopt(dfunc, ics, times, params, target, rate=1, tol=.00001):
    losses = [np.inf]
    A0 = params # Need to get this A0 to override the global A0
    p_target = np.array(target / np.linalg.norm(target))
    count = 0
    dloss = 100
    while dloss > tol or count < 10:
        stepsoln = odeint(dfunc, ics, times, args=(A0,))
        n_stable = np.array(stepsoln[-1])
        p_vect = n_stable / np.linalg.norm(n_stable)
        e_vect = p_target - p_vect
        l_vect = e_vect**2
        for j in range(len(A0)):
            A0[j,j] += np.sign(e_vect[j])*l_vect[j]*rate
        
        # Diagnostic - remove later
        print("matrix at step ", count, "is ", A0)
        print("n_stable at step ", count, "is ", n_stable)
        print("p_vect at step ", count, "is ", p_vect)

        loss = sum(l_vect)
        dloss = losses[count] - loss
        losses.append(loss)
        count += 1
    return [A0, losses]

# Testing
def nprime(y, t, Amat):
    dN = list(np.matmul(Amat,y)*(1-sum(y)/ktot))
    return dN

a11, a22, a33 = 1, 1, 1
A0 = np.array([[a11,0,0],[0,a22,0],[0,0,a33]])
ktot = 30000
times = np.linspace(0,10,10000)