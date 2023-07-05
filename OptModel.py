""" 
Optimize parameters for a numerically-solved differential equation for logistic growth with multiple populations
"""

import numpy as np
from scipy.integrate import odeint

def naiveopt(dfunc, ics, times, params, target, rate=1, tol=1e-6):
    """
    optimizes parameters for a multi-species logistic population model with no interactions (interaction matrix is diagonal)
    Args:   dfunc -- function used for ODEINT with 
            ics -- initial conditions for ODEINT
            times -- times for ODEINT to evaluate
            params -- initial guess for diagonal interaction matrix
            target -- target population proportions
            rate -- "learning rate" of the model
            tol -- tolerance of the model (determines when to stop)
    Returns: a list containing the optimized parameter matrix (A0), a list of losses beginning w/o initial loss of inf, and the output of the final step of ODEINT
    """
    losses = [np.inf]
    A0 = params
    p_target = np.array(target / np.linalg.norm(target))
    count = 0
    dloss = 100
    while dloss > tol and count < 1000:
        stepsoln = odeint(dfunc, ics, times, args=(A0,))
        n_stable = np.array(stepsoln[-1])
        p_vect = n_stable / np.linalg.norm(n_stable)
        e_vect = p_target - p_vect
        l_vect = e_vect**2
        for j in range(len(A0)):
            A0[j,j] += np.sign(e_vect[j])*l_vect[j]*rate
        
        # Diagnostic - can remove later
        # print("matrix at step ", count, "is ", A0)
        # print("n_stable at step ", count, "is ", n_stable)
        # print("p_vect at step ", count, "is ", p_vect)

        loss = sum(l_vect)
        dloss = losses[count] - loss
        losses.append(loss)
        count += 1
    return [A0, losses[1:], stepsoln]