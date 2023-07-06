""" 
Optimize parameters for a numerically-solved differential equation for logistic growth with multiple populations given the end population distribution
NOTE: All methods assume that the population of cells fully expands to the carrying capacity of the niche prior to sampling
"""

import numpy as np
from scipy.integrate import odeint

def naiveopt(dfunc, ics, times, params, target, rate=1, tol=1e-6, order=1, maxiter=1000):
    """
    Optimizes parameters for a multi-species logistic population model with no interactions (interaction matrix is diagonal) and a shared carrying capacity.
    NOTE: this method only takes one target (see naivemultiopt() for multiple sets of data)
    Args:   dfunc -- function used for ODEINT with 
            ics -- initial conditions for ODEINT
            times -- times for ODEINT to evaluate
            params -- initial guess for diagonal interaction matrix
            target -- target population proportions
            rate -- "learning rate" of the model
            tol -- tolerance of the model (determines when to stop)
            order -- order of norms taken, set to L1-norm
            maxiter -- maximum number of steps to take
    Returns: a list containing the optimized parameter matrix (A0), a list of losses beginning w/o initial loss of inf, and the output of the final step of ODEINT
    """
    losses = [np.inf]
    A0 = params
    p_target = np.array(target / np.linalg.norm(target, ord=order))
    count = 0
    dloss = 100
    while dloss > tol and count < maxiter:
        stepsoln = odeint(dfunc, ics, times, args=(A0,))
        n_stable = np.array(stepsoln[-1])
        p_vect = n_stable / np.linalg.norm(n_stable, ord=order)
        e_vect = p_target - p_vect
        l_vect = e_vect**2
        for j in range(len(A0)):
            A0[j,j] += np.sign(e_vect[j])*l_vect[j]*rate
        # print("matrix at step ", count, "is ", A0)
        # print("n_stable at step ", count, "is ", n_stable)
        # print("p_vect at step ", count, "is ", p_vect)
        loss = sum(l_vect)
        dloss = losses[count] - loss
        losses.append(loss)
        count += 1
    if count == maxiter:
        print("Terminated due to max iterations ({no}).".format(no=count))
    else:
        print("Terminated at {no} iterations due to reaching error change specified".format(no=count))
    return [A0, losses[1:], stepsoln]

def naivemultiopt(dfunc, ics, times, params, targets, rate=1, tol=1e-6, order=1, maxiter=1000):
    """
    Given multiple sets of ICs and targets, optimizes parameters for a multi-species logistic with no interactions (interaction matrix is diagonal) and a shared carrying capacity. Structure is similar to naiveopt().
    Args:   dfunc -- function used for ODEINT with 
            ics -- LIST of initial conditions for ODEINT
            times -- times for ODEINT to evaluate
            params -- initial guess for diagonal interaction matrix
            targets -- LIST of target population proportions
            rate -- "learning rate" of the model
            tol -- tolerance of the model (determines when to stop)
            order -- order of norms take, set to L1-norm
            maxiter -- maximum number of steps to take
    Returns: a list containing the optimized parameter matrix (A0) and a list of losses beginning w/o initial loss of inf. Does not return last step of ODEINT (unlike naiveopt(), which does).
    """
    losses = [np.inf]
    A0 = params
    targets = [np.array(target / np.linalg.norm(target, ord=order)) for target in targets]
    count = 0
    dtotloss = 100
    while dtotloss > tol and count < maxiter:
        totloss = 0
        dmat = np.zeros_like(A0)
        for i in range(len(targets)):
            stepsoln = odeint(dfunc, ics[i], times, args=(A0,))
            n_stable = np.array(stepsoln[-1])
            p_vect = n_stable / np.linalg.norm(n_stable, ord=order)
            e_vect = targets[i] - p_vect
            l_vect = e_vect**2
            for j in range(len(A0)):
                dmat[j,j] += np.sign(e_vect[j])*l_vect[j]*rate
            totloss += sum(l_vect)
        A0 += dmat
        dtotloss = losses[count] - totloss
        losses.append(totloss)
        count += 1
    if count == 1000:
        print("Terminated due to max iterations (1000).")
    else:
        print("Terminated at {no} iterations due to reaching error change specified".format(no=count))        
    return [A0, losses]