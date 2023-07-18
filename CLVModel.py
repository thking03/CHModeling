"""
Competitive Lotka-Volterra Model of cell growth

NOTE: OptModel() contains a method to numerically fit equilibrium values given enough time in an ODE solver. This file contains a different method FitInTime() that is used to fit timed experimental data and is designed to be used with scipy.optimize.minimize() as it returns a sum of squares rather than providing its own parameter-optimization method.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from OptModel import *
from scipy.optimize import minimize
import time

ktot = 30000
def LVnprime(y, t, r, Amat):
    """
    Returns the derivative at a time-step for the competitive Lotka-Volterra model with carrying capacity ktot defined outside of the function.
    """
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
    
    if len(params) != 9:
        raise Exception("9 parameters must be passed into the function as a 1D list.")
    else:
        rvec = np.array(params[:3])
        aprog = np.array(params[3:])
        for i in range(len(rvec)):
            aprog = np.insert(aprog, 4*i, 1)
        amat = aprog.reshape((3,3))
    
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
        psoln = nsoln/(sum(nsoln))
        ptarget = [i/100 for i in profile.probs]
        lvect = (ptarget - psoln)**2
        pointloss = sum(lvect)
        totloss += pointloss
    
    # Penalty function for constrained optimization
    # growth rates cannot be less than zero or greater than two (see [1], pg. 38 where this is the truncated range of birthrates; also [2]); we use (always positive) ramp functions on either side of an acceptable range to create a "well" in which rates can exist without penalty
    penalty = 0
    for rate in rvec:
        penalty += max(abs(min(rate, 0)),abs(max(rate-2,0)))

    return totloss + penalty

def do_CLVopt(chdata, verbose=True, savefig=False, savepath="plots", getloss=False):
    """
    Runs the optimization routine for the specified datapoint. 
    Args:   
            chdata: the datapoint specified, must be CHData
            verbose: if true, will print runtime and plot data/model in addition to returning the parameters
            savefig: if true, will save the plots generated if verbose is set to true
            savepath: allows user to specify where to save plots
    """
        
    start = time.time()        
    guess0 = np.concatenate((np.ones(3).reshape(1,3), np.zeros((2,3)))).flatten()
    minobj = minimize(FitInTime, guess0, args=(LVnprime, chdata), method="Nelder-Mead", options={"maxiter":5000, "disp":False, "xatol":5e-6, "fatol":5e-6})
    end = time.time()

    optparams = minobj["x"]
    print(optparams)
    optr = optparams[:3]
    optAprog = optparams[3:]
    for i in range(3):
        optAprog = np.insert(optAprog, 4*i, 1)
    print(optAprog)
    optA = optAprog.reshape((3,3))
    
    if verbose:
        print("Optimization took {} seconds.".format(end-start))
        
        if "2X" in chdata.type:
            ics = [400, 50, 50]
        elif "Tet2" in chdata.type:
            ics = [450, 0, 50]
        elif "TP53" in chdata.type:
            ics = [450, 50, 0]

        times = np.linspace(0,20,20000)
        soln = odeint(LVnprime, ics, times, args=(optr, optA))
        
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(1,1,1)
        ax.plot(times, soln[:,0], 'b-', label="WT")
        ax.plot(times, soln[:,1], 'g-', label="TP53")
        ax.plot(times, soln[:,2], 'k-', label="Tet2")
        ax.set(ylabel="population size", xlabel="time (weeks)", title="Population trajectories for model {}".format(chdata.name))
        for profile in chdata.data:
            colororder = ['bo', 'go', 'ko']
            for i in range(3):
                ax.plot(profile.week, profile.probs[i]/100*sum(soln[1000*profile.week-1]), colororder[i])
        ax.legend()
        fig.show()

        if savefig:
            fig.savefig("{dir}/{ID}.png".format(dir=savepath, ID=chdata.name))

        print("Model ", chdata.name)
        print("---")
        print("The loss for this model is ", minobj["fun"])
        print("Rate vector is: ", optr)
        print("Interaction matrix is: ", optA)

    returnlist = [optr, optA]

    if getloss:
        returnlist.append(minobj["fun"])

    return returnlist

def getstat_CLVopt(datadict, illustrate=True):
    """
    Function that takes in a dictionary of CHData objects, runs the optimizer, and returns information about the parameter distribution of the fits. Always will also return information about stats.
    Args:
            datadict: the dictionary to be iterated over
            illustrate: boolean that determines whether or not to display a histogram of distributions and other statistical charts
    Returns:
            a list of lists (as an np.array) [flattened parameter list, loss] for each data point passed in; the parameter list INCLUDES the diagonal ones, which the method will discard later
    """

    account = []
    for key in datadict.keys():
        result = do_CLVopt(datadict[key], verbose=False, getloss=True)
        paramarr = np.concatenate((result[0].reshape(1,3), result[1])).flatten()
        paramarr = np.append(paramarr, result[2])
        account.append(paramarr)
    account = np.array(account)

    if illustrate:
        n_bins = 20
        for i in range(len(account[0])-1):
            if i < 3:
                param = "r{}".format(i+1)
            else:
                param = "a{}{}".format((i-3)//3+1, (i-3)%3+1)
            
            # Histogram plot
            fig = plt.figure(num=1, clear=True)
            ax = fig.add_subplot(1,1,1)
            ax.hist(account[:,i], bins=n_bins)
            ax.set(xlabel="value", ylabel="number", title="Distribution of {}".format(param))
            fig.show()
            fig.savefig("plots/stats/rawhist_{}.jpg".format(param)) # .jpg because .png was not saving other info

            # Statistics

    return account


# Testing to make sure that methods work properly
from parsedata import *
if __name__=='__main__':
    testdict = {}
    sheet = r"C:\Users\tyler\Downloads\Tet2+TP53_summary.xlsx"
    readdata(sheet, testdict, "CW6_BM")
    readdata(sheet, testdict, "CW8_WBM")
    readdata(sheet, testdict, "CW6_PB")
    readdata(sheet, testdict, "CW8_PB")
    trialdata = [testdict["258a"], testdict["258b"], testdict["258c"], testdict["259a"],testdict["259c"],testdict["259d"]]
    
    tottimestart = time.time()
    for data in trialdata:
        do_CLVopt(data, savefig=False)
    tottimeend = time.time()
    print("In total took {t} seconds to evaluate {d} datapoints.".format(t=tottimeend-tottimestart, d=len(trialdata)))