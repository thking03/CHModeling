"""
Competitive Lotka-Volterra Model of cell growth

NOTE: OptModel() contains a method to numerically fit equilibrium values given enough time in an ODE solver. This file contains a different method FitInTime() that is used to fit timed experimental data and is designed to be used with scipy.optimize.minimize() as it returns a sum of squares rather than providing its own parameter-optimization method.

List of functions:
    - LVnprime(): Returns derivative of population vector for a competitive Lotka-Voltera model
    - FitInTime(): Given parameters, evaluates a system of ODEs across a period of interest and returns loss (including penalty functions)
    - TreatmentFitInTime(): FitInTime() for non-control data, incorporating multiphase experimental model and multiple parameter sets
    - do_CLVOpt(): Performs the optimization routine for a control-type CHData point 
    - do_treat_CLVOpt(): do_CLVOpt() for non-control data, incorporating multiphase experimental model and multiple parameter sets
    - find_neighbors(): for a non-control data point, explores control data to establish strict optimization constraints for parameters
    - getstat_CLVOpt(): given a dict of CHData, performs optimization routine on each one, compiles parameter information, and returns information about the distribution of parameters
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from OptModel import *
from parsedata import *
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

def FitInTime(params, dfunc, chdata, interaction_const=np.inf, samplingrate=1000):
    """
    Solves an ODE given parameters and then evaluates the results against time-associated data, returning a sum of squares. 
    Args:   
            params: 1D array of parameters to be fit. Must be flattened list of rvec and then Amat in the standard order [WT, TP53, Tet2].
            dfunc: ODE to be evaluated
            data: CHData object that contains what data should be fitted.
            samplingrate: how many points (per week) the function should use in ODE solving
            interaction_const: value for constraints on interactions which will take the form (-interaction_const, +interaction_const)
    Returns: Sum of squared errors from the data.
    """
    if type(chdata) != type(CHData()):
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

    if interaction_const != np.inf:
        for aij in aprog:
            penalty += max(abs(min(rate+interaction_const, 0)),abs(max(rate-interaction_const,0)))

    return totloss + penalty

def do_CLVopt(chdata, verbose=True, savefig=False, savepath="plots", getloss=False, **kwargs):
    """
    Runs the optimization routine for the specified datapoint. 
    Args:   
            chdata: the datapoint specified, must be CHData
            verbose: if true, will print runtime and plot data/model in addition to returning the parameters
            savefig: if true, will save the plots generated if verbose is set to true
            savepath: allows user to specify where to save plots
            getloss: bool that if set to true will have the function return the loss in adition to parameters
    Accepted kwargs:
            interaction_const: optional argument to be passed to FitInTime()
    Returns: A list of parameters (rate vector and interaction matrix), with the loss appended if getloss is set to true
    """
        
    start = time.time()        
    guess0 = np.concatenate((np.ones(3).reshape(1,3), np.zeros((2,3)))).flatten()
    base_args = (LVnprime, chdata)
    if "interaction_const" in kwargs:
        pass_args = base_args + (kwargs.get("interaction_const"),)
    else:
        pass_args = base_args

    minobj = minimize(FitInTime, guess0, args=pass_args, method="Nelder-Mead", options={"maxiter":5000, "disp":False, "xatol":5e-6, "fatol":5e-6})
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

def do_treat_CLVopt(chdata, verbose=True, savefig=False, savepath="plots", getloss=False, **kwargs):
    """
    Equivalent function to do_CLVOpt() for treatment data incorporating the find_neighbors routine and a two-phase CLV model.

    This function can also optionally take in a dictionary containing control data, which allows constraint identification for the first phase of the CLV model (nontreatment). If no control dictionary is passed (or a pair of neighbors is not identified), then the routine allows the entire search area to be permitted. Interaction constraints can still be set in the same manner as the do_CLVopt function.
    
    Args:   
            chdata: the datapoint specified, must be CHData
            verbose: if true, will print runtime and plot data/model in addition to returning the parameters
            savefig: if true, will save the plots generated if verbose is set to true
            savepath: allows user to specify where to save plots
            getloss: bool that if set to true will have the function return the loss in adition to parameters
    Accepted kwargs:
            controldict: optional dictionary of control-type data to be used in a find_neighbors call
            interaction_const: optional argument to be passed to FitInTime()
    Returns: A list of parameters (rate vector and interaction matrix), with the loss appended if getloss is set to true
    """
    if "controldict" in kwargs:
        cdict = kwargs.get("controldict")
        bounds = find_neighbors(chdata, cdict)
        if bounds[0] != bounds[1]:
            cdict[bounds[0]].data[0]
            cdict[bounds[1]].data[0]
            # Need to do some stuff w/ the data type so we can store parameters & call optimization at-will
    pass

def find_neighbors(chdata, controldict):
    """
    Helper function for treatment optimization (i.e. cisplatin). 
    
    Takes in a CHData and compares it to a reference dictionary containing fitted control CHData. NOTE: For now this function assumes Wk. 5 PB data is PRESENT and in the right format (i.e. we only compare 2X to 2X), or we raise an exception.

    Args:
        chdata: a CHData of non-control type
        controldict: a dictionary containing control CHData
    Returns: A list containing the two keys defining the minimal-area rectangle containing the specified CHData at wk 5.
    """
    for data in chdata.data:
        if data.week == 5:
            target = data.probs
            break
        else:
            target = False
    if not target:
        raise Exception("No week 5 data for comparison available in the specified CHData.")
    
    log = []
    for key in controldict.keys():
        thisdata = controldict[key]
        for data in thisdata.data:
            if data.week == 5:
                compare = data.probs
                break
            else:
                compare = False
        if not compare:
            print("Warning: CHData {} does not contain comparable data.".format(key))
            continue
        diff = [compare[i+1] - target[i+1] for i in range(len(compare)-1)] # Drop the first datapoint (WT), since for these values it is essentially dependent on the other population proportions (they must sum to 1)
        dist = np.linalg.norm(diff)
        log.append([dist, diff, key])
    
    log.sort(key=lambda x: x[0])

    start = 0
    end = len(log)
    vec1pos = 0
    vec2pos = False
    area = np.inf
    for i in range(end):
        base = np.array([obs for obs in log[i][1]])
        signs = np.sign(base)
        j = 0
        for compare in log[:i] + log[i+1:end]:
            compvals = np.array([obs for obs in compare[1]])
            compsigns = np.sign(compvals)
            if all(t == 0 for t in signs+compsigns):
                temparea = abs((base[0]-compvals[0])*(base[1]-compvals[1]))
                if temparea < area:
                    vec1pos = i
                    vec2pos = j + (j >= i)
                    area = temparea
                    end = j + (j >= i)
                break
            j += 1
        if i == end:
            break  

    if vec2pos == False:
        print("ERROR")
        pass

    keys = [log[vec1pos][-1],log[vec2pos][-1]]
    print("The minimal rectangle containing {} is given by {} and {}.".format(chdata.name, *keys))
    return keys

def getstat_CLVopt(datadict, illustrate=True, **kwargs):
    """
    Function that takes in a dictionary of CHData objects, runs the optimizer, and returns information about the parameter distribution of the fits. Always will also return information about stats.
    Args:
            datadict: the dictionary to be iterated over
            illustrate: boolean that determines whether or not to display a histogram of distributions and other statistical charts
    Accepted kwargs:
            interaction_const: interaction constraint for FitInTime
    Returns:
            a list of lists (as an np.array) [flattened parameter list, loss] for each data point passed in; the parameter list INCLUDES the diagonal ones, which the method will discard later
    """

    account = []
    for key in datadict.keys():
        result = do_CLVopt(datadict[key], verbose=False, getloss=True, **kwargs)
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
if __name__=='__main__':
    testdict = {}
    sheet = r"C:\Users\tyler\Downloads\Tet2+TP53_summary.xlsx"
    readdata(sheet, testdict, "CW8_WBM")
    readdata(sheet, testdict, "CW6_BM")
    readdata(sheet, testdict, "CW6_PB")
    readdata(sheet, testdict, "CW8_PB")
    trialdata = [testdict["258a"], testdict["258b"], testdict["258c"], testdict["259a"],testdict["259c"],testdict["259d"]]
    
    # tottimestart = time.time()
    # for data in trialdata:
    #     do_CLVopt(data, savefig=False)
    # tottimeend = time.time()
    # print("In total took {t} seconds to evaluate {d} datapoints.".format(t=tottimeend-tottimestart, d=len(trialdata))) 