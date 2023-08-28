"""
Competitive Lotka-Volterra Model of cell growth

NOTE: OptModel() contains a method to numerically fit equilibrium values given enough time in an ODE solver. This file contains a different method FitInTime() that is used to fit timed experimental data and is designed to be used with scipy.optimize.minimize() as it returns a sum of squares rather than providing its own parameter-optimization method.

List of functions:
    - LVnprime(): Returns derivative of population vector for a competitive Lotka-Voltera model
    - FitInTime(): Given parameters, evaluates a system of ODEs across a period of interest and returns loss (including penalty functions)
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

def LVnprime(y, t, r, Amat, ktot=30000):
    """
    Returns the derivative at a time-step for the competitive Lotka-Volterra model.
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
    if type(chdata) != type(CHData("foo")):
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
    
    if chdata.data[-1].type == "BM" and hasattr(chdata.data[-1], "bmcount"):
        ntarget = [i/100*chdata.data[-1].bmcount for i in chdata.data[-1].probs]
        nlvect = (ntarget - soln[-1])**2
        adjnloss = sum(nlvect)/chdata.data[-1].bmcount**2 # square this to match squared error
        totloss += adjnloss

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

def do_treat_CLVopt(chdata, verbose=True, savefig=False, savepath="plots", getloss=False, samplingrate=1000, neg_rate=False, **kwargs):
    """
    Equivalent function to do_CLVOpt() for treatment data incorporating the find_neighbors routine and a two-phase CLV model.

    This function can also optionally take in a dictionary containing control data, which allows constraint identification for the first phase of the CLV model (nontreatment). If no control dictionary is passed (or a pair of neighbors is not identified), then the routine allows the entire search area to be permitted. Interaction constraints can still be set in the same manner as the do_CLVopt function. shortloss() and longloss() are written as inner functions here, 
    
    Args:   
            chdata: the datapoint specified, must be CHData
            verbose: if true, will print runtime and plot data/model in addition to returning the parameters
            savefig: if true, will save the plots generated if verbose is set to true
            savepath: allows user to specify where to save plots
            getloss: bool that if set to true will have the function return the loss in adition to parameters
            samplingrate: how many points (per week) the function should use in ODE solving
            neg_rate: if true, will allow rates from [-2, 2] instead of [0, 2]
    Accepted kwargs:
            controldict: optional dictionary of control-type data to be used in a find_neighbors call
            interaction_const: optional argument to be passed to FitInTime()
    Returns: A list of parameters (rate vector and interaction matrix), with the loss appended if getloss is set to true
    """
    if "controldict" in kwargs:
        cdict = kwargs.get("controldict")
        bounds = find_neighbors(chdata, cdict)
        constraints = []
        if bounds[0] == bounds[1]:
            print("No pair of neighbors identified; no constraints will be assigned.")
            constraints = False
            # Also might want to throw a warning here
        else:
            pt1 = cdict[bounds[0]]
            pt2 = cdict[bounds[1]]
            if pt1.opt != True:
                pt1.optimize()
            if pt2.opt != True:
                pt2.optimize()
            for i in range(3):
                constraints.append([pt1.rates[i],pt2.rates[i]])
            for i in range(3):
                for j in range(3):
                    if i != j:
                        constraints.append([pt1.interactions[i,j], pt2.interactions[i,j]])
                        print("Constraints assigned")
    else:
        constraints = False
        print("No constraints assigned")
    
    # Assign initial conditions to be accessed by inner functions
    if "2X" in chdata.type:
            ics = [400, 50, 50]
    elif "Tet2" in chdata.type:
            ics = [450, 0, 50]
    elif "TP53" in chdata.type:
            ics = [450, 50, 0]

    # inner function to pass into minimize, which incorporates the constraints. basically the same as FitInTime() but for only up to wk. 5.
    def shortloss(params, dfunc, constlist=False, interaction_const=False):
        if len(params) != 9:
            raise Exception("9 parameters must be passed into the function as a 1D list.")
        else:
            rvec = np.array(params[:3])
            aprog = np.array(params[3:])
            for i in range(len(rvec)):
                aprog = np.insert(aprog, 4*i, 1)
            amat = aprog.reshape((3,3))
        
        soln = odeint(dfunc, ics, np.linspace(0, 6, 6*samplingrate), args=(rvec, amat))
        nsoln = soln[samplingrate*(5)-1]
        lvect = ([i/100 for i in chdata.data[0].probs] - nsoln/(sum(nsoln)))**2
        loss = sum(lvect)
        
        # Penalty function for constrained optimization. Constraints in control OVERRIDE interaction_const / other defaults.
        penalty = 0
        if constlist:
            for i in range(3):
                penalty += max(abs(min(rvec[i]-min(constlist[i]), 0)),abs(max(rvec[i]-max(constlist[i]),0)))
            # for i in range(6):
            #     penalty += max(abs(min(aprog[i]-min(constlist[i+3]), 0)),abs(max(aprog[i]-max(constlist[i+3]),0)))
        elif interaction_const:
            for i in range(3):
                penalty += max(abs(min(rvec[i], 0)),abs(max(rvec[i]-2,0)))
            for i in range(6):
                penalty += max(abs(min(rvec[i]+interaction_const, 0)),abs(max(rvec[i]-interaction_const,0)))

        # Constraint penalty function
        return loss+penalty
    
    args = (LVnprime, constraints)
    if "interaction_const" in kwargs:
        args = args + (kwargs.get("interaction_const"))

    guess0 = np.concatenate((np.ones(3).reshape(1,3), np.zeros((2,3)))).flatten()
    startminobj = minimize(shortloss, guess0, args=args, method="Nelder-Mead", options={"maxiter":5000, "disp":False, "xatol":5e-6, "fatol":5e-6})

    rvec1 = np.array(startminobj['x'][:3])
    aprog1 = np.array(startminobj['x'][3:12])
    for i in range(len(rvec1)):
        aprog1 = np.insert(aprog1, 4*i, 1)
    amat1 = aprog1.reshape((3,3))

    returnlist = [rvec1, amat1]
    if getloss:
        returnlist.append(startminobj['fun'])
    ics2 = odeint(LVnprime, ics, np.linspace(0, 6, 6*samplingrate), args=(rvec1, amat1))[samplingrate*(5)-1]

    # Now that the initial parameters have been optimized, proceed with minimization using TreatFitInTime()... tbw. Rn returns parameters to test.
    # another inner function to minimize
    def longloss(params, dfunc, interaction_const=False):
        if len(params) != 9:
            raise Exception("9 parameters must be passed into the function as a 1D list.")
        else:
            rvec = np.array(params[:3])
            aprog = np.array(params[3:])
            for i in range(len(rvec)):
                aprog = np.insert(aprog, 4*i, 1)
            amat = aprog.reshape((3,3))

        soln = odeint(dfunc, ics2, np.linspace(6, 14, 8*samplingrate), args=(rvec, amat))
        
        totloss = 0
        for profile in chdata.data[1:]: # Exclude up to week 6
            nsoln = soln[samplingrate*(profile.week-6)-1] # week 6 --> apparent week 0
            psoln = nsoln/(sum(nsoln))
            ptarget = [i/100 for i in profile.probs]
            lvect = (ptarget - psoln)**2
            pointloss = sum(lvect)
            totloss += pointloss
        
        penalty = 0
        for rate in rvec:
            penalty += max(abs(min(rate+2*neg_rate, 0)),abs(max(rate-2,0))) # In this phase we allow cells to have negative death rate due to treatment if neg_rate is enabled

        if interaction_const:
            for aij in aprog:
                penalty += max(abs(min(rate+interaction_const, 0)),abs(max(rate-interaction_const,0)))

        return totloss + penalty
    
    midminobj = minimize(longloss, startminobj['x'], args=(LVnprime,), method="Nelder-Mead", options={"maxiter":5000, "disp":False, "xatol":5e-6, "fatol":5e-6})

    rvec2 = np.array(midminobj['x'][:3])
    aprog2 = np.array(midminobj['x'][3:12])
    for i in range(len(rvec2)):
        aprog2 = np.insert(aprog2, 4*i, 1)
    amat2 = aprog2.reshape((3,3))

    returnlist.append(rvec2)
    returnlist.append(amat2)
    if getloss:
        returnlist.append(midminobj['fun'])

    return returnlist

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