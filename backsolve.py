"""
Script for back-solving the DE system w/ week 14 data, week 12 data, and week 5 data to estimate engraftment potentials. Written modularly to allow for future use. Only control data is passed.
"""
# Import data; then optimize the control range.
import numpy as np
import matplotlib.pyplot as plt
from parsedata import *
from CLVModel import *
from scipy.integrate import odeint

cw8dict = {}
sheet = r"C:\Users\tyler\Downloads\updated_hsc_summary.xlsx"
readdata(sheet, cw8dict, "CW8_WBM")
readdata(sheet, cw8dict, "CW8_PB")
add_bm_counts(sheet, cw8dict, "CW8_BM_counts", "CW8_HSC")

control8dict = {}
for key in cw8dict.keys():
    if cw8dict[key].type.lower() != "2x":
        pass
    elif cw8dict[key].treatment.lower() != "control":
        pass
    elif len(cw8dict[key].data) != 3:
        pass
    else:
        control8dict[key] = cw8dict[key]

treat8dict = {}
for key in cw8dict.keys():
    if cw8dict[key].treatment.lower() == "control":
        pass
    elif cw8dict[key].type.lower() != "2x":
        pass
    elif len(cw8dict[key].data) == 3:
        treat8dict[key] = cw8dict[key]

for key in control8dict.keys():
    control8dict[key].optimize()

treat8tracker = []
for key in treat8dict.keys():
    treat8tracker.append(do_treat_CLVopt(treat8dict[key], verbose=False, controldict=control8dict))

# Bootstrapping implementation
def rev_LVnprime(y, t, r, Amat, ktot=30000):
    neg = -1*np.array(LVnprime(y, t, r, Amat, ktot=ktot))
    return neg

def back_bootstrap(iternum, sampledict):
    """
    Bootstrapping method for back-calculation of initial cell compositions. Random models are sampled with replacement for final cell composition and growth rates. This method is only compatible with control data currently. Additionally, the growth rates are not sampled independently -- a model is randomly selected, and its growth rates are used. 
    """
    i = 0
    tracker = []
    while i < iternum:
        rand_key1 = np.random.choice(list(sampledict.keys())) # final cell composition
        rand_key2 = np.random.choice(list(sampledict.keys())) # growth rates
        rand_key3 = np.random.choice(list(sampledict.keys())) # interaction matrix
        counts = sampledict[rand_key1].data[-1].bmcount * np.array(sampledict[rand_key1].data[-1].probs)/100
        rvec = sampledict[rand_key2].rates
        amat = sampledict[rand_key3].interactions
        soln = odeint(rev_LVnprime, counts, np.linspace(0,14,140000), args=(np.array(rvec), np.array(amat)))
        imputed_ics = soln[-1]
        tracker.append(imputed_ics)
        i +=1
    return tracker

# Create the PDF for rates (and also maybe cell count distributions???)

# For each cell count sample rates from the PDF and run ODEINT backwards (i.e. -dy/dx is the step)

# %%
