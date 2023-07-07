# CHModeling
Baseline modeling for clonal hematopoiesis

## 7/3/23 - Present
- Currently working with estimated data on a naive (simple) logistic model
- Using Python in `NaiveModel.py` to perform numeric integration
- `OptModel.py` contains code used to "fit" parameters to the differential model (solved numerically)
    - The function `naiveopt()` takes in a single data point (equilibrium proportions of the population) and returns optimized parameters, losses, and the final ODE model
    - The function `naivemultiopt()` takes in multiple data points (equilibrium proportions of the population) and returns optimized parameters and losses

### Notes / TBD
- Should `naivemultiopt()` use the average of nudges (in dmat) rather than the sum when updating the parameters? -- slower learning rate, but maybe more accurate?
- Beginning work on new `multiopt()` function which admits the entire nxn matrix A, not just a diagonal matrix --> need to explore how to optimize the interaction coefficients, currently leaning toward something like $\text{sign}({e_{c1}})\sqrt{l_{c1}*l_{c2}}$
- `multiopt()` needs a slower learning rate since more things are changing -- initialized to .1 instead of 1
- Need to investigate an error in `multiopt()` that allows populations that begin at size 0 to spontaneously grow (has to do with how the interaction rates are defined)

### Approximated Data & conditions from Katharina's slides for naive modeling
- mice sac'ed @ wk 14
- cisplatin admin from wk. 6-10 -- ignore in initial naive modeling because only endpoints available
- Have information about TP53 and Tet2 in PB and BM in the format (min, q1, median, q3, max) -- data manually collected and tabulated at `naivedata.csv`, so there is likely some small human error in these values 
- Data consists of 16 groups consisting of all combinations of (1X OR 2X) AND (TP53 OR Tet2)