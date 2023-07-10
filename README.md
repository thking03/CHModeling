# CHModeling
Baseline modeling for clonal hematopoiesis

## 7/10/23 - present :: Competitive Lotka-Volterra w/ shared carryign cap
- Optimizing a competitive Lotka-Volterra model using data
    - Model takes the form: $\vec{N}'(t) = \vec{r}\circ\vec{N}(t)\circ(\vec{1}-\frac{\textbf{A}\vec{N}(t)}{K})$
    - (Optimizable) parameters are stored in the interaction matrix $\textbf{A}$ and the growth-rate vector $\vec{r}$
    - $\vec{a}\circ\vec{b}$ represents the Hadamard product of the two vectors (element-wise vector multiplication)
- Using Python in `CLVModel.py` to perform numeric integration, with fitting code still contained in `OptModel.py`

## 7/3/23 - 7/10/23 :: Multivariable logistic w/ shared carrying cap
- Optimizes a naive (simple) logistic model using data
    - Model takes the form: $\vec{N}'(t) = \textbf{A}\vec{N}(t)(1-\frac{\vec{1}\cdot\vec{N}(t)}{K})$
    - (optimizable) parameters are stored in the interaction matrix $\textbf{A}$, where the principal diagonal (self-interaction) regulate growth rates
- Using Python in `NaiveModel.py` to perform numeric integration
- `OptModel.py` contains code used to "fit" parameters to the differential model (solved numerically)
    - The function `naiveopt()` takes in a single data point (equilibrium proportions of the population) and returns optimized parameters, losses, and the final ODE model
    - The function `naivemultiopt()` takes in multiple data points (equilibrium proportions of the population) and returns optimized parameters and losses

### Notes / Issues
- Should `naivemultiopt()` use the average of nudges (in dmat) rather than the sum when updating the parameters? -- slower learning rate, but maybe more accurate?
- Beginning work on new `multiopt()` function which admits the entire nxn matrix A, not just a diagonal matrix --> need to explore how to optimize the interaction coefficients, currently leaning toward something like $\text{sign}({e_{c1}})\sqrt{l_{c1}*l_{c2}}$
- `multiopt()` needs a slower learning rate since more things are changing
    - learning rate now initialized to .05
- Need to investigate an error in `multiopt()` that allows populations that begin at size 0 to spontaneously grow (has to do with how the interaction rates are defined)
    - addition of a "truth" matrix that prohibits populations from growing if they are zero in the initial conditions vector -- this doesn't account for populations that tend to zero over time, but has led to a error reduction of around two orders of magnitude

### Approximated Data & conditions from Katharina's slides for naive modeling
- mice sac'ed @ wk 14
- cisplatin admin from wk. 6-10 -- ignore in initial naive modeling because only endpoints available
- Have information about TP53 and Tet2 in PB and BM in the format (min, q1, median, q3, max) -- data manually collected and tabulated at `naivedata.csv`, so there is likely some small human error in these values 
- Data consists of 16 groups consisting of all combinations of (1X OR 2X) AND (TP53 OR Tet2)