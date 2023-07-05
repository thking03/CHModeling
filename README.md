# CHModeling
Baseline modeling for clonal hematopoiesis

## 7/3/23
- Currently working with estimated data on a naive (simple) logistic model
- Using Python in `NaiveModel.py` to perform numeric integration

### Approximated Data & conditions from Katharina's slides for naive modeling
- mice sac'ed @ wk 14
- cisplatin admin from wk. 6-10 -- ignore in initial naive modeling because only endpoints available
- Have information about TP53 and Tet2 in PB and BM in the format (min, q1, median, q3, max) -- data manually collected and tabulated at `naivedata.csv`, so there is likely some small human error in these values 
- Data consists of 16 groups consisting of all combinations of (1X OR 2X) AND (TP53 OR )