# Linear Model Sampler (lmsampler)

This package contains methods to sample permutations of model parameters for OLS regression, plot the distribution of results across sampled models, and analyze the consequences of selecting certain parameters that produce sizable changes to the regression estimates
in comparison to when those parameters are not included.

Author: Natalie Ahn
Updated: April 2018

### Usage:

These tools can be used on any dataset stored as a pandas dataframe. The model sampling allows for the specification of both
required and optional control variables, fixed effects factors, clustering factor(s) for standard errors (only one will be chosen),
and variable suffixes (which can be used to select different sets of X variables, such as to randomly sample the lag period for time
series data, if the X variables have already been lagged and given different suffixes for each lag in the dataframe). The .plot method
plots a set of figures as shown below, with one row per X variable of interest, stacked for comparison. Each row in the figure includes
the mean coefficient with confidence interval bars, a histogram of coefficients across sampled models, and a histogram of p-values
across sampled models. In the figure below, the relationship between the Y variable and "X var 1" is estimated right around zero
(with p-values in the upper range because the probability of the null hypothesis is high), while the relationship for "X var 2"
is positive across all sample models with robust significance (over 90% of models had p-values below 0.05), and the relationship
for "X var 3" is negative across all sampled models with quite robust significance as well (over 75% of models had p-values
below 0.05).

![alt text](https://github.com/natalieahn/lmsamplr/blob/master/example_plot.png)

