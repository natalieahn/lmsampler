# Linear Model Sampler (lmsampler)

This package contains methods to sample permutations of model parameters for OLS regression, plot the distribution of results across sampled models, and analyze the consequences of selecting certain parameters that produce sizable changes to the regression estimates
in comparison to when those parameters are not included.

Author: Natalie Ahn
Updated: April 2018

### Usage:

These tools can be used on any dataset stored as a pandas dataframe. The model sampling allows for the specification of both
required and optional control variables, fixed effects factors, clustering factor(s) for standard errors (only one will be chosen),
and variable suffixes (which can be used to select different sets of X variables, such as to randomly sample the lag period for time
series data, if the X variables have already been lagged and given different suffixes for each lag in the dataframe).

![alt text](https://github.com/natalieahn/lmsamplr/blob/master/lmsamplr_example_plot.png)

### Public methods:

fit_permute(data, y_var, X_vars, ...):	Takes a pandas dataframe, the name of a y_var,
											list of names of X_vars, optional lists of
											names of parameters to permute, and number of
											iterations, and returns a results dict object.

summarize(results, ...):				Takes a results dict object (from fit_permute)
											an optional list of X_var names (by default,
											all that appear in results will be used), and
											an optional output file object (by default,
											stdio will be used), and prints summary results.

plot(results, filepath, ...):			Takes a results dict object (from fit_permute),
											a filepath to the desired output image file,
											and other optional arguments, and plots a set of
											figures showing the distribution of regression
											estimates across sampled models.

analyze_params(results, ...):			Takes a results dict object (from fit_permute),
											and optional other arguments including an output
											file object, and prints summary statistics for
											certain parameters that produce sizable differences
											in the regression estimates when those parameters
											are chosen vs. when they are not.
