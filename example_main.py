# Example code for using the methods in the LMSampler class
#
# Natalie Ahn
# April 2018

import pandas as pd
from lmsampler import LMSampler
sampler = LMSampler()

data = pd.read_csv(...)
y_var = 'outcome'
X_vars = ['treatment1', 'treatment2']
controls_optional = ['age', 'income', 'education']
fixed_use = ['year']
fixed_optional = ['household']
clusters_optional = ['city', 'state']
lag_suffixes = ['_lag%dmo' % mo for mo in range(1, 13)]
n_iter = 1000

results = sampler.fit_permute(data, y_var, X_vars,
                             controls_optional=controls_optional, fixed_use=fixed_use,
                             fixed_optional=fixed_optional, clusters_optional=clusters_optional,
                             X_suffixes=lag_suffixes, n_iter = n_iter)

with open('test_results_summary.txt', 'w') as f:
  sampler.summarize(results, f)

sampler.plot(results, 'test_results_figure.png')

with open('test_results_byparams.txt', 'w') as f:
  sampler.analyze_params(results, f, threshold=.1, max_interact=2)

# Notes: The .analyze_params method compares the distribution of regression estimates in models
# with the given parameter, to models without that parameter. Parameters are named by the type
# and specific value, e.g.:
#      "control_var(income)" indicates that income is included as a control variable,
#      "fixed_effects(city)" indicates that city fixed effects have been included,
#      "clustered_se(state)" indicates that standard errors have been clustered by state.

# The printed results show how the estimated coefficient on the given X variable differs between
# models with the given parameter chosen versus models without it. If the average coefficient
# change shown is positive, the coefficients on the X variable are greater when the parameter
# is selected than when it is not. The overlap score shows what percentage of the distribution
# of model coefficients overlap across models with the given parameter vs models without it.

# The argument max_interact refers to the highest order of dependencies (or number of interacting
# parameters) that the method searches for, to find combinations that produce sizably different
# model estimates. Time complexity is exponential in max_interact (i.e. O(n^k) where n = number of
# parameters and k = max_interact), although some intuitive stopping criteria are included to try to
# keep the run time manageable for small numbers of parameters (e.g. dozens) and interactions
# (e.g. 2 or 3).
