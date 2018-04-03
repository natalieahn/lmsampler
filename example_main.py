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
  sampler.analyze_params(results, f, threshold=.1, max_order=2)
