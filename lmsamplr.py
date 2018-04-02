# LMSamplr
#
# Code to sample permutations of model parameters for OLS regression, 
# plot distribution of results, analyze parameter consequences.
#
# Natalie Ahn
# April 2018
#
# Public methods:
#
#	fit_permute(data, y_var, X_vars, ...):	Takes a pandas dataframe, the name of a y_var,
#											list of names of X_vars, optional lists of
#											names of parameters to permute, and number of
#											iterations, and returns a results dict object.
#
#	summarize(results, ...):				Takes a results dict object (from fit_permute)
#											an optional list of X_var names (by default,
#											all that appear in results will be used), and
#											an optional output file object (by default,
#											stdio will be used), and prints summary results.
#
#	plot(results, filepath, ...):			Takes a results dict object (from fit_permute),
#											a filepath to the desired output image file,
#											and other optional arguments, and plots a set of
#											figures showing the distribution of regression
#											estimates across sampled models.
#
#	analyze_params(results, ...):			Takes a results dict object (from fit_permute),
#											and optional other arguments including an output
#											file object, and prints summary statistics for
#											certain parameters that produce sizable differences
#											in the regression estimates when those parameters
#											are chosen vs. when they are not.

import re, csv, sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import sem, ttest_ind
from collections import Hashable, Iterable
from statistics import mean


class LMSamplr:

	def __init__(self):
		self.e = .1**10

	def fit_permute(self, data, y_var, X_vars, controls_use=[], controls_optional=[],
					fixed_use=[], fixed_optional=[], cluster_use=None, clusters_optional=[],
					X_suffixes=[], control_suffixes=[], max_controls=None, n_iter=1, ols_fit_kwargs={}):
		if max_controls is None: max_controls = len(controls_optional)
		param_names = ['X_suffix(%s)' % suffix for suffix in X_suffixes] \
					+ ['control_var(%s)' % var for var in controls_optional] \
					+ ['control_suffix(%s)' % suffix for suffix in control_suffixes] \
					+ ['fixed_factor(%s)' % factor for factor in fixed_optional] \
					+ ['cluster_factor(%s)' % factor for factor in clusters_optional]
		results = {'y_var':y_var, 'X_vars':X_vars,
				   'param_names':param_names, 'params':[],
				   'rsquared_adj':[], 'fvalue':[], 'coefs':[], 'bses':[], 'pvals':[]}
		for i in range(n_iter):
			data_model, clust, params = self._permute_params(data, y_var, X_vars,
									controls_use, controls_optional, fixed_use, fixed_optional,
						 			cluster_use, clusters_optional, X_suffixes, control_suffixes,
						 			max_controls)
			if data_model.shape[0] > 0:
				if clust: y,X,c = data_model.iloc[:,0], data_model.iloc[:,1:-1], data_model.iloc[:,-1]
				else: y,X = data_model.iloc[:,0], data_model.iloc[:,1:]
				corr = np.corrcoef(X, rowvar=False)
				if not np.isnan(corr[0][0]):
					X = np.concatenate([np.ones([X.shape[0],1]), X], axis=1)
					if clust: model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups':c})
					else: model = sm.OLS(y, X).fit(**ols_fit_kwargs)
					try: fvalue = model.fvalue[0,0]
					except (np.linalg.linalg.LinAlgError, ValueError) as e: fvalue = None
					results['params'].append(params)
					results['rsquared_adj'].append(model.rsquared_adj)
					results['fvalue'].append(fvalue)
					results['coefs'].append(model.params[1:1+len(X_vars)])
					results['bses'].append(model.bse[1:1+len(X_vars)])
					results['pvals'].append(model.pvalues[1:1+len(X_vars)])
		for stat in ['params','coefs','bses','pvals']:
			results[stat] = np.array(results[stat])
		return results

	def _permute_params(self, data, y_var, X_vars, controls_use, controls_optional,
						fixed_use, fixed_optional, cluster_use, clusters_optional,
						X_suffixes, control_suffixes, max_controls):
		params = []
		if X_suffixes:
			X_suffix = np.random.choice(X_suffixes)
			params += [1. if suffix == X_suffix else 0. for suffix in X_suffixes]
		else: X_suffix = ''
		X_vars_suffixed = [var + X_suffix for var in X_vars]
		if controls_optional:
			nvars = np.random.choice(range(max_controls+1))
			control_include = list(np.random.choice(controls_optional, nvars, replace=False))
			params += [1. if var in control_include else 0. for var in controls_optional]
			if control_suffixes:
				control_suffix = np.random.choice(control_suffixes)
				params += [1. if suffix == control_suffix else 0. for suffix in control_suffixes]
			else: control_suffix = ''
			control_include = controls_use + [var + control_suffix for var in control_include]
		else: control_include = controls_use
		if fixed_optional:
			nfixed = np.random.choice(range(len(fixed_optional)))
			fixed_include = fixed_use + list(np.random.choice(fixed_optional, nfixed, replace=False))
			params += [1. if factor in fixed_include else 0. for factor in fixed_optional]
		else: fixed_include = fixed_use
		if cluster_use: cluster_include = cluster_use
		elif clusters_optional:
			cluster_include = np.random.choice(clusters_optional)
			if len(data[cluster_include].astype('category').cat.categories) < 2:
				cluster_include = None
			params += [1. if factor == cluster_include else 0. for factor in clusters_optional]
		else: cluster_include = None
		data_model = data[[y_var] + X_vars_suffixed + control_include]
		for factor in fixed_include:
			data_fixed = pd.get_dummies(data[factor].astype('category'), prefix=factor)
			data_model = pd.concat([data_model, data_fixed.iloc[:,:-1]], axis=1)
		if cluster_include: data_model[cluster_include] = data[cluster_include]
		data_model = data_model.dropna(axis=0, how='any')
		return data_model, cluster_include, params

	def summarize(self, results, X_vars=None, out=sys.stdout):
		if not X_vars: X_vars = list(results['X_vars'])
		_ = out.write('***** RESULTS *****\n')
		_ = out.write('Summary of %d models per X var; y var: %s\n' \
					  % (len(results['params']), results['y_var']))
		_ = out.write('%40s %10s %10s %10s\n' % ('X var', 'Avg Coef', 'Avg SE', 'Avg Pval'))
		X_stats = {}
		for i,X_var in enumerate(X_vars):
			print_text = '%40s %10s %10s %10s' % (X_var, \
						  '%.4g' % np.nanmean(results['coefs'][:,i]),
						  '%.4g' % np.nanmean(results['bses'][:,i]),
						  '%.4g' % np.nanmean(results['pvals'][:,i]))
			for signif in [0.05, 0.01, 0.001]:
				if signif >= np.nanmean(results['pvals'][:,i]):
					print_text += '*'
				else: break
			print_text += '\n'
			_ = out.write(print_text)
		_ = out.write('\n')

	def plot(self, results, filepath, X_vars=None, X_labs=None, title=None,
			 nbins=50, figsize=None, colors=None, dpi=None, ci_legend=False):
		if not X_vars: X_vars = list(results['X_vars'])
		var_inds = {var:i for i,var in enumerate(results['X_vars'])}
		X_inds = [var_inds[X_var] for X_var in X_vars]
		if not X_labs: X_labs = X_vars
		colors = self._fix_colors(colors, len(X_vars))
		fig = plt.figure()
		if title: _ = plt.title(title)
		fig, subplts = plt.subplots(1, 3, sharey=True)
		self._plot_mean_coef_se(results, X_inds, X_labs, colors, figsize, ci_legend)
		self._plot_hists_coefs(results, X_inds, colors, nbins)
		self._plot_hists_pvals(results, X_inds, colors, nbins)
		if figsize is not None: fig.set_figwidth(figsize[0])
		if figsize is not None: fig.set_figheight(figsize[1])
		if dpi: plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
		else: _ = plt.savefig(filepath, bbox_inches='tight')
		plt.close()

	def _plot_mean_coef_se(self, results, X_inds, X_labs, colors, figsize, ci_legend):
		axes = np.array([0., 0., 0., float(len(X_inds))])
		ys = (np.array(range(len(X_inds))) + 0.5)[::-1]
		_ = plt.yticks(ys, X_labs)
		plt.subplot(1, 3, 1)
		_ = plt.title('Mean coefs\nw/ mean CIs', size='large')
		coefs = results['coefs'][:,X_inds]
		ses = results['bses'][:,X_inds]
		mean_coefs = np.array([np.mean(vals) for vals in coefs])
		_ = plt.plot([0,0], [0,axes[-1]], color='k', linewidth=.5)
		for i in range(len(X_inds)):
			_ = plt.plot(mean_coefs[i], ys[i], color=colors[i], marker='.', linestyle='')
			colors[i][-1] = 0.25
			for c,z in [('99%',2.576), ('95%',1.96), ('90%',1.645)]:
				ci = np.mean(ses[i]) * z
				if mean_coefs[i] - ci < axes[0]: axes[0] = mean_coefs[i] - ci
				if mean_coefs[i] + ci > axes[1]: axes[1] = mean_coefs[i] + ci
				_ = plt.errorbar(mean_coefs[i], ys[i], xerr=ci, color=colors[i], linestyle='', linewidth=3)
				colors[i][-1] += 0.25
		if ci_legend:
			for c,z,r in [('99%',2.576,'lightgrey'),('95%',1.96,'darkgrey'),('90%',1.645,'dimgrey')]:
				_ = plt.errorbar([0], [axes[-1]*2], xerr=[1], color=r, linewidth=3, linestyle='', label=c)
			_ = plt.legend(numpoints=1, loc=[-.5,.95], frameon=False, labelspacing=.05,
						   handletextpad=.1, fontsize='medium')
		_ = plt.xticks([axes[0],0.,axes[1]], ['%.1f'%axes[0],'0','%.1f'%axes[1]])
		_ = plt.yticks(ys, X_labs, size='large')
		axes[:2] *= 1.1
		_ = plt.axis(axes)
		if figsize is not None:
			adjust = .1 * max([len(part) for lab in X_labs for part in lab.split('\n')]) / figsize[0]
			_ = plt.subplots_adjust(left=adjust)

	def _plot_hists_coefs(self, results, X_inds, colors, nbins):
		nvars = len(X_inds)
		coefs = results['coefs'][:,X_inds]
		coefmin = np.min([np.min(coefs[:,]) for i in range(nvars)])
		coefmax = np.max([np.max(coefs[:,]) for i in range(nvars)])
		minmax = max([abs(coefmin), abs(coefmax)])
		for i in range(nvars):
			plt.subplot(nvars, 3, 3*i+2)
			if i == 0: _ = plt.title('Histogram of\nmodel coefs', size='large')
			_ = plt.hist(coefs[:,i], bins=np.linspace(-minmax, minmax, nbins), color=colors[i], edgecolor='none')
			_ = plt.xticks([-minmax*.75, 0, minmax*.75], ['%.1f'%(-minmax*.75),'0','%.1f'%(minmax*.75)])
			ax = plt.gca()
			_ = plt.plot([0,0], [0,ax.get_yticks()[-1]], color='k', linewidth=.5)
			_ = ax.set_yticks(self._round_ticks(ax.get_yticks(), 2))
			_ = ax.tick_params(axis='y', which='major', pad=1)
			_ = ax.tick_params(axis='x', which='major', pad=2)

	def _plot_hists_pvals(self, results, X_inds, colors, nbins):
		nvars = len(X_inds)
		pvals = results['pvals'][:,X_inds]
		for i in range(nvars):
			plt.subplot(nvars, 3, 3*(i+1))
			if i == 0: _ = plt.title('Histogram of\nmodel p-values', size='large')
			_ = plt.hist(pvals[:,i], bins=np.linspace(0., .99, nbins), color=colors[i], edgecolor='none')
			_ = plt.xticks(np.array(range(1,10))*.1, ['.%d' % n for n in range(1,10)])
			ax = plt.gca()
			_ = ax.set_yticks(self._skip_ticks(ax.get_yticks(), 3))
			_ = ax.tick_params(axis='y', which='major', pad=1)
			_ = ax.tick_params(axis='x', which='major', pad=2)
			_ = plt.subplots_adjust(left=-.1, bottom=-.3)

	def _fix_colors(self, colors, k):
		if colors is None:
			colors = [[0.6,0.0,0.0,1.0],[0.1,0.4,0.2,1.0],[0.0,0.0,0.7,1.0]]
		if not isinstance(colors[0], Iterable): colors = [colors]
		if len(colors) < k:
			n_orig, n_add = len(colors), k - len(colors)
			for i in range(n_add): colors.append(colors[i%n_orig])
		return np.array(colors)

	def _round_ticks(self, nums, maxnums=2):
		q2 = (nums[0] + nums[-1]) * .5
		q3 = (nums[0] + nums[-1]) * .75
		for unit in [500,100,50,10,5,1]:
			round_down = (q3 // unit) * unit
			if q2 <= round_down <= q3:
				return [nums[0], round_down]
		return [nums[0], nums[-1]]

	def _skip_ticks(self, nums, maxnums):
		while len(nums) > maxnums:
			nums = nums[::2]
		return nums

	def analyze_params(self, results, X_vars=None, out=sys.stdout, threshold=.05, max_order=1):
		if not X_vars: X_vars = list(results['X_vars'])
		_ = out.write('***** ANALYSIS OF PARAMETERS *****\n')
		_ = out.write('Summary of %d models per X var; y var: %s\n' \
					  % (len(results['params']), results['y_var']))
		_ = out.write('%40s %15s %15s\n' % ('Parameter', 'Avg Coef Chng', 'Distro Overlap'))
		nparams = len(results['param_names'])
		for i,X_var in enumerate(X_vars):
			_ = out.write('\X var: %s\n' % X_var)
			row_filter = [True for _ in range(len(results['params']))]
			self._analyze_params_recurs(results, i, [], row_filter, max_order, '', out, threshold)

	def _analyze_params_recurs(self, results, X_ind, params_in, prev_filter, max_order, prev_text, out, threshold):
		for j in range(len(results['param_names'])):
			print_text = prev_text
			if j not in params_in:
				row_filter = prev_filter & (results['params'][:,j] == 1.)
				if sum(row_filter):
					dif, overlap, concent = self._compare_distros(results, X_ind, row_filter)
					if len(params_in) == 0:
						print_text = '%40s ' % results['param_names'][j]
					else:
						if print_text[-1] != '\n': print_text += '\n'
						print_text += len(params_in) * '  '
						rem = 38 - 2 * len(params_in)
						print_text += ('+ %' + '%d' % rem + 's ') % results['param_names'][j]
					if overlap <= threshold:
						print_text += '%15s %15s\n' % ('%.4g'%dif, '%.4g'%overlap)
						_ = out.write(print_text)
					elif len(params_in) < max_order - 1 \
					and concent > 1:
						self._analyze_params_recurs(results, X_ind, params_in+[j], row_filter,
													max_order, print_text, out, threshold)

	def _compare_distros(self, results, X_ind, row_filter):
		rows_in = np.where(row_filter)[0]
		rows_out = np.where(1 - row_filter)[0]
		if len(rows_in) == 0 or len(rows_out) == 0: return (np.nan, np.nan)
		vals_in = results['coefs'][rows_in, X_ind]
		vals_out = results['coefs'][rows_out, X_ind]
		mean_in, mean_out = np.nanmean(vals_in), np.nanmean(vals_out)
		mean_dif = mean_in - mean_out
		min_in, max_in = np.nanmin(vals_in), np.nanmax(vals_in)
		min_out, max_out = np.nanmin(vals_out), np.nanmax(vals_out)
		overlap_in = [val for val in vals_in if min_out <= val <= max_out]
		overlap_out = [val for val in vals_out if min_in <= val <= max_in]
		overlap = (len(overlap_in)/len(vals_in) + len(overlap_out)/len(vals_out)) / 2
		concent = np.std(vals_in) / np.std(vals_out)
		return mean_dif, overlap, concent

