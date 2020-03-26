import itertools

import numpy as np
import pandas as pd

from pharmpy.data import ColumnType
from pharmpy.math import conditional_joint_normal
from pharmpy.parameter_sampling import sample_from_covariance_matrix
from pharmpy.random_variables import VariabilityLevel
from pharmpy.results import Results


class FREMResults(Results):
    def __init__(self, frem_model, covariates):
        self.frem_model = frem_model
        n = 10
        parvecs = sample_from_covariance_matrix(frem_model, n=n)

        _, dist = list(frem_model.random_variables.distributions(level=VariabilityLevel.IIV))[-1]
        sigma_symb = dist.sigma

        df = frem_model.input.dataset
        df.pharmpy.column_type[covariates] = ColumnType.COVARIATE
        covariate_baselines = df.pharmpy.covariate_baselines
        cov_stdevs = covariate_baselines.std()[covariates]
        cov_means = covariate_baselines.mean()[covariates]
        cov_5th = covariate_baselines.quantile(0.05)[covariates]
        cov_95th = covariate_baselines.quantile(0.95)[covariates]

        ncovs = len(covariates)
        npars = sigma_symb.rows - ncovs
        param_indices = list(range(npars))
        scaling = np.diag(np.concatenate((np.ones(npars), cov_stdevs.values)))

        mu_bars_given_5th = np.empty((n, ncovs, npars))
        mu_bars_given_95th = np.empty((n, ncovs, npars))

        for sample_no, params in parvecs.iterrows():
            sigma = sigma_symb.subs(dict(params))
            sigma = np.array(sigma).astype(np.float64)
            scaled_sigma = scaling @ sigma @ scaling.T
            for i, cov in enumerate(covariates):
                indices = param_indices + [i + npars]
                cov_sigma = scaled_sigma[indices][:, indices]
                cov_mu = np.array([0] * npars + [cov_means[cov]])
                mu_bar_given_5th_cov, _ = conditional_joint_normal(cov_mu, cov_sigma,
                                                                   np.array([cov_5th[cov]]))
                mu_bar_given_95th_cov, _ = conditional_joint_normal(cov_mu, cov_sigma,
                                                                    np.array([cov_95th[cov]]))
                mu_bars_given_5th[sample_no, i, :] = mu_bar_given_5th_cov
                mu_bars_given_95th[sample_no, i, :] = mu_bar_given_95th_cov

        mu_bars_given_5th = np.exp(mu_bars_given_5th)
        mu_bars_given_95th = np.exp(mu_bars_given_95th)

        means_5th = np.mean(mu_bars_given_5th, axis=0)
        means_95th = np.mean(mu_bars_given_95th, axis=0)
        q5_5th = np.quantile(mu_bars_given_5th, 0.05, axis=0)
        q5_95th = np.quantile(mu_bars_given_95th, 0.05, axis=0)
        q95_5th = np.quantile(mu_bars_given_5th, 0.95, axis=0)
        q95_95th = np.quantile(mu_bars_given_95th, 0.95, axis=0)

        df = pd.DataFrame(columns=['parameter', 'covariate', 'condition', '5th', 'mean', '95th'])
        for param, cov in itertools.product(range(npars), range(ncovs)):
            df = df.append({'parameter': param, 'covariate': covariates[cov], 'condition': '5th',
                            '5th': q5_5th[cov, param], 'mean': means_5th[cov, param],
                            '95th': q95_5th[cov, param]}, ignore_index=True)
            df = df.append({'parameter': param, 'covariate': covariates[cov], 'condition': '95th',
                            '5th': q5_95th[cov, param], 'mean': means_95th[cov, param],
                            '95th': q95_95th[cov, param]}, ignore_index=True)
        self.covariate_effects = df
        print(df)
