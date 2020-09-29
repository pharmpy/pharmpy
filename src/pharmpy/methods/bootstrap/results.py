from pathlib import Path

import numpy as np
import pandas as pd

import pharmpy.visualization
from pharmpy.methods.psn_helpers import cmd_line_model_path, model_paths
from pharmpy.model_factory import Model
from pharmpy.results import Results


class BootstrapResults(Results):
    # FIXME: Should inherit from results that take multiple runs like bootstrap, cdd etc.
    def __init__(self, parameter_statistics=None, parameter_distribution=None,
                 covariance_matrix=None, ofv_distribution=None):
        self.parameter_statistics = parameter_statistics
        self.parameter_distribution = parameter_distribution
        self.covariance_matrix = covariance_matrix
        self.ofv_distribution = ofv_distribution

    def plot_ofv(self):
        plot = pharmpy.visualization.histogram(self.ofv, title='Bootstrap OFV')
        return plot


def calculate_results(bootstrap_models, original_model):
    results = [m.modelfit_results for m in bootstrap_models if m.modelfit_results is not None]
    original_results = original_model.modelfit_results

    df = pd.DataFrame()
    for res in results:
        df = df.append(res.parameter_estimates, ignore_index=True, sort=False)
    df = df.reindex(results[0].parameter_estimates.index, axis=1)
    parameter_estimates = df.reset_index(drop=True)

    covariance_matrix = df.cov()

    boot_ofvs = [x.ofv for x in results]
    ofvs = pd.Series(boot_ofvs, name='ofv')

    df = pd.DataFrame(ofvs)
    ofv_dist = pd.DataFrame({'min': df.min(), '0.05%': df.quantile(0.0005),
                             '0.5%': df.quantile(0.005), '2.5%': df.quantile(0.025),
                             '5%': df.quantile(0.05), 'median': df.median(),
                             '95%': df.quantile(0.95), '97.5%': df.quantile(0.975),
                             '99.5%': df.quantile(0.995), '99.95%': df.quantile(0.9995),
                             'max': df.max()})

    df = parameter_estimates.copy()
    mean = df.mean()
    if original_results is not None:
        orig = original_results.parameter_estimates
        bias = mean - orig
    else:
        bias = np.nan
    statistics = pd.DataFrame({'mean': mean, 'median': df.median(),
                               'bias': bias, 'stderr': df.std()})

    df = parameter_estimates
    distribution = pd.DataFrame({'min': df.min(), '0.05%': df.quantile(0.0005),
                                 '0.5%': df.quantile(0.005), '2.5%': df.quantile(0.025),
                                 '5%': df.quantile(0.05), 'median': df.median(),
                                 '95%': df.quantile(0.95), '97.5%': df.quantile(0.975),
                                 '99.5%': df.quantile(0.995), '99.95%': df.quantile(0.9995),
                                 'max': df.max()})

    res = BootstrapResults(covariance_matrix=covariance_matrix, parameter_statistics=statistics,
                           parameter_distribution=distribution, ofv_distribution=ofv_dist)

    return res


def psn_bootstrap_results(path):
    """ Create bootstrapresults from a PsN bootstrap run

        :param path: Path to PsN boostrap run directory
        :return: A :class:`BootstrapResults` object
    """
    path = Path(path)

    models = [Model(p) for p in model_paths(path, 'bs_pr1_*.mod')]
    base_model = Model(cmd_line_model_path(path))
    res = calculate_results(models, base_model)
    return res
