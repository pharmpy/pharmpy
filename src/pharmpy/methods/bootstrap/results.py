import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

import pharmpy.visualization
from pharmpy.methods.psn_helpers import cmd_line_model_path, model_paths
from pharmpy.model_factory import Model
from pharmpy.results import Results


class BootstrapResults(Results):
    rst_path = Path(__file__).parent / 'report.rst'

    # FIXME: Should inherit from results that take multiple runs like bootstrap, cdd etc.
    def __init__(self, parameter_statistics=None, parameter_distribution=None,
                 covariance_matrix=None, ofv_distribution=None, included_individuals=None,
                 ofvs=None, base_ofv=None, parameter_estimates=None):
        self.parameter_statistics = parameter_statistics
        self.parameter_distribution = parameter_distribution
        self.covariance_matrix = covariance_matrix
        self.ofv_distribution = ofv_distribution
        self.included_individuals = included_individuals
        self.ofvs = ofvs
        self.base_ofv = base_ofv
        self.parameter_estimates = parameter_estimates

    def add_plots(self):
        self.ofv_plot = self.plot_ofv()
        self.parameter_estimates_correlation_plot = self.plot_parameter_estimates_correlation()
        self.base_ofv_plot = self.plot_base_ofv()

    def plot_ofv(self):
        plot = pharmpy.visualization.histogram(self.ofvs['ofv'], title='Bootstrap OFV')
        return plot

    def plot_base_ofv(self):
        ofvs = self.ofvs
        dofvs = ofvs['base_ofv'] - ofvs['ofv']
        dofvs.sort_values(inplace=True)
        quantiles = np.linspace(0.0, 1.0, num=len(dofvs))
        degrees = len(self.parameter_distribution)
        chi2_dist = scipy.stats.chi2(df=degrees)
        chi2 = chi2_dist.ppf(quantiles)
        df = pd.DataFrame({'Bootstrap': dofvs, 'quantiles': quantiles,
                           f'Reference χ²({degrees})': chi2})
        plot = pharmpy.visualization.line_plot(df, 'quantiles', xlabel='Distribution quantiles',
                                               ylabel='dOFV', legend_title='Distribution',
                                               title='OFV original model - OFV bootstrap model')
        return plot

    def plot_parameter_estimates_correlation(self):
        pe = self.parameter_estimates
        plot = pharmpy.visualization.scatter_matrix(pe)
        return plot


def calculate_results(bootstrap_models, original_model, included_individuals=None):
    results = [m.modelfit_results for m in bootstrap_models if m.modelfit_results is not None]
    if original_model:
        original_results = original_model.modelfit_results
    else:
        original_results = None

    if original_results is None:
        warnings.warn('No results for the base model could be read. Cannot calculate bias and '
                      'base_ofv')

    df = pd.DataFrame()
    for res in results:
        df = df.append(res.parameter_estimates, ignore_index=True, sort=False)
    df = df.reindex(results[0].parameter_estimates.index, axis=1)
    parameter_estimates = df.reset_index(drop=True)

    covariance_matrix = df.cov()

    df = parameter_estimates.copy()
    mean = df.mean()
    if original_results is not None:
        orig = original_results.parameter_estimates
        bias = mean - orig
    else:
        bias = np.nan
    statistics = pd.DataFrame({'mean': mean, 'median': df.median(),
                               'bias': bias, 'stderr': df.std()})
    statistics['RSE'] = statistics['stderr'] / statistics['mean']

    df = parameter_estimates
    distribution = create_distribution(df)

    boot_ofvs = [x.ofv for x in results]
    ofvs = pd.Series(boot_ofvs, name='ofv')
    ofvs = pd.DataFrame(ofvs)
    if original_results and included_individuals:
        ofvs['base_ofv'] = np.nan
        base_iofv = original_results.individual_ofv
        for i, included in enumerate(included_individuals):
            base_ofv = base_iofv[included].sum()
            ofvs.at[i, 'base_ofv'] = base_ofv

    ofv_dist = create_distribution(ofvs)

    if original_results is not None:
        base_ofv = original_results.ofv
    else:
        base_ofv = None

    res = BootstrapResults(covariance_matrix=covariance_matrix, parameter_statistics=statistics,
                           parameter_distribution=distribution, ofv_distribution=ofv_dist,
                           included_individuals=included_individuals, ofvs=ofvs,
                           base_ofv=base_ofv,
                           parameter_estimates=parameter_estimates)

    return res


def create_distribution(df):
    dist = pd.DataFrame({'min': df.min(), '0.05%': df.quantile(0.0005),
                         '0.5%': df.quantile(0.005), '2.5%': df.quantile(0.025),
                         '5%': df.quantile(0.05), 'median': df.median(),
                         '95%': df.quantile(0.95), '97.5%': df.quantile(0.975),
                         '99.5%': df.quantile(0.995), '99.95%': df.quantile(0.9995),
                         'max': df.max()})
    return dist


def psn_bootstrap_results(path):
    """ Create bootstrapresults from a PsN bootstrap run

        :param path: Path to PsN boostrap run directory
        :return: A :class:`BootstrapResults` object
    """
    path = Path(path)

    models = [Model(p) for p in model_paths(path, 'bs_pr1_*.mod')]
    # Read the results already now to give an appropriate error if no results exists
    results = [m.modelfit_results for m in models if m.modelfit_results is not None]
    if not results:
        raise FileNotFoundError("No model results available in m1")
    base_model = Model(cmd_line_model_path(path))

    incinds = pd.read_csv(path / 'included_individuals1.csv', header=None).values.tolist()
    res = calculate_results(models, base_model, included_individuals=incinds)
    return res
