from pathlib import Path

import numpy as np
import pandas as pd

import pharmpy.visualization
from pharmpy.methods.psn_helpers import cmd_line_model_path, model_paths
from pharmpy.model_factory import Model
from pharmpy.results import Results


class BootstrapResults(Results):
    rst_path = Path(__file__).parent / 'report.rst'

    # FIXME: Should inherit from results that take multiple runs like bootstrap, cdd etc.
    def __init__(self, parameter_statistics=None, parameter_distribution=None,
                 covariance_matrix=None, ofv_distribution=None, included_individuals=None,
                 ofvs=None, base_ofv=None):
        self.parameter_statistics = parameter_statistics
        self.parameter_distribution = parameter_distribution
        self.covariance_matrix = covariance_matrix
        self.ofv_distribution = ofv_distribution
        self.included_individuals = included_individuals
        self.ofvs = ofvs
        self.base_ofv = base_ofv

    def add_plots(self):
        self.ofv_plot = self.plot_ofv()
        self.delta_base_ofv_plot = self.plot_delta_base_ofv()

    def plot_ofv(self):
        plot = pharmpy.visualization.histogram(self.ofvs['ofv'], title='Bootstrap OFV')
        return plot

    def plot_delta_base_ofv(self):
        df = self.ofvs.copy()
        df['dOFV'] = df['base_ofv'] - self.base_ofv
        plot = pharmpy.visualization.histogram(df['dOFV'], title='sum(base iOFV) - base_ofv')
        return plot


def calculate_results(bootstrap_models, original_model, included_individuals=None):
    results = [m.modelfit_results for m in bootstrap_models if m.modelfit_results is not None]
    original_results = original_model.modelfit_results

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
    if original_model and included_individuals:
        ofvs['base_ofv'] = np.nan
        base_iofv = original_model.modelfit_results.individual_ofv
        for i, included in enumerate(included_individuals):
            base_ofv = base_iofv[included].sum()
            ofvs.at[i, 'base_ofv'] = base_ofv

    ofv_dist = create_distribution(ofvs)

    res = BootstrapResults(covariance_matrix=covariance_matrix, parameter_statistics=statistics,
                           parameter_distribution=distribution, ofv_distribution=ofv_dist,
                           included_individuals=included_individuals, ofvs=ofvs,
                           base_ofv=original_model.modelfit_results.ofv)

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
    base_model = Model(cmd_line_model_path(path))
    incinds = pd.read_csv(path / 'included_individuals1.csv', header=None).values.tolist()
    res = calculate_results(models, base_model, included_individuals=incinds)
    return res
