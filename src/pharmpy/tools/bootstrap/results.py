import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

import pharmpy.visualization
from pharmpy.model_factory import Model
from pharmpy.results import ModelfitResults, Results
from pharmpy.tools.psn_helpers import cmd_line_model_path, model_paths


class BootstrapResults(Results):
    rst_path = Path(__file__).parent / 'report.rst'

    # FIXME: Should inherit from results that take multiple runs like bootstrap, cdd etc.
    def __init__(
        self,
        parameter_statistics=None,
        parameter_distribution=None,
        covariance_matrix=None,
        ofv_distribution=None,
        ofv_statistics=None,
        included_individuals=None,
        ofvs=None,
        parameter_estimates=None,
        ofv_plot=None,
        parameter_estimates_correlation_plot=None,
        dofv_quantiles_plot=None,
        parameter_estimates_histogram=None,
    ):
        self.parameter_statistics = parameter_statistics
        self.parameter_distribution = parameter_distribution
        self.covariance_matrix = covariance_matrix
        self.ofv_distribution = ofv_distribution
        self.ofv_statistics = ofv_statistics
        self.included_individuals = included_individuals
        self.ofvs = ofvs
        self.parameter_estimates = parameter_estimates
        self.ofv_plot = ofv_plot
        self.parameter_estimates_correlation_plot = parameter_estimates_correlation_plot
        self.dofv_quantiles_plot = dofv_quantiles_plot
        self.parameter_estimates_histogram = parameter_estimates_histogram

    def add_plots(self):
        self.ofv_plot = self.plot_ofv()
        self.parameter_estimates_correlation_plot = self.plot_parameter_estimates_correlation()
        self.dofv_quantiles_plot = self.plot_dofv_quantiles()
        self.parameter_estimates_histogram = self.plot_parameter_estimates_histogram()

    def plot_ofv(self):
        plot = pharmpy.visualization.histogram(
            self.ofvs['bootstrap_bootdata_ofv'], title='Bootstrap OFV'
        )
        return plot

    def plot_dofv_quantiles(self):
        ofvs = self.ofvs
        dofvs = ofvs['delta_bootdata'].sort_values().reset_index(drop=True)
        dofvs_boot_base = ofvs['delta_origdata'].sort_values().reset_index(drop=True)
        quantiles = np.linspace(0.0, 1.0, num=len(dofvs))
        degrees = len(self.parameter_distribution)
        chi2_dist = scipy.stats.chi2(df=degrees)
        chi2 = chi2_dist.ppf(quantiles)
        degrees_dofvs = self.ofv_statistics['mean']['delta_bootdata']
        degrees_boot_base = self.ofv_statistics['mean']['delta_origdata']
        df_dict = {'quantiles': quantiles, f'Reference χ²({degrees})': chi2}
        if not np.isnan(degrees_dofvs):
            df_dict[
                (
                    'Original model OFV - Bootstrap model OFV (both using bootstrap datasets)',
                    f'Estimated df = {degrees_dofvs:.2f}',
                )
            ] = dofvs
        if not np.isnan(degrees_boot_base):
            df_dict[
                (
                    'Bootstrap model OFV - Original model OFV (both using original dataset)',
                    f'Estimated df = {degrees_boot_base:.2f}',
                )
            ] = dofvs_boot_base
        df = pd.DataFrame(df_dict)
        plot = pharmpy.visualization.line_plot(
            df,
            'quantiles',
            xlabel='Distribution quantiles',
            ylabel='dOFV',
            legend_title='Distribution',
            title='dOFV quantiles',
        )
        return plot

    def plot_parameter_estimates_correlation(self):
        pe = self.parameter_estimates
        plot = pharmpy.visualization.scatter_matrix(pe)
        return plot

    def plot_parameter_estimates_histogram(self):
        pe = self.parameter_estimates
        plot = pharmpy.visualization.facetted_histogram(pe)
        return plot


def calculate_results(
    bootstrap_models, original_model=None, included_individuals=None, dofv_results=None
):
    results = [m.modelfit_results for m in bootstrap_models if m.modelfit_results is not None]
    if original_model:
        original_results = original_model.modelfit_results
    else:
        original_results = None

    if original_results is None:
        warnings.warn(
            'No results for the base model could be read. Cannot calculate bias and '
            'original_bootdata_ofv'
        )

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
    statistics = pd.DataFrame(
        {'mean': mean, 'median': df.median(), 'bias': bias, 'stderr': df.std()}
    )
    statistics['RSE'] = statistics['stderr'] / statistics['mean']

    df = parameter_estimates
    distribution = create_distribution(df)

    boot_ofvs = [x.ofv for x in results]
    ofvs = pd.Series(boot_ofvs, name='bootstrap_bootdata_ofv')
    ofvs = pd.DataFrame(ofvs)
    ofvs['original_bootdata_ofv'] = np.nan
    ofvs['bootstrap_origdata_ofv'] = np.nan
    if original_results:
        base_iofv = original_results.individual_ofv
        if included_individuals and base_iofv is not None:
            for i, included in enumerate(included_individuals):
                base_ofv = base_iofv[included].sum()
                ofvs.at[i, 'original_bootdata_ofv'] = base_ofv

    if dofv_results is not None:
        for i, res in enumerate(dofv_results):
            if res is not None:
                ofvs.at[i, 'bootstrap_origdata_ofv'] = res.ofv

    if original_results is not None:
        base_ofv = original_results.ofv
        ofvs['original_origdata_ofv'] = base_ofv

    ofvs['delta_bootdata'] = ofvs['original_bootdata_ofv'] - ofvs['bootstrap_bootdata_ofv']

    if original_results is not None:
        ofvs['delta_origdata'] = ofvs['bootstrap_origdata_ofv'] - base_ofv
    else:
        ofvs['delta_origdata'] = np.nan

    with warnings.catch_warnings():
        # Catch numpy warnings beause of NaN in ofvs
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        ofv_dist = create_distribution(ofvs)
        ofv_stats = pd.DataFrame(
            {'mean': ofvs.mean(), 'median': ofvs.median(), 'stderr': ofvs.std()}
        )

    res = BootstrapResults(
        covariance_matrix=covariance_matrix,
        parameter_statistics=statistics,
        parameter_distribution=distribution,
        ofv_distribution=ofv_dist,
        ofv_statistics=ofv_stats,
        included_individuals=included_individuals,
        ofvs=ofvs,
        parameter_estimates=parameter_estimates,
    )

    return res


def create_distribution(df):
    dist = pd.DataFrame(
        {
            'min': df.min(),
            '0.05%': df.quantile(0.0005),
            '0.5%': df.quantile(0.005),
            '2.5%': df.quantile(0.025),
            '5%': df.quantile(0.05),
            'median': df.median(),
            '95%': df.quantile(0.95),
            '97.5%': df.quantile(0.975),
            '99.5%': df.quantile(0.995),
            '99.95%': df.quantile(0.9995),
            'max': df.max(),
        }
    )
    return dist


def psn_bootstrap_results(path):
    """Create bootstrapresults from a PsN bootstrap run

    :param path: Path to PsN boostrap run directory
    :return: A :class:`BootstrapResults` object
    """
    path = Path(path)

    models = [Model(p) for p in model_paths(path, 'bs_pr1_*.mod')]
    # Read the results already now to give an appropriate error if no results exists
    results = [m.modelfit_results for m in models if m.modelfit_results is not None]
    if not results:
        raise FileNotFoundError("No model results available in m1")
    try:
        base_model = Model(cmd_line_model_path(path))
    except FileNotFoundError:
        base_model = None

    # Read dOFV results in NONMEM specific way. Models have multiple $PROBLEM
    # Create proper result objects to pass to calculate_results
    dofv_results = None
    if (path / 'm1' / 'dofv_1.mod').is_file():
        from pharmpy.plugins.nonmem.table import NONMEMTableFile

        dofv_results = []
        for table_path in (path / 'm1').glob('dofv_*.ext'):
            table_file = NONMEMTableFile(table_path)
            next_table = 1
            for table in table_file:
                while next_table != table.number:
                    dofv_results.append(None)
                    next_table += 1
                res = ModelfitResults(ofv=table.final_ofv)
                dofv_results.append(res)
                next_table += 1

    incinds = pd.read_csv(path / 'included_individuals1.csv', header=None).values.tolist()
    res = calculate_results(
        models, original_model=base_model, included_individuals=incinds, dofv_results=dofv_results
    )
    return res
