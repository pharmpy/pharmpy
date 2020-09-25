from pathlib import Path

import numpy as np
import pandas as pd

import pharmpy.visualization
from pharmpy.methods.psn_helpers import cmd_line_model_path, model_paths
from pharmpy.model_factory import Model
from pharmpy.results import Results


class BootstrapResults(Results):
    # FIXME: Should inherit from results that take multiple runs like bootstrap, cdd etc.
    def __init__(self, bootstrap_models, original_model=None):
        if original_model is None and bootstrap_models is None:
            # FIXME: this is a special case for now for json handling before we have modelfit json
            return
        if original_model is not None:
            self._original_results = original_model.modelfit_results
        else:
            self._original_results = None
        self._bootstrap_results = [m.modelfit_results for m in bootstrap_models
                                   if m.modelfit_results is not None]
        self._total_number_of_models = len(bootstrap_models)

    @property
    def ofv(self):
        boot_ofvs = [x.ofv for x in self._bootstrap_results]
        return pd.Series(boot_ofvs, name='ofv')

    @property
    def parameter_estimates(self):
        df = pd.DataFrame()
        for res in self._bootstrap_results:
            df = df.append(res.parameter_estimates, ignore_index=True, sort=False)
        df = df.reindex(self._bootstrap_results[0].parameter_estimates.index, axis=1)
        df = df.reset_index(drop=True)
        return df

    @property
    def standard_errors(self):
        pass
        # FIXME: Continue here

    @property
    def statistics(self):
        try:
            return self._statistics
        except AttributeError:
            pass
        df = self.parameter_estimates
        ofvs = self.ofv
        df.insert(0, 'OFV', ofvs)
        mean = df.mean()
        if self._original_results is not None:
            orig = self._original_results.parameter_estimates
            ofv_ser = pd.Series({'OFV': self._original_results.ofv})
            orig = pd.concat([ofv_ser, orig])
            bias = mean - orig
        else:
            bias = np.nan
        summary = pd.DataFrame({'mean': mean, 'bias': bias, 'stderr': df.std()})
        self._statistics = summary
        return summary

    @property
    def distribution(self):
        try:
            return self._distribution
        except AttributeError:
            pass
        df = self.parameter_estimates
        ofvs = self.ofv
        df.insert(0, 'OFV', ofvs)
        summary = pd.DataFrame({'min': df.min(), '0.05%': df.quantile(0.0005),
                                '0.5%': df.quantile(0.005), '2.5%': df.quantile(0.025),
                                '5%': df.quantile(0.05), 'median': df.median(),
                                '95%': df.quantile(0.95), '97.5%': df.quantile(0.975),
                                '99.5%': df.quantile(0.995), '99.95%': df.quantile(0.9995),
                                'max': df.max()})
        self._distribution = summary
        return summary

    def __repr__(self):
        inclusions = f'Inclusion\n\nTotal number of models: {self._total_number_of_models}\n' \
                     f'Models not included because of failure: ' \
                     f'{self._total_number_of_models - len(self._bootstrap_results)}\n' \
                     f'Models included in analysis: {len(self._bootstrap_results)}'
        statistics = f'Statistics\n{repr(self.statistics)}'
        distribution = f'Distribution\n{repr(self.distribution)}'
        return f'{inclusions}\n\n{statistics}\n\n{distribution}'

    def to_dict(self):
        return {'statistics': self.statistics}

    def plot_ofv(self):
        plot = pharmpy.visualization.histogram(self.ofv, title='Bootstrap OFV')
        return plot


def psn_bootstrap_results(path):
    """ Create bootstrapresults from a PsN bootstrap run

        :param path: Path to PsN boostrap run directory
        :return: A :class:`BootstrapResults` object
    """
    path = Path(path)

    models = [Model(p) for p in model_paths(path, 'bs_pr1_*.mod')]
    base_model = Model(cmd_line_model_path(path))
    # FIXME: should have a calculate results function
    res = BootstrapResults(models, original_model=base_model)
    return res
