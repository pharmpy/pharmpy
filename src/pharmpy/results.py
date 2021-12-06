"""

.. list-table:: Options for the results module
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``native_shrinkage``
     - ``True``
     - bool
     - Should shrinkage calculation of external tool be used.
       Otherwise pharmpy will calculate shrinkage
"""

import copy
import json
import lzma
import math
from collections.abc import MutableSequence
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

import pharmpy.config as config
from pharmpy.math import cov2corr


class ResultsConfiguration(config.Configuration):
    module = 'pharmpy.results'
    """ FIXME: This setting should be moved to the NONMEM plugin
    """
    native_shrinkage = config.ConfigItem(True, 'Use shrinkage results from external tool')


conf = ResultsConfiguration()


class ResultsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            d = json.loads(obj.to_json(orient='table'))
            if isinstance(obj, pd.DataFrame):
                d['__class__'] = 'DataFrame'
            elif isinstance(obj, pd.Series):
                d['__class__'] = 'Series'
            else:
                d['__class__'] = obj.__class__.__name__
            return d
        elif obj.__class__.__module__.startswith('altair.'):
            d = obj.to_dict()
            d['__class__'] = 'vega-lite'
            return d
        else:
            return json.JSONEncoder.encode(self, obj)


class ResultsJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        self.cls = None

    def object_hook(self, dct):
        if '__class__' in dct:
            cls = dct['__class__']
            del dct['__class__']
            if cls == 'DataFrame' or cls == 'Series':
                res = pd.read_json(json.dumps(dct), orient='table')
                return res
            elif cls == 'vega-lite':
                res = alt.Chart.from_dict(dct)
                return res
            else:
                self.cls = cls
        return dct


def read_results(path_or_buf):
    if '{' in str(path_or_buf):  # Heuristic to determine if path or buffer
        s = path_or_buf
    else:
        path = Path(path_or_buf)
        if path.is_dir():
            path /= 'results.json'
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.name.endswith('.xz'):
            with lzma.open(path, 'r') as json_file:
                s = json_file.read().decode('utf-8')
        else:
            with open(path, 'r') as json_file:
                s = json_file.read()
    decoder = ResultsJSONDecoder()
    d = decoder.decode(s)
    if decoder.cls == 'FREMResults':
        from pharmpy.tools.frem import FREMResults

        res = FREMResults.from_dict(d)
    elif decoder.cls == 'BootstrapResults':
        from pharmpy.tools.bootstrap import BootstrapResults

        res = BootstrapResults.from_dict(d)
    elif decoder.cls == 'CDDResults':
        from pharmpy.tools.cdd import CDDResults

        res = CDDResults.from_dict(d)
    elif decoder.cls == 'SCMResults':
        from pharmpy.tools.scm import SCMResults

        res = SCMResults.from_dict(d)
    elif decoder.cls == 'QAResults':
        from pharmpy.tools.qa import QAResults

        res = QAResults.from_dict(d)
    elif decoder.cls == 'LinearizeResults':
        from pharmpy.tools.linearize import LinearizeResults

        res = LinearizeResults.from_dict(d)
    elif decoder.cls == 'ResmodResults':
        from pharmpy.tools.resmod import ResmodResults

        res = ResmodResults.from_dict(d)
    elif decoder.cls == 'SimevalResults':
        from pharmpy.tools.simeval import SimevalResults

        res = SimevalResults.from_dict(d)
    elif decoder.cls == 'CrossvalResults':
        from pharmpy.tools.crossval import CrossvalResults

        res = CrossvalResults.from_dict(d)

    return res


class Results:
    """Base class for all result classes"""

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_json(self, path=None, lzma=False):
        json_dict = self.to_dict()
        json_dict['__class__'] = self.__class__.__name__
        s = json.dumps(json_dict, cls=ResultsJSONEncoder)
        if path:
            if not lzma:
                with open(path, 'w') as fh:
                    fh.write(s)
            else:
                xz_path = path.parent / (path.name + '.xz')
                with lzma.open(xz_path, 'w') as fh:
                    fh.write(bytes(s, 'utf-8'))
        else:
            return s

    def get_and_reset_index(self, attr, **kwargs):
        """Wrapper to reset index of attribute or result from method.

        Used to facilitate importing multiindex dataframes into R
        """
        val = getattr(self, attr)
        if callable(val):
            df = val(**kwargs)
        else:
            df = val
        return df.reset_index()

    def to_dict(self):
        """Convert results object to a dictionary"""
        return vars(self).copy()

    def __str__(self):
        start = self.__class__.__name__
        s = f'{start}\n\n'
        d = self.to_dict()
        for key, value in d.items():
            if value.__class__.__module__.startswith('altair.'):
                continue
            s += f'{key}\n'
            if isinstance(value, pd.DataFrame):
                s += value.to_string()
            elif isinstance(value, list):  # Print list of lists as table
                if len(value) > 0 and isinstance(value[0], list):
                    df = pd.DataFrame(value)
                    df_str = df.to_string(index=False)
                    df_str = df_str.split('\n')[1:]
                    s += '\n'.join(df_str)
            else:
                s += str(value) + '\n'
            s += '\n\n'
        return s

    def to_csv(self, path):
        """Save results as a human readable csv file

        Index will not be printed if it is a basic range.
        """
        d = self.to_dict()
        s = ""
        for key, value in d.items():
            if value.__class__.__module__.startswith('altair.'):
                continue
            s += f'{key}\n'
            if isinstance(value, pd.DataFrame):
                if isinstance(value.index, pd.RangeIndex):
                    use_index = False
                else:
                    use_index = True
                s += value.to_csv(index=use_index)
            elif isinstance(value, pd.Series):
                s += value.to_csv()
            elif isinstance(value, list):  # Print list of lists as table
                if len(value) > 0 and isinstance(value[0], list):
                    for row in value:
                        s += f'{",".join(map(str, row))}\n'
            else:
                s += str(value) + '\n'
            s += '\n'
        with open(path, 'w', newline='') as fh:
            print(s, file=fh)

    def create_report(self, path):
        import pharmpy.reporting.reporting as reporting

        reporting.generate_report(self.rst_path, path)

    def add_plots(self):
        """Create and add all plots to results object"""
        raise NotImplementedError()


class ModelfitResults(Results):
    """Base class for results from a modelfit operation

    model_name - name of model that generated the results model

    Attributes
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix of the population parameter estimates
    covariance_matrix : pd.DataFrame
        Covariance matrix of the population parameter estimates
    information_matrix : pd.DataFrame
        Fischer information matrix of the population parameter estimates
    evaluation_ofv : float
        The objective function value as if the model was evaluated. Currently
        workfs for classical estimation methods by taking the OFV of the first
        iteration.
    individual_ofv : pd.Series
        OFV for each individual
    individual_estimates : pd.DataFrame
        Estimates for etas
    individual_estimates_covariance : pd.Series
        Estimated covariance between etas
    parameter_estimates : pd.Series
        Population parameter estimates
    parameter_estimates_sdcorr : pd.Series
        Population parameter estimates with variability parameters as standard deviations and
        correlations
    residuals: pd.DataFrame
        Table of various residuals
    runtime_total : float
        Total runtime of estimation
    standard_errors : pd.Series
        Standard errors of the population parameter estimates
    standard_errors_sdcorr : pd.Series
        Standard errors of the population parameter estimates on standard deviation and correlation
        scale
    """

    def __init__(
        self,
        ofv=None,
        parameter_estimates=None,
        parameter_estimates_sdcorr=None,
        covariance_matrix=None,
        correlation_matrix=None,
        information_matrix=None,
        standard_errors=None,
        minimization_successful=None,
        individual_ofv=None,
        individual_estimates=None,
        individual_estimates_covariance=None,
        residuals=None,
        runtime_total=None,
    ):
        self.ofv = ofv
        self.parameter_estimates = parameter_estimates
        self.parameter_estimates_sdcorr = parameter_estimates_sdcorr
        self.covariance_matrix = covariance_matrix
        self.correlation_matrix = correlation_matrix
        self.standard_errors = standard_errors
        self._minimization_successful = minimization_successful
        self.individual_estimates = individual_estimates
        self.individual_ofv = individual_ofv
        self.residuals = residuals
        self.runtime_total = runtime_total

    def __bool__(self):
        return bool(self.ofv) and bool(self.parameter_estimates)

    def to_dict(self):
        """Convert results object to a dictionary"""
        return {'parameter_estimates': self.parameter_estimates}

    @property
    def aic(self):
        """Final AIC value assuming the OFV to be -2LL"""
        parameters = self.model.parameters.copy()
        parameters.remove_fixed()
        return self.ofv + 2 * len(parameters)

    @property
    def bic(self):
        """Final BIC value assuming the OFV to be -2LL"""
        parameters = self.model.parameters.copy()
        parameters.remove_fixed()
        return self.ofv + len(parameters) * math.log(len(self.model.dataset.pharmpy.observations))

    @property
    def minimization_successful(self):
        """Was the minimization successful"""
        return self._minimization_successful

    def _cov_from_inf(self):
        Im = self.information_matrix
        cov = pd.DataFrame(np.linalg.inv(Im.values), index=Im.index, columns=Im.columns)
        return cov

    def _cov_from_corrse(self):
        se = self.standard_errors
        corr = self.correlation_matrix
        x = se.values @ se.values.T
        cov = x * corr.values
        cov_df = pd.DataFrame(cov, index=corr.index, columns=corr.columns)
        return cov_df

    def _inf_from_cov(self):
        C = self.covariance_matrix
        Im = pd.DataFrame(np.linalg.inv(C.values), index=C.index, columns=C.columns)
        return Im

    def _inf_from_corrse(self):
        se = self.standard_errors
        corr = self.correlation_matrix
        x = se.values @ se.values.T
        cov = x * corr.values
        Im = pd.DataFrame(np.linalg.inv(cov), index=corr.index, columns=corr.columns)
        return Im

    def _corr_from_cov(self):
        C = self.covariance_matrix
        corr = pd.DataFrame(cov2corr(C.values), index=C.index, columns=C.columns)
        return corr

    def _corr_from_inf(self):
        Im = self.information_matrix
        corr = pd.DataFrame(cov2corr(np.linalg.inv(Im.values)), index=Im.index, columns=Im.columns)
        return corr

    @property
    def relative_standard_errors(self):
        """Relative standard errors of population parameter estimates"""
        if self.standard_errors is not None:
            ser = self.standard_errors / self.parameter_estimates
            ser.name = 'RSE'
            return ser

    def _se_from_cov(self):
        """Calculate the standard errors from the covariance matrix
        can be used by subclasses
        """
        cov = self.covariance_matrix
        se = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
        return se

    def _se_from_inf(self):
        """Calculate the standard errors from the information matrix"""
        Im = self.information_matrix
        se = pd.Series(np.sqrt(np.linalg.inv(Im.values)), index=Im.index)
        return se

    def near_bounds(self, zero_limit=0.001, significant_digits=2):
        return self.model.parameters.is_close_to_bound(
            values=self.parameter_estimates,
            zero_limit=zero_limit,
            significant_digits=significant_digits,
        )

    def parameter_summary(self):
        """Summary of parameter estimates and uncertainty"""
        pe = self.parameter_estimates
        ses = self.standard_errors
        rses = self.relative_standard_errors
        df = pd.DataFrame({'estimate': pe, 'SE': ses, 'RSE': rses})
        return df

    def __repr__(self):
        df = self.parameter_summary()
        return df.to_string()


class ChainedModelfitResults(MutableSequence, ModelfitResults):
    """A sequence of modelfit results given in order from first to final
    inherits from both list and ModelfitResults. Each method from ModelfitResults
    will be performed on the final modelfit object
    """

    def __init__(self, results=None):
        if isinstance(results, ChainedModelfitResults):
            self._results = copy.deepcopy(results._results)
        elif results is None:
            self._results = []
        else:
            self._results = list(results)

    def __getitem__(self, ind):
        return self._results[ind]

    def __setitem__(self, ind, value):
        self._results[ind] = value

    def __delitem__(self, ind):
        del self._results[ind]

    def __len__(self):
        return len(self._results)

    def insert(self, ind, value):
        self._results.insert(ind, value)

    @property
    def ofv(self):
        return self[-1].ofv

    @property
    def evaluation_ofv(self):
        return self[0].evaluation_ofv

    @property
    def minimization_successful(self):
        # If last step was estimation
        est_steps = self.model.estimation_steps
        if not est_steps[-1].evaluation and self[-1].minimization_successful:
            return self[-1].minimization_successful
        # If last was evaluation, find latest estimation
        for i in reversed(range(len(self))):
            if not est_steps[i].evaluation:
                return self[i].minimization_successful
        # If all steps were evaluation the last evaluation step is relevant
        return self[-1].minimization_successful

    @property
    def parameter_estimates(self):
        return self[-1].parameter_estimates

    @parameter_estimates.setter
    def parameter_estimates(self, value):
        self[-1].parameter_estimates = value

    @property
    def parameter_estimates_sdcorr(self):
        return self[-1].parameter_estimates_sdcorr

    @property
    def covariance_matrix(self):
        return self[-1].covariance_matrix

    @property
    def information_matrix(self):
        return self[-1].information_matrix

    @property
    def correlation_matrix(self):
        return self[-1].correlation_matrix

    @property
    def standard_errors(self):
        return self[-1].standard_errors

    @property
    def standard_errors_sdcorr(self):
        return self[-1].standard_errors_sdcorr

    @property
    def individual_ofv(self):
        return self[-1].individual_ofv

    @property
    def individual_estimates(self):
        return self[-1].individual_estimates

    @property
    def individual_estimates_covariance(self):
        return self[-1].individual_estimates_covariance

    @property
    def residuals(self):
        return self[-1].residuals

    @property
    def predictions(self):
        return self[-1].predictions

    @property
    def model_name(self):
        return self[-1].model_name

    @property
    def runtime_total(self):
        return self[-1].runtime_total

    def result_summary(self, include_all_estimation_steps=False):
        if not include_all_estimation_steps:
            summary_dict = self._summarize_step(-1)
            summary_df = pd.DataFrame(summary_dict, index=[self.model_name])
            return summary_df
        else:
            summary_dicts = []
            tuples = []
            for i in range(len(self)):
                summary_dict = self._summarize_step(i)
                is_evaluation = self.model.estimation_steps[i].evaluation
                if is_evaluation:
                    run_type = 'evaluation'
                else:
                    run_type = 'estimation'
                summary_dict = {**{'run_type': run_type}, **summary_dict}
                summary_dicts.append(summary_dict)
                tuples.append((self.model_name, i + 1))
            index = pd.MultiIndex.from_tuples(tuples, names=['model_name', 'step'])
            summary_df = pd.DataFrame(summary_dicts, index=index)
            return summary_df

    def _summarize_step(self, i):
        summary_dict = dict()

        if i >= 0:
            step = self[i]
        else:
            step = self

        if step.minimization_successful is not None:
            summary_dict['minimization_successful'] = step.minimization_successful
        else:
            summary_dict['minimization_successful'] = False

        summary_dict['ofv'] = step.ofv
        summary_dict['runtime_total'] = step.runtime_total

        pe = step.parameter_estimates
        ses = step.standard_errors
        rses = step.relative_standard_errors

        for param in pe.index:
            summary_dict[f'{param}_estimate'] = pe[param]
            if ses is not None:
                summary_dict[f'{param}_SE'] = ses[param]
            if rses is not None:
                summary_dict[f'{param}_RSE'] = rses[param]

        return summary_dict

    def __repr__(self):
        return repr(self._results[-1])

    # FIXME: To not have to manually intercept everything here. Could do it in a general way.
