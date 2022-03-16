import copy
import importlib
import json
import lzma
from collections.abc import MutableSequence
from pathlib import Path

import altair as alt
import pandas as pd

import pharmpy.model


class ResultsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            if isinstance(obj, pd.DataFrame) and str(obj.columns.dtype) == 'int64':
                # Workaround for https://github.com/pandas-dev/pandas/issues/46392
                obj.columns = obj.columns.map(str)
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
        elif isinstance(obj, pharmpy.model.Model):
            # TODO: consider using other representation, e.g. path
            return None
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
    tool_name = decoder.cls[:-7].lower()
    tool_module = importlib.import_module(f'pharmpy.tools.{tool_name}')
    results_class = tool_module.results_class
    res = results_class.from_dict(d)
    return res


class Results:
    """Base class for all result classes"""

    @classmethod
    def from_dict(cls, d):
        """Create results object from dictionary"""
        return cls(**d)

    def to_json(self, path=None, lzma=False):
        """Serialize results object as json

        Parameters
        ----------
        path : Path
            Path to save json file or None to serialize to string
        lzma : bool
            Set to compress file with lzma

        Returns
        -------
        str
            Json as string unless path was used
        """
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

        Parameters
        ----------
        path : Path
            Path to csv-file
        """
        d = self.to_dict()
        s = ""
        for key, value in d.items():
            if value.__class__.__module__.startswith('altair.'):
                continue
            elif isinstance(value, pharmpy.model.Model):
                continue
            elif isinstance(value, list) and isinstance(value[0], pharmpy.model.Model):
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
    estimation_runtime : float
        Runtime for one estimation step
    runtime_total : float
        Total runtime of estimation
    standard_errors : pd.Series
        Standard errors of the population parameter estimates
    standard_errors_sdcorr : pd.Series
        Standard errors of the population parameter estimates on standard deviation and correlation
        scale
    termination_cause : str
        The cause of premature termination. One of 'maxevals_exceeded' and 'rounding_errors'
    function_evaluations : int
        Number of function evaluations
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
        termination_cause=None,
        function_evaluations=None,
    ):
        self.ofv = ofv
        self.parameter_estimates = parameter_estimates
        self.parameter_estimates_sdcorr = parameter_estimates_sdcorr
        self.covariance_matrix = covariance_matrix
        self.correlation_matrix = correlation_matrix
        self.standard_errors = standard_errors
        self.minimization_successful = minimization_successful
        self.individual_estimates = individual_estimates
        self.individual_ofv = individual_ofv
        self.residuals = residuals
        self.runtime_total = runtime_total
        self.termination_cause = termination_cause
        self.function_evaluations = function_evaluations

    def __bool__(self):
        return bool(self.ofv) and bool(self.parameter_estimates)

    def to_dict(self):
        """Convert results object to a dictionary"""
        return {'parameter_estimates': self.parameter_estimates}

    @property
    def relative_standard_errors(self):
        """Relative standard errors of population parameter estimates"""
        if self.standard_errors is not None:
            ser = self.standard_errors / self.parameter_estimates
            ser.name = 'RSE'
            return ser


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
        return self._get_last_est('minimization_successful')

    @property
    def estimation_runtime(self):
        return self._get_last_est('estimation_runtime')

    def _get_last_est(self, attr):
        est_steps = self.model.estimation_steps
        # Find last estimation
        for i in reversed(range(len(self))):
            if not est_steps[i].evaluation and getattr(self[i], attr) is not None:
                return getattr(self[i], attr)
        # If all steps were evaluation the last evaluation step is relevant
        return getattr(self[-1], attr)

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
    def function_evaluations(self):
        return self._get_last_est('function_evaluations')

    @property
    def termination_cause(self):
        return self._get_last_est('termination_cause')

    @property
    def runtime_total(self):
        return self[-1].runtime_total

    def __repr__(self):
        return repr(self._results[-1])
