from __future__ import annotations

import importlib
import json
import lzma
import re
import warnings
from contextlib import closing
from dataclasses import dataclass
from io import StringIO
from lzma import open as lzma_open
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, overload

import pharmpy
from pharmpy.deps import altair as alt
from pharmpy.internals.immutable import Immutable
from pharmpy.model import Model

if TYPE_CHECKING:
    import pandas as pd

    from pharmpy.workflows import Log
else:
    from pharmpy.deps import pandas as pd


def mfr(res: ModelfitResults) -> ModelfitResults:
    assert isinstance(res, ModelfitResults)
    return res


def _df_to_json(df: pd.DataFrame) -> dict[str, Any]:
    if str(df.columns.dtype) == 'int64':
        # Workaround for https://github.com/pandas-dev/pandas/issues/46392
        df.columns = df.columns.map(str)
    # Set double precision to 15 to remove some round-trip errors, however 17 should be set when its possible
    # See: https://github.com/pandas-dev/pandas/issues/38437
    df_json = df.to_json(orient='table', double_precision=15)
    assert df_json is not None
    return json.loads(df_json)


def _index_to_json(index: Union[pd.Index, pd.MultiIndex]) -> dict[str, Any]:
    if isinstance(index, pd.MultiIndex):
        return {'__class__': 'MultiIndex', **_df_to_json(index.to_frame(index=False))}
    return {'__class__': 'Index', **_df_to_json(index.to_frame(index=False))}


class ResultsJSONEncoder(json.JSONEncoder):
    def default(self, obj) -> Union[dict[str, Any], None]:
        # NOTE: This function is called when the base JSONEncoder does not know
        # how to encode the given object, so it will not be called on int,
        # float, str, list, tuple, and dict. It could be called on set for
        # instance, or any custom class.
        from pharmpy.workflows import Log

        if isinstance(obj, Results):
            d = obj.to_dict()
            d['__module__'] = obj.__class__.__module__
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, pd.DataFrame):
            d = _df_to_json(obj)
            d['__class__'] = 'DataFrame'
            return d
        elif isinstance(obj, pd.Series):
            if obj.size >= 1 and isinstance(obj.iloc[0], pd.DataFrame):
                # NOTE: Hack special case for Series of DataFrame objects
                return {
                    'data': [{'__class__': 'DataFrame', **_df_to_json(df)} for df in obj.values],
                    'index': _index_to_json(obj.index),
                    'name': obj.name,
                    'dtype': str(obj.dtype),
                    '__class__': 'Series[DataFrame]',
                }

            # NOTE: Hack to work around poor support of to_json/read_json of
            # pd.Series with MultiIndex
            df = obj.to_frame()
            d = _df_to_json(df)
            d['__class__'] = 'Series'
            return d
        elif obj.__class__.__module__.startswith('altair.'):
            with warnings.catch_warnings():
                # FIXME: Remove filter once altair stops relying on deprecated APIs
                warnings.filterwarnings(
                    "ignore",
                    message=".*iteritems is deprecated and will be removed in a future version. Use .items instead.",
                    category=FutureWarning,
                )
                # This was fixed in altair v2.1.1 can be removed once we require that version
                warnings.filterwarnings(
                    "ignore",
                    message=".*the convert_dtype parameter is deprecated",
                    category=FutureWarning,
                )
                d = obj.to_dict()
            d['__module__'] = obj.__class__.__module__
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, Model):
            # TODO: Consider using other representation, e.g. path
            return None
        elif isinstance(obj, Log):
            d: Dict[Any, Any] = obj.to_dict()
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, Path):
            d = {'path': str(obj), '__class__': 'PosixPath'}
            return d
        else:
            # NOTE: This will raise a proper TypeError
            return super().default(obj)


def _df_read_json(obj) -> pd.DataFrame:
    # Convert time strings to naive datetime and then to string
    # Needed because of https://github.com/pandas-dev/pandas/issues/52595
    for row in obj['data']:
        if 'time' in row:
            row['time'] = pd.to_datetime(row['time']).tz_localize(None)
            row['time'] = row['time'].isoformat()

    return pd.read_json(StringIO(json.dumps(obj)), typ='frame', orient='table', precise_float=True)


def _multi_index_read_json(obj) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(_df_read_json(obj))


class ResultsJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        # NOTE: This hook will be called for every dict produced by the
        # base JSONDecoder. It will not be called on int, float, str, or list.
        module = None
        cls = None

        if '__module__' in obj:
            module = obj['__module__']
            del obj['__module__']

        if '__class__' in obj:
            cls = obj['__class__']
            del obj['__class__']

        # NOTE: Handling cls not None and module is None is kept for backwards
        # compatibility

        if cls is None and module is not None:
            raise ValueError('Cannot specify module without specifying class')

        if module is None or module.startswith('pandas.'):
            if cls == 'DataFrame':
                return _df_read_json(obj)
            elif cls == 'Series':
                # NOTE: Hack to work around poor support of to_json/read_json of
                # pd.Series with MultiIndex
                df = _df_read_json(obj)
                series = df.iloc[:, 0]  # NOTE: First and only column.
                return series
            elif cls == 'Series[DataFrame]':
                # NOTE: Hack to work around poor support of Series of DataFrame
                # objects. All subobjects have already been converted.
                return pd.Series(
                    obj['data'], index=obj['index'], dtype=obj['dtype'], name=obj['name']
                )
            elif cls == 'Index':
                return _multi_index_read_json(obj).get_level_values(0)
            elif cls == 'MultiIndex':
                return _multi_index_read_json(obj)

        if module is None:
            if cls == 'vega-lite':
                # NOTE: Slow parsing for parsing PsN frem output and old format
                return alt.Chart.from_dict(obj, validate=True)

        if module is not None and module.startswith('altair.'):
            # NOTE: Fast parsing when reading own output
            assert cls is not None
            try:
                class_ = getattr(alt, cls)
            except AttributeError:
                raise ValueError(f'Unknown class {cls} in {module}')
            return class_.from_dict(obj, validate=False)

        if cls is not None and cls.endswith('Results'):
            if module is None:
                # NOTE: Kept for backwards compatibility: we guess the module
                # path based on the class name.
                tool_name = cls[:-7].lower()  # NOTE: Trim "Results" suffix
                tool_module = importlib.import_module(f'pharmpy.tools.{tool_name}')
                results_class = tool_module.results_class
            else:
                tool_module = importlib.import_module(module)
                results_class = getattr(tool_module, cls)

            return results_class.from_dict(obj)

        if cls == 'PosixPath':
            return Path(obj)
        if cls == 'Log':
            from pharmpy.workflows import Log

            return Log.from_dict(obj)

        return obj


def _is_likely_to_be_json(source: str):
    # NOTE: Heuristic to determine if path or buffer: first non-space character
    # is '{'.
    match = re.match(r'\s*([^\s])', source)
    return match is not None and match.group(1) == '{'


def read_results(path_or_str: Union[str, Path]):
    if isinstance(path_or_str, str) and _is_likely_to_be_json(path_or_str):
        manager = closing(StringIO(path_or_str))
    else:
        path = Path(path_or_str)
        if path.is_dir():
            path /= 'results.json'

        if path.name.endswith('.xz'):
            manager = lzma.open(path, 'r', encoding='utf-8')
        else:
            manager = open(path, 'r')

    with manager as readable:
        return json.load(readable, cls=ResultsJSONDecoder)


@dataclass(frozen=True)
class Results(Immutable):
    """Base class for all result classes"""

    __version__: str = pharmpy.__version__  # NOTE: Default version if not overridden

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        """Create results object from dictionary"""
        removed_keys = {
            '__version__',
            'best_model',  # NOTE: Was removed in d5b3503 and 8578c8b
            'input_model',  # NOTE: Was removed in d5b3503 and 8578c8b
        }
        return cls(
            __version__=d.get('__version__', 'unknown'),  # NOTE: Override default version
            **{k: v for k, v in d.items() if k not in removed_keys},
        )

    @overload
    def to_json(self, path: None = None, lzma: Literal[False] = False) -> str: ...

    @overload
    def to_json(self, path: Path, lzma: bool = False) -> None: ...

    def to_json(self, path: Optional[Path] = None, lzma: bool = False) -> Union[str, None]:
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
        s = ResultsJSONEncoder().encode(self)
        if path:
            if not lzma:
                with open(path, 'w') as fh:
                    fh.write(s)
            else:
                xz_path = path.parent / (path.name + '.xz')
                with lzma_open(xz_path, 'w') as fh:
                    fh.write(bytes(s, 'utf-8'))
        else:
            return s

    def get_and_reset_index(self, attr: str, **kwargs) -> pd.DataFrame:
        """Wrapper to reset index of attribute or result from method.

        Used to facilitate importing multiindex dataframes into R
        """
        val = getattr(self, attr)
        if callable(val):
            df = val(**kwargs)
        else:
            df = val
        return df.reset_index()

    def to_dict(self) -> dict[str, Any]:
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

    def to_csv(self, path: Path):
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
            elif isinstance(value, Model):
                continue
            elif isinstance(value, list) and isinstance(value[0], Model):
                continue
            s += f'{key}\n'
            if isinstance(value, pd.DataFrame):
                if isinstance(value.index, pd.RangeIndex):
                    use_index = False
                else:
                    use_index = True
                csv = value.to_csv(index=use_index)
                assert isinstance(csv, str)
                s += csv
            elif isinstance(value, pd.Series):
                csv = value.to_csv()
                assert isinstance(csv, str)
                s += csv
            elif isinstance(value, list):  # Print list of lists as table
                if len(value) > 0 and isinstance(value[0], list):
                    for row in value:
                        s += f'{",".join(map(str, row))}\n'
            else:
                s += str(value) + '\n'
            s += '\n'
        with open(path, 'w', newline='') as fh:
            print(s, file=fh)


@dataclass(frozen=True)
class ModelfitResults(Results):
    """Base class for results from a modelfit operation

    Attributes
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix of the population parameter estimates
    covariance_matrix : pd.DataFrame
        Covariance matrix of the population parameter estimates
    precision_matrix : pd.DataFrame
        Precision matrix of the population parameter estimates
    evaluation_ofv : float
        The objective function value as if the model was evaluated. Currently
        works for classical estimation methods by taking the OFV of the first
        iteration.
    individual_ofv : pd.Series
        OFV for each individual
    individual_estimates : pd.DataFrame
        Estimates for etas
    individual_estimates_covariance : pd.Series
        Estimated covariance between etas
    parameter_estimates : pd.Series
        Population parameter estimates
    parameter_estimates_iterations : pd.DataFrame
        All recorded iterations for parameter estimates
    parameter_estimates_sdcorr : pd.Series
        Population parameter estimates with variability parameters as standard deviations and
        correlations
    residuals: pd.DataFrame
        Table of various residuals
    predictions: pd.DataFrame
        Table of various predictions
    derivaitves: pd.DataFrame
        Table of various derivatives
    estimation_runtime : float
        Runtime for one estimation step
    runtime_total : float
        Total runtime of estimation
    standard_errors : pd.Series
        Standard errors of the population parameter estimates
    standard_errors_sdcorr : pd.Series
        Standard errors of the population parameter estimates on standard deviation and correlation
        scale
    relative_standard_errors : pd.Series
        Relative standard errors of the population parameter estimates
    termination_cause : str
        The cause of premature termination. One of 'maxevals_exceeded' and 'rounding_errors'
    function_evaluations : int
        Number of function evaluations
    evaluation : pd.Series
        A bool for each estimation step. True if this was a model evaluation and False otherwise
    covstep_successful : bool or None
        Covariance status.
    gradients : pd.Series
        Final parameter gradients
    gradients_iterations : pd.DataFrame
        All recorded parameter gradients
    warnings : List
        List of warnings
    """

    ofv: Optional[float] = None
    ofv_iterations: Optional[pd.Series] = None
    parameter_estimates: Optional[pd.Series] = None
    parameter_estimates_sdcorr: Optional[pd.Series] = None
    parameter_estimates_iterations: Optional[pd.DataFrame] = None
    covariance_matrix: Optional[pd.DataFrame] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    precision_matrix: Optional[pd.DataFrame] = None
    standard_errors: Optional[pd.Series] = None
    standard_errors_sdcorr: Optional[pd.Series] = None
    relative_standard_errors: Optional[pd.Series] = None
    minimization_successful: Optional[bool] = None
    minimization_successful_iterations: Optional[pd.DataFrame] = None
    estimation_runtime: Optional[float] = None
    estimation_runtime_iterations: Optional[pd.DataFrame] = None
    individual_ofv: Optional[pd.Series] = None
    individual_estimates: Optional[pd.DataFrame] = None
    individual_estimates_covariance: Optional[pd.DataFrame] = None
    residuals: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None
    derivatives: Optional[pd.DataFrame] = None
    runtime_total: Optional[float] = None
    termination_cause: Optional[str] = None
    termination_cause_iterations: Optional[pd.Series] = None
    function_evaluations: Optional[float] = None
    function_evaluations_iterations: Optional[pd.Series] = None
    significant_digits: Optional[float] = None
    significant_digits_iterations: Optional[pd.Series] = None
    log_likelihood: Optional[float] = None
    log: Optional['Log'] = None
    evaluation: Optional[pd.Series] = None
    covstep_successful: Optional[bool, None] = None
    gradients: Optional[pd.Series] = None
    gradients_iterations: Optional[pd.DataFrame] = (None,)
    warnings: Optional[List[str]] = None

    def __repr__(self):
        return '<Pharmpy modelfit results object>'


@dataclass(frozen=True)
class SimulationResults(Results):
    """Base class for resutls from simulation operation

    Attributes
    ----------
    table : pd.DataFrame
        Table file of model
    """

    table: Optional[pd.DataFrame] = None

    def __repr__(self):
        return '<Pharmpy simulation results object>'
