import json
import warnings
from dataclasses import dataclass
from lzma import open as lzma_open
from pathlib import Path
from typing import Any, Dict, Literal, Union, overload

import pharmpy
from pharmpy.deps import pandas as pd
from pharmpy.internals.immutable import Immutable

from .model import Model


@dataclass(frozen=True)
class Results(Immutable):
    """Base class for all result classes"""

    __version__: str = pharmpy.__version__  # NOTE Default version if not overridden

    @classmethod
    def from_dict(cls, d):
        """Create results object from dictionary"""
        removed_keys = {
            '__version__',
            'best_model',  # NOTE Was removed in d5b3503 and 8578c8b
            'input_model',  # NOTE Was removed in d5b3503 and 8578c8b
        }
        return cls(
            __version__=d.get('__version__', 'unknown'),  # NOTE Override default version
            **{k: v for k, v in d.items() if k not in removed_keys},
        )

    @overload
    def to_json(self, path: None = None, lzma: Literal[False] = False) -> str:
        ...

    @overload
    def to_json(self, path: Union[str, Path], lzma: bool = False) -> None:
        ...

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
                csv = value.to_csv(index=use_index)  # pyright: ignore [reportGeneralTypeIssues]
                assert isinstance(csv, str)
                s += csv
            elif isinstance(value, pd.Series):
                csv = value.to_csv()  # pyright: ignore [reportGeneralTypeIssues]
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


def _df_to_json(df):
    if str(df.columns.dtype) == 'int64':
        # Workaround for https://github.com/pandas-dev/pandas/issues/46392
        df.columns = df.columns.map(str)
    # Set double precision to 15 to remove some round-trip errors, however 17 should be set when its possible
    # See: https://github.com/pandas-dev/pandas/issues/38437
    return json.loads(df.to_json(orient='table', double_precision=15))


def _index_to_json(index):
    if isinstance(index, pd.MultiIndex):
        return {'__class__': 'MultiIndex', **_df_to_json(index.to_frame(index=False))}
    return {'__class__': 'Index', **_df_to_json(index.to_frame(index=False))}


class ResultsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # NOTE this function is called when the base JSONEncoder does not know
        # how to encode the given object, so it will not be called on int,
        # float, str, list, tuple, and dict. It could be called on set for
        # instance, or any custom class.
        from pharmpy.workflows import LocalDirectoryToolDatabase, Log

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
                # NOTE Hack special case for Series of DataFrame objects
                return {
                    'data': [{'__class__': 'DataFrame', **_df_to_json(df)} for df in obj.values],
                    'index': _index_to_json(obj.index),
                    'name': obj.name,
                    'dtype': str(obj.dtype),
                    '__class__': 'Series[DataFrame]',
                }

            # NOTE Hack to work around poor support of to_json/read_json of
            # pd.Series with MultiIndex
            df = obj.to_frame()
            d = _df_to_json(df)
            d['__class__'] = 'Series'
            return d
        elif obj.__class__.__module__.startswith('altair.'):
            with warnings.catch_warnings():
                # FIXME Remove filter once altair stops relying on deprecated APIs
                warnings.filterwarnings(
                    "ignore",
                    message=".*iteritems is deprecated and will be removed in a future version. Use .items instead.",
                    category=FutureWarning,
                )
                d = obj.to_dict()
            d['__module__'] = obj.__class__.__module__
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, Model):
            # TODO consider using other representation, e.g. path
            return None
        elif isinstance(obj, Log):
            d: Dict[Any, Any] = obj.to_dict()
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, LocalDirectoryToolDatabase):
            d = obj.to_dict()
            d['__module__'] = obj.__class__.__module__
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, Path):
            d = {'path': str(obj), '__class__': 'PosixPath'}
            return d
        else:
            # NOTE this will raise a proper TypeError
            return super().default(obj)
