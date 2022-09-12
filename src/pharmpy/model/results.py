import json
from pathlib import Path

from pharmpy.deps import pandas as pd

from .model import Model


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
        s = ResultsJSONEncoder().encode(self)
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


class ResultsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # NOTE this function is called when the base JSONEncoder does not know
        # how to encode the given object, so it will not be called on int,
        # float, str, list, tuple, and dict. It could be called on set for
        # instance, or any custom class.
        from pharmpy.workflows.log import Log

        if isinstance(obj, Results):
            d = obj.to_dict()
            d['__module__'] = obj.__class__.__module__
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, pd.DataFrame):
            if str(obj.columns.dtype) == 'int64':
                # Workaround for https://github.com/pandas-dev/pandas/issues/46392
                obj.columns = obj.columns.map(str)
            d = json.loads(obj.to_json(orient='table'))
            d['__class__'] = 'DataFrame'
            return d
        elif isinstance(obj, pd.Series):
            d = json.loads(obj.to_json())
            if obj.name is not None:
                d['__name__'] = obj.name
            d['__class__'] = 'Series'
            return d
        elif obj.__class__.__module__.startswith('altair.'):
            d = obj.to_dict()
            d['__class__'] = 'vega-lite'
            return d
        elif isinstance(obj, Model):
            # TODO consider using other representation, e.g. path
            return None
        elif isinstance(obj, Log):
            d = obj.to_dict()
            d['__class__'] = obj.__class__.__qualname__
            return d
        elif isinstance(obj, Path):
            d = {'path': str(obj), '__class__': 'PosixPath'}
            return d
        else:
            # NOTE this will raise a proper TypeError
            return super().default(obj)
