from pathlib import Path

import pandas as pd

import pharmpy.data


class DatasetError(Exception):
    pass


class DatasetWarning(Warning):
    pass


class PharmDataFrame(pd.DataFrame):
    """A DataFrame with additional metadata.

    ============  =============
    ColumnType    Description
    ============  =============
    ID            Individual identifier. Max one per DataFrame. All values have to be unique
    IDV           Independent variable. Max one per DataFrame.
    DV            Dependent variable
    COVARIATE     Covariate
    DOSE          Dose amount
    EVENT         0 = observation
    UNKOWN        Unkown type. This will be the default for columns that hasn't been assigned a type
    ============  =============

    """

    _metadata = ['_column_types', 'name']

    @property
    def _constructor(self):
        return PharmDataFrame

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __deepcopy__(self, memo):
        return self.copy()

    def copy(self, *kwargs):
        """ """
        # FIXME: Set empty docstring to avoid getting documentation from base class
        #        would like sphinx to do this so that in object docstring is kept.
        new_df = super().copy(*kwargs)
        try:
            new_df._column_types = self._column_types.copy()
        except AttributeError:
            pass
        return new_df

    def to_json(self, **kwargs):
        """ """
        # FIXME: Same docstring issue as for copy
        # FIXME: Directly using to_json on PharmDataFrame doesn't work
        return pd.DataFrame(self).to_json(**kwargs)


@pd.api.extensions.register_dataframe_accessor('pharmpy')
class DataFrameAccessor:
    def __init__(self, obj):
        self._obj = obj

    def generate_path(self, path=None, force=False):
        """Generate the path of dataframe if written.
        If no path is supplied or does not contain a filename a name is created
        from the name property of the PharmDataFrame.
        Will not overwrite unless forced.
        """
        if path is None:
            path = Path("")
        else:
            path = Path(path)
        if not path or path.is_dir():
            try:
                filename = f'{self._obj.name}.csv'
            except AttributeError:
                raise ValueError(
                    'Cannot name data file as no path argument was supplied and '
                    'the DataFrame has no name property.'
                )
            path /= filename
        if not force and path.exists():
            raise FileExistsError(f'File at {path} already exists.')
        return path

    def write_csv(self, path=None, force=False):
        """Write PharmDataFrame to a csv file
        return path for the written file
        """
        path = self.generate_path(path, force)
        self._obj.to_csv(path, na_rep=pharmpy.data.conf.na_rep, index=False)
        return path
