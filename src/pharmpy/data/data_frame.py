import enum

import pandas as pd


class ColumnType(enum.Enum):
    UNKNOWN = enum.auto()
    ID = enum.auto()
    IDV = enum.auto()
    DV = enum.auto()
    COVARIATE = enum.auto()

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.name}'


class PharmDataFrame(pd.DataFrame):
    _metadata = [ '_column_types' ]

    @property
    def _constructor(self):
        return PharmDataFrame

    @property
    def _constructor_sliced(self):
        return pd.Series
