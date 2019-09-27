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
