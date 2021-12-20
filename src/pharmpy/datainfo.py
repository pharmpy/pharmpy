"""DataInfo is a companion to the dataset. It contains metadata of the dataset
"""
from collections.abc import MutableSequence

import pandas as pd
import sympy

from pharmpy.utils import parse_units


class ColumnInfo:
    all_types = ['id', 'dv', 'idv', 'unknown', 'dose', 'event', 'covariate']
    all_scales = ['nominal', 'ordinal', 'interval', 'ratio']

    def __init__(
        self,
        name,
        tp='unknown',
        unit=sympy.Integer(1),
        scale='ratio',
        continuous=True,
        categories=None,
    ):
        if scale in ['nominal', 'ordinal']:
            continuous = False
        self._continuous = continuous
        self.name = name
        self.type = tp
        self._unit = unit
        self.scale = scale
        self.continuous = continuous
        if categories is not None:
            self.categories = categories  # dict from value to descriptive string

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if value not in ColumnInfo.all_types:
            raise TypeError(f"Unknown column type {value}")
        self._type = value

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        a = parse_units(value)
        self._unit = a

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value not in ColumnInfo.all_scales:
            raise TypeError(
                f"Unknown scale of measurement {value}. Only {ColumnInfo.all_scales} are possible."
            )
        self._scale = value
        if self.continuous and value in ['nominal', 'ordinal']:
            self.continuous = False

    @property
    def continuous(self):
        return self._continuous

    @continuous.setter
    def continuous(self, value):
        if value and self.is_categorical():
            raise ValueError(
                f"Cannot set variable on {self.scale} scale of measurement to continuous"
            )
        self._continuous = value

    def is_categorical(self):
        return self.scale in ['nominal', 'ordinal']

    def is_numerical(self):
        return self.scale in ['interval', 'ratio']


class DataInfo(MutableSequence):
    def __init__(self, column_names):
        self._columns = []
        for name in column_names:
            colinf = ColumnInfo(name)
            self._columns.append(colinf)

    def __len__(self):
        return len(self._columns)

    def _getindex(self, i):
        if isinstance(i, str):
            for n, col in enumerate(self._columns):
                if col.name == i:
                    return n
        elif isinstance(i, int):
            return i
        else:
            raise TypeError(f"Cannot index DataInfo by {type(i)}")

    def __getitem__(self, i):
        return self._columns[self._getindex(i)]

    def __setitem__(self, i, value):
        self._columns[self._getindex(i)] = value

    def __delitem__(self, i):
        del self._columns[self._getindex(i)]

    def insert(self, i, value):
        self._columns.index(self._getindex(i), value)

    def _get_one_label_by_type(self, tp):
        for col in self._columns:
            if tp == col.type:
                return col.name
        return None

    def _set_one_label_to_type(self, name, tp):
        for col in self._columns:
            if name == col.name:
                col.type = tp
                return
        raise KeyError(f"No column with the name {name}")

    @property
    def id_label(self):
        return self._get_one_label_by_type('id')

    @id_label.setter
    def id_label(self, name):
        self._set_one_label_to_type(name, 'id')

    @property
    def dv_label(self):
        return self._get_one_label_by_type('dv')

    @dv_label.setter
    def dv_label(self, name):
        self._set_one_label_to_type(name, 'dv')

    @property
    def idv_label(self):
        return self._get_one_label_by_type('idv')

    @idv_label.setter
    def idv_label(self, name):
        self._set_one_label_to_type(name, 'idv')

    @property
    def column_names(self):
        return [col.name for col in self._columns]

    def set_column_type(self, labels, tp):
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            for col in self._columns:
                if col.name == label:
                    col.type = tp
                    break
            else:
                raise KeyError(f"No column named {label}")

    def get_column_type(self, label):
        for col in self._columns:
            if col.name == label:
                return col.type
        raise KeyError(f"No column named {label}")

    def get_column_label(self, tp):
        for col in self._columns:
            if col.type == tp:
                return col.name
        return None

    def get_column_labels(self, tp):
        labels = [col.name for col in self._columns if col.type == tp]
        return labels

    def __repr__(self):
        labels = [col.name for col in self._columns]
        types = [col.type for col in self._columns]
        df = pd.DataFrame(columns=labels)
        df.loc['type'] = types
        return df.to_string()
