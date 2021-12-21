"""DataInfo is a companion to the dataset. It contains metadata of the dataset
"""
import json
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
        type='unknown',
        unit=sympy.Integer(1),
        scale='ratio',
        continuous=True,
        categories=None,
        drop=False,
    ):
        if scale in ['nominal', 'ordinal']:
            continuous = False
        self._continuous = continuous
        self.name = name
        self.type = type
        self.unit = unit
        self.scale = scale
        self.continuous = continuous
        self.categories = categories  # dict from value to descriptive string
        self.drop = drop

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.type == other.type
            and self.unit == other.unit
            and self.scale == other.scale
            and self.continuous == other.continuous
            and self.categories == other.categories
            and self.drop == other.drop
        )

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

    def __repr__(self):
        di = DataInfo([self])
        return repr(di)


class DataInfo(MutableSequence):
    def __init__(self, columns):
        if len(columns) > 0 and isinstance(columns[0], str):
            self._columns = []
            for name in columns:
                colinf = ColumnInfo(name)
                self._columns.append(colinf)
        else:
            self._columns = columns

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for col1, col2 in zip(self, other):
            if col1 != col2:
                return False
        return True

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

    def to_json(self, path=None):
        a = []
        for col in self._columns:
            d = {
                "name": col.name,
                "type": col.type,
                "scale": col.scale,
                "continuous": col.continuous,
                "unit": str(col.unit),
            }
            a.append(d)
        s = json.dumps({"columns": a})
        if path is None:
            return s
        else:
            with open(path, 'w') as fp:
                fp.write(s)

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        columns = []
        for col in d['columns']:
            ci = ColumnInfo(
                name=col['name'],
                type=col['type'],
                scale=col['scale'],
                continuous=col.get('continuous', True),
                unit=col.get('unit', sympy.Integer(1)),
            )
            columns.append(ci)
        return DataInfo(columns)

    @staticmethod
    def read_json(path):
        with open(path, 'r') as fp:
            s = fp.read()
        return DataInfo.from_json(s)

    def __repr__(self):
        labels = [col.name for col in self._columns]
        types = [col.type for col in self._columns]
        scales = [col.scale for col in self._columns]
        cont = [col.continuous for col in self._columns]
        cats = [col.categories for col in self._columns]
        units = [col.unit for col in self._columns]
        drop = [col.drop for col in self._columns]
        df = pd.DataFrame(columns=labels)
        df.loc['type'] = types
        df.loc['scale'] = scales
        df.loc['continuous'] = cont
        df.loc['categories'] = cats
        df.loc['unit'] = units
        df.loc['drop'] = drop
        return df.to_string()
