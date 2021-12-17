"""DataInfo is a companion to the dataset. It contains metadata of the dataset
"""
import pandas as pd


class ColumnInfo:
    all_types = ['id', 'dv', 'idv', 'unknown', 'dose', 'event', 'covariate']

    def __init__(self, name, tp='unknown'):
        self.name = name
        self._type = tp

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if value not in ColumnInfo.all_types:
            raise KeyError(f"Unknown column type {value}")
        self._type = value


class DataInfo:
    def __init__(self, column_names):
        self.columns = []
        for name in column_names:
            colinf = ColumnInfo(name)
            self.columns.append(colinf)

    def _get_one_label_by_type(self, tp):
        for col in self.columns:
            if tp == col.type:
                return col.name
        return None

    def _set_one_label_to_type(self, name, tp):
        for col in self.columns:
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
        return [col.name for col in self.columns]

    def set_column_type(self, labels, tp):
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            for col in self.columns:
                if col.name == label:
                    col.type = tp
                    break
            else:
                raise KeyError(f"No column named {label}")

    def get_column_type(self, label):
        for col in self.columns:
            if col.name == label:
                return col.type
        raise KeyError(f"No column named {label}")

    def get_column_label(self, tp):
        for col in self.columns:
            if col.type == tp:
                return col.name
        return None

    def get_column_labels(self, tp):
        labels = [col.name for col in self.columns if col.type == tp]
        return labels

    def __repr__(self):
        labels = [col.name for col in self.columns]
        types = [col.type for col in self.columns]
        df = pd.DataFrame(columns=labels)
        df.loc['type'] = types
        return df.to_string()
