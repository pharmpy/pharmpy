"""DataInfo is a companion to the dataset. It contains metadata of the dataset
"""


class ColumnInfo:
    def __init__(self, name, tp='unknown'):
        self.name = name
        self.type = tp


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
