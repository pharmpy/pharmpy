import enum
from pathlib import Path

import numpy as np
import pandas as pd

import pharmpy.data


class DatasetError(Exception):
    pass


class DatasetWarning(Warning):
    pass


class ColumnType(enum.Enum):
    """The type of the data in a column"""

    UNKNOWN = enum.auto()
    ID = enum.auto()
    IDV = enum.auto()
    DV = enum.auto()
    COVARIATE = enum.auto()
    DOSE = enum.auto()
    EVENT = enum.auto()

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.name}'

    @property
    def max_one(self):
        """Can this ColumnType only be assigned to at most one column?"""
        return self in (ColumnType.ID, ColumnType.IDV, ColumnType.DOSE, ColumnType.EVENT)


class PharmDataFrame(pd.DataFrame):
    """A DataFrame with additional metadata.

    Each column can have a ColumnType. The default ColumnType is UNKNOWN.

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
        """"""
        # FIXME: Set empty docstring to avoid getting documentation from base class
        #        would like sphinx to do this so that in object docstring is kept.
        new_df = super().copy(*kwargs)
        try:
            new_df._column_types = self._column_types.copy()
        except AttributeError:
            pass
        return new_df


class ColumnTypeIndexer:
    """Indexing a PharmDataFrame to get or set column types
    An instance of ColumnTypeIndexer can be retrieved by df.pharmpy.column_type
    """

    def __init__(self, df):
        self._obj = df

    def _set_one_column_type(self, label, tp):
        if label not in self._obj.columns:
            raise KeyError(str(label))
        if (
            tp.max_one
            and hasattr(self._obj, '_column_types')
            and tp in self._obj._column_types.values()
        ):
            raise KeyError(f'Only one column of type {tp} is allowed in a PharmDataFrame.')
        try:
            self._obj._column_types[label] = tp
        except AttributeError:
            self._obj._column_types = {}
            self._obj._column_types[label] = tp

    def __setitem__(self, ind, tp):
        if isinstance(ind, str):
            self._set_one_column_type(ind, tp)
        else:
            if hasattr(tp, '__len__') and not len(tp) == 1:
                if len(ind) == len(tp):
                    for label, one_tp in zip(ind, tp):
                        self._set_one_column_type(label, one_tp)
                else:
                    raise ValueError(f'Cannot set {len(ind)} columns using {len(tp)} column types')
            else:
                # Broadcasting of tp
                try:
                    len(ind)
                except TypeError:
                    ind = [ind]
                for label in ind:
                    self._set_one_column_type(label, tp)

    def _get_one_column_type(self, label):
        """Get the column type of one column"""
        if label in self._obj.columns:
            try:
                d = self._obj._column_types
            except AttributeError:
                return ColumnType.UNKNOWN
            try:
                return d[label]
            except KeyError:
                return ColumnType.UNKNOWN
        else:
            raise KeyError(str(label))

    def __getitem__(self, ind):
        if isinstance(ind, str):
            return self._get_one_column_type(ind)
        else:
            try:
                return [self._get_one_column_type(label) for label in ind]
            except TypeError:
                return self._get_one_column_type(ind)


class LabelsByTypeIndexer:
    """Indexing a PharmDataFrame to get labels from ColumnTypes"""

    def __init__(self, acc):
        self._acc = acc

    def _get_one_label(self, tp):
        labels = self._get_many_labels(tp)
        if len(labels) == 0:
            raise KeyError(str(tp))
        elif len(labels) > 1:
            raise KeyError('Did not expect more than one ' + str(tp))
        else:
            return labels

    def _get_many_labels(self, column_type):
        """Will raise if no columns of the type exists
        Always returns a list of labels
        """
        return [
            label for label in self._acc._obj.columns if self._acc.column_type[label] == column_type
        ]

    def __getitem__(self, tp):
        try:
            len(tp)
            labels = []
            for t in tp:
                if t.max_one:
                    labels.extend(self._get_one_label(t))
                else:
                    labels.extend(self._get_many_labels(t))
            return labels
        except TypeError:
            if tp.max_one:
                return self._get_one_label(tp)
            else:
                return self._get_many_labels(tp)


@pd.api.extensions.register_dataframe_accessor('pharmpy')
class DataFrameAccessor:
    def __init__(self, obj):
        self._obj = obj

    @property
    def column_type(self):
        return ColumnTypeIndexer(self._obj)

    @property
    def labels_by_type(self):
        return LabelsByTypeIndexer(self)

    @property
    def id_label(self):
        """Return the label of the id column"""
        return self.labels_by_type[ColumnType.ID][0]

    @property
    def idv_label(self):
        """Return the label of the idv column"""
        return self.labels_by_type[ColumnType.IDV][0]

    @property
    def dv_label(self):
        """Return the label of the dv column"""
        return self.labels_by_type[ColumnType.DV][0]

    @property
    def ids(self):
        """Return the ids in the dataset"""
        return self._obj[self.id_label].unique()

    @property
    def time_varying_covariates(self):
        """Return a list of labels for all time varying covariates"""
        cov_labels = self.labels_by_type[ColumnType.COVARIATE]
        if len(cov_labels) == 0:
            return []
        else:
            time_var = self._obj.groupby(by=self.id_label)[cov_labels].nunique().gt(1).any()
            return list(time_var.index[time_var])

    @property
    def baselines(self):
        """Baselines for each id.
        Baseline is taken to be the first row even if that has a missing value.
        """
        idlab = self.id_label
        return self._obj.groupby(idlab).nth(0)

    @property
    def covariate_baselines(self):
        """Return a dataframe with baselines of all covariates for each id.
        Baseline is taken to be the first row even if that has a missing value.
        """
        covariates = self.labels_by_type[ColumnType.COVARIATE]
        idlab = self.id_label
        df = self._obj[covariates + [idlab]]
        return df.groupby(idlab).nth(0)

    @property
    def observations(self):
        """Return a series with observations. Indexed with ID and TIME"""
        try:
            label = self.labels_by_type[ColumnType.EVENT]
        except KeyError:
            try:
                label = self.labels_by_type[ColumnType.DOSE]
            except KeyError:
                raise DatasetError('Could not identify observation rows in dataset')

        label = label[0]
        idcol = self.id_label
        idvcol = self.idv_label
        df = self._obj.query(f'{label} == 0')

        if df.empty:
            df = self._obj.astype({label: 'float'})
            df = df.query(f'{label} == 0')

        df = df[[idcol, idvcol, self.dv_label]]
        df.set_index([idcol, idvcol], inplace=True)
        return df.squeeze()

    @property
    def doses(self):
        """Return a series with all doses. Indexed with ID and TIME"""
        try:
            label = self.labels_by_type[ColumnType.DOSE]
        except KeyError:
            raise DatasetError('Could not identify dosing rows in dataset')

        label = label[0]
        idcol = self.id_label
        idvcol = self.idv_label
        df = self._obj.query(f'{label} != 0')
        df = df[[idcol, idvcol, label]]
        df.set_index([idcol, idvcol], inplace=True)
        return df.squeeze()

    def add_doseid(self):
        """Add a column DOSEID with id of each dose period starting from 1"""
        try:
            dose = self.labels_by_type[ColumnType.DOSE]
        except KeyError:
            raise DatasetError('Could not identify dosing rows in dataset')
        df = self._obj
        df['DOSEID'] = df[dose]
        df.loc[df['DOSEID'] > 0, 'DOSEID'] = 1
        df['DOSEID'] = df['DOSEID'].astype(int)
        df['DOSEID'] = df.groupby(self.id_label)['DOSEID'].cumsum()

    def add_time_after_dose(self):
        """Calculate and add a TAD column to the dataset"""
        # FIXME: Should not rely on name here. Use coltypes for TAD and DOSEID
        # FIXME: TIME is converted to float. Should be handled when reading in dataset
        df = self._obj
        if 'DOSEID' in df.columns:
            had_doseid = True
        else:
            self.add_doseid()
            had_doseid = False
        idv = self.idv_label
        idlab = self.id_label
        df[idv] = df[idv].astype(np.float64)
        df['TAD'] = df.groupby([idlab, 'DOSEID'])[idv].diff().fillna(0)
        df['TAD'] = df.groupby([idlab, 'DOSEID'])['TAD'].cumsum()
        if not had_doseid:
            df.drop('DOSEID', axis=1, inplace=True)

    def concentration_parameters(self):
        """Create a dataframe with concentration parameters"""
        df = self._obj.copy()
        df.pharmpy.add_doseid()
        df.pharmpy.add_time_after_dose()
        idlab = self.id_label
        dv = self.dv_label
        noobs = df.groupby([idlab, 'DOSEID']).size() == 1
        idx = df.groupby([idlab, 'DOSEID'])[dv].idxmax()
        params = df.loc[idx].set_index([idlab, 'DOSEID'])
        params = params[[dv, 'TAD']]
        params.rename(columns={dv: 'Cmax', 'TAD': 'Tmax'}, inplace=True)
        params.loc[noobs] = np.nan

        grpind = df.groupby(['ID', 'DOSEID']).indices
        keep = []
        for ind, rows in grpind.items():
            index = idx.loc[ind]
            p = params.loc[ind]
            if not np.isnan(p['Tmax']):
                keep += [row for row in rows if row > index]
        minidx = df.iloc[keep].groupby([idlab, 'DOSEID'])[dv].idxmin()
        params2 = df.loc[minidx].set_index([idlab, 'DOSEID'])
        params2 = params2[[dv, 'TAD']]
        params2.rename(columns={dv: 'Cmin', 'TAD': 'Tmin'}, inplace=True)
        res = params.join(params2)
        return res

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
