"""DataInfo is a companion to the dataset. It contains metadata of the dataset
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from typing import Sequence as TypingSequence
from typing import Tuple, Union, cast, overload

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.expr.units import parse as parse_units
from pharmpy.internals.fs.path import path_absolute, path_relative_to
from pharmpy.internals.immutable import Immutable


class ColumnInfo(Immutable):
    """Information about one data column

    Parameters
    ----------
    name : str
        Colum name
    type : str
        Type (see the "type" attribute)
    unit : str
        Unit (see the "unit" attribute)
    scale : str
        Scale of measurement (see the "scale" attribute)
    continuous : bool
        True if continuous or False if discrete
    categories : list
        List of all possible categories
    drop : bool
        Should column be dropped (i.e. barred from being used)
    datatype : str
        Pandas datatype or special Pharmpy datatype (see the "dtype" attribute)
    descriptor : str
        Descriptor (kind) of data
    """

    _all_types = [
        'id',
        'dv',
        'idv',
        'unknown',
        'dose',
        'rate',
        'additional',
        'ii',
        'ss',
        'event',
        'covariate',
        'mdv',
        'compartment',
    ]
    _all_scales = ['nominal', 'ordinal', 'interval', 'ratio']
    _all_dtypes = [
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
        'uint32',
        'uint64',
        'float16',
        'float32',
        'float64',
        'float128',
        'nmtran-time',
        'nmtran-date',
        'str',
    ]
    _all_descriptors = [
        None,
        'age',
        'body weight',
        'lean body mass',
        'fat free mass',
        'time after dose',
        'plasma concentration',
        'subject identifier',
    ]

    @staticmethod
    def convert_pd_dtype_to_datatype(dtype):
        """Convert pandas dtype to Pharmpy datatype

        Parameters
        ----------
        dtype : str
            String representing a pandas dtype

        Returns
        -------
        str
            String representing a Pharmpy datatype

        Examples
        --------
        >>> from pharmpy.model import ColumnInfo
        >>> ColumnInfo.convert_pd_dtype_to_datatype("float64")
        'float64'
        """
        return dtype if dtype in ColumnInfo._all_dtypes else 'str'

    @staticmethod
    def convert_datatype_to_pd_dtype(datatype):
        """Convert Pharmpy datatype to pandas dtype

        Parameters
        ----------
        datatype : str
            String representing a Pharmpy datatype

        Returns
        -------
        str
            String representing a pandas dtype

        Examples
        --------
        >>> from pharmpy.model import ColumnInfo
        >>> ColumnInfo.convert_datatype_to_pd_dtype("float64")
        'float64'
        >>> ColumnInfo.convert_datatype_to_pd_dtype("nmtran-date")
        'str'
        """
        if datatype.startswith('nmtran'):
            return 'str'
        else:
            return datatype

    def __init__(
        self,
        name,
        type='unknown',
        unit=None,
        scale='ratio',
        continuous=None,
        categories=None,
        drop=False,
        datatype="float64",
        descriptor=None,
    ):
        if scale in ('nominal', 'ordinal'):
            if continuous is True:
                raise ValueError("A nominal or ordinal column cannot be continuous")
            else:
                continuous = False
        if continuous is None:
            continuous = True
        self._continuous = continuous
        if not isinstance(name, str):
            raise TypeError("Column name must be a string")
        self._name = name
        if type not in ColumnInfo._all_types:
            raise TypeError(f"Unknown column type {type}")
        self._type = type
        if scale not in ColumnInfo._all_scales:
            raise TypeError(
                f"Unknown scale of measurement {scale}. Only {ColumnInfo._all_scales} are possible."
            )
        self._unit = sympy.Integer(1) if unit is None else parse_units(unit)
        self._scale = scale
        self._continuous = continuous
        self._categories = categories  # dict from value to descriptive string
        self._drop = drop
        if datatype not in ColumnInfo._all_dtypes:
            raise ValueError(
                f"{datatype} is not a valid datatype. Valid datatypes are {ColumnInfo._all_dtypes}"
            )
        self._datatype = datatype
        if descriptor not in ColumnInfo._all_descriptors:
            raise TypeError(f"Unknown column descriptor {descriptor}")
        self._descriptor = descriptor

    def derive(self, **kwargs):
        """Derive a new ColumnInfo with new properties"""
        d = {key[1:]: value for key, value in self.__dict__.items()}
        d.update(kwargs)
        new = ColumnInfo(**d)
        return new

    def __eq__(self, other):
        return (
            self._name == other._name
            and self._type == other._type
            and self._unit == other._unit
            and self._scale == other._scale
            and self._continuous == other._continuous
            and self._categories == other._categories
            and self._drop == other._drop
            and self._datatype == other._datatype
        )

    @property
    def name(self):
        """Column name"""
        return self._name

    @property
    def type(self):
        """Type of column

        ============  =============
        type          Description
        ============  =============
        id            Individual identifier. Max one per DataFrame. All values have to be unique
        idv           Independent variable. Max one per DataFrame.
        dv            Dependent variable
        covariate     Covariate
        dose          Dose amount
        rate          Rate of infusion
        additional    Number of additional doses
        ii            Interdose interval
        ss            Steady state dosing
        event         0 = observation
        mdv           0 = DV is observation value, 1 = DV is missing
        compartment   Compartment information (not yet exactly specified)
        unknown       Unkown type. This will be the default for columns that hasn't been
                      assigned a type
        ============  =============
        """
        return self._type

    @property
    def descriptor(self):
        """Kind of data

        ==================== ==========================================
        descriptor           Description
        ==================== ==========================================
        age                  Age (since birth)
        body weight          Human body weight
        lean body mass       Lean body mass
        fat free mass        Fat free mass
        time after dose      Time after dose
        plasma concentration Concentration of substance in blood plasma
        subject identifier   Unique integer identifier for a subject
        ==================== ==========================================
        """
        return self._descriptor

    @property
    def unit(self):
        """Unit of the column data

        Custom units are allowed, but units that are available in sympy.physics.units can be
        recognized. The default unit is 1, i.e. without unit.
        """
        return self._unit

    @property
    def scale(self):
        """Scale of measurement

        The statistical scale of measurement for the column data. Can be one of
        'nominal', 'ordinal', 'interval' and 'rational'.
        """
        return self._scale

    @property
    def continuous(self):
        """Is the column data continuous

        True for continuous data and False for discrete. Note that nominal and ordinal data have to
        be discrete.
        """
        return self._continuous

    @property
    def categories(self):
        """List of allowed categories"""
        return self._categories

    @property
    def drop(self):
        """Should this column be dropped"""
        return self._drop

    @property
    def datatype(self):
        """Column datatype

        ============ ================ ======== ================================= ===========
        datatype     Description      Size     Range                             NA allowed?
        ============ ================ ======== ================================= ===========
        int8         Signed integer   8 bits   -128 to +127.                     No
        int16        Signed integer   16 bits  -32,768 to +32,767.               No
        int32        Signed integer   32 bits  -2,147,483,648 to +2,147,483,647. No
        int64        Signed integer   64 bits  -9,223,372,036,854,775,808 to     No
                                               9,223,372,036,854,775,807.
        uint8        Unsigned integer 8 bits   0 to 256.                         No
        uint16       Unsigned integer 16 bit   0 to 65,535.                      No
        uint32       Unsigned integer 32 bit   0 to 4,294,967,295.               No
        uint64       Unsigned integer 64 bit   0 to 18,446,744,073,709,551,615   No
        float16      Binary float     16 bits  ≈ ±6.55×10⁴                       Yes
        float32      Binary float     32 bits  ≈ ±3.4×10³⁸                       Yes
        float64      Binary float     64 bits  ≈ ±1.8×10³⁰⁸                      Yes
        float128     Binary float     128 bits ≈ ±1.2×10⁴⁹³²                     Yes
        nmtran-time  NM-TRAN time     n                                          No
        nmtran-date  NM-TRAN date     n                                          No
        str          General string   n                                          No
        ============ ================ ========================================== ===========

        The default, and most common datatype, is float64.
        """
        return self._datatype

    def is_categorical(self):
        """Check if the column data is categorical

        Returns
        -------
        bool
            True if categorical (nominal or ordinal) and False otherwise.

        See also
        --------
        is_numerical : Check if the column data is numerical

        Examples
        --------
        >>> from pharmpy.model import ColumnInfo
        >>> col1 = ColumnInfo("WGT", scale='ratio')
        >>> col1.is_categorical()
        False
        >>> col2 = ColumnInfo("ID", scale='nominal')
        >>> col2.is_categorical()
        True

        """
        return self.scale in ['nominal', 'ordinal']

    def is_numerical(self):
        """Check if the column data is numerical

        Returns
        -------
        bool
            True if numerical (interval or ratio) and False otherwise.

        See also
        --------
        is_categorical : Check if the column data is categorical

        Examples
        --------
        >>> from pharmpy.model import ColumnInfo
        >>> col1 = ColumnInfo("WGT", scale='ratio')
        >>> col1.is_numerical()
        True
        >>> col2 = ColumnInfo("ID", scale='nominal')
        >>> col2.is_numerical()
        False

        """
        return self.scale in ['interval', 'ratio']

    def __repr__(self):
        ser = pd.Series(
            [
                self._type,
                self._scale,
                self._continuous,
                self._categories,
                self._unit,
                self._drop,
                self._datatype,
                self._descriptor,
            ],
            index=[
                'type',
                'scale',
                'continuous',
                'categories',
                'unit',
                'drop',
                'datatype',
                'descriptor',
            ],
            name=self._name,
        )
        return ser.to_string(name=True)


class DataInfo(Sequence, Immutable):
    """Metadata for the dataset

    Can be indexed to get ColumnInfo for the columns.

    Parameters
    ----------
    columns : list
        List of column names
    path : Path
        Path to dataset file
    separator : str
        Character or regexp separator for dataset
    """

    def __init__(
        self,
        columns: Optional[Union[TypingSequence[ColumnInfo], TypingSequence[str]]] = None,
        path: Optional[Union[str, Path]] = None,
        separator: str = ',',
        force_absolute_path: bool = True,
    ):
        if columns is None:
            self._columns: Tuple[ColumnInfo, ...] = ()
        elif len(columns) > 0 and isinstance(columns[0], str):
            self._columns = tuple(map(ColumnInfo, columns))
        else:
            self._columns = cast(Tuple[ColumnInfo, ...], tuple(columns))
        if path is not None:
            path = Path(path)
        assert not force_absolute_path or path is None or path.is_absolute()
        self._path = path
        self._separator = separator

    def derive(self, **kwargs):
        columns = kwargs.get('columns', self._columns)
        path = kwargs.get('path', self._path)
        separator = kwargs.get('separator', self._separator)
        return DataInfo(columns=columns, path=path, separator=separator)

    def __add__(self, other):
        if isinstance(other, DataInfo):
            return DataInfo(
                columns=self._columns + other._columns, path=self.path, separator=self.separator
            )
        elif isinstance(other, ColumnInfo):
            return DataInfo(
                columns=self._columns + (other,), path=self.path, separator=self.separator
            )
        else:
            return DataInfo(
                columns=self._columns + tuple(other), path=self.path, separator=self.separator
            )

    def __radd__(self, other):
        if isinstance(other, ColumnInfo):
            return DataInfo(
                columns=(other,) + self._columns, path=self.path, separator=self.separator
            )
        else:
            return DataInfo(
                columns=tuple(other) + self._columns, path=self.path, separator=self.separator
            )

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
            raise IndexError(f"Cannot find column {i} in DataInfo")
        elif isinstance(i, int):
            return i
        else:
            raise TypeError(f"Cannot index DataInfo by {type(i)}")

    @overload
    def __getitem__(self, i: Union[list, slice]) -> DataInfo:
        ...

    @overload
    def __getitem__(self, i: Union[int, str]) -> ColumnInfo:
        ...

    def __getitem__(self, i):
        if isinstance(i, list):
            cols = []
            for ind in i:
                index = self._getindex(ind)
                cols.append(self._columns[index])
            return DataInfo(columns=cols)
        if isinstance(i, slice):
            return DataInfo(self._columns[i], path=self._path, separator=self._separator)

        return self._columns[self._getindex(i)]

    @property
    def path(self):
        r"""Path of dataset file

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> str(model.datainfo.path).replace('\\', '/')     # doctest: +ELLIPSIS
        '.../pharmpy/modeling/example_models/pheno.dta'
        """
        return self._path

    @property
    def separator(self):
        """Separator for dataset file

        Can be a single character or a regular expression
        string.
        """
        return self._separator

    @property
    def typeix(self):
        """Type indexer

        Example
        -------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.typeix['covariate'].names
        ['WGT', 'APGR']
        """
        return TypeIndexer(self)

    @property
    def descriptorix(self):
        """Descriptor indexer

        Example
        -------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.descriptorix['body weight'].names
        ['WGT']
        """
        return DescriptorIndexer(self)

    def set_column(self, col):
        """Set ColumnInfo of an existing column of the same name

        Parameters
        ----------
        col : ColumnInfo
            New ColumnInfo

        Returns
        -------
        DataInfo
            Updated DataInfo
        """
        newcols = []
        for cur in self:
            if cur.name != col.name:
                newcols.append(cur)
            else:
                newcols.append(col)
        return self.derive(columns=newcols)

    @property
    def id_column(self):
        """The id column

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.id_column.name
        'ID'
        """
        return self.typeix['id'][0]

    def _set_column_type(self, name, type):
        for i, col in enumerate(self):
            if col.name != name and col.type == type:
                raise ValueError(
                    f"Cannot set new {type} column: column {col.name} already has type {type}"
                )

        for i, col in enumerate(self):
            if col.name == name:
                mycol = col
                ind = i
                break
        else:
            raise IndexError(f"No column {name} in DataInfo")

        newcol = mycol.derive(type=type)
        cols = self._columns[0:ind] + (newcol,) + self._columns[ind + 1 :]
        return DataInfo(cols, path=self._path, separator=self._separator)

    def set_id_column(self, name):
        return self._set_column_type(name, 'id')

    @property
    def dv_column(self):
        """The dv column

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.dv_column.name
        'DV'
        """
        return self.typeix['dv'][0]

    def set_dv_column(self, name):
        return self._set_column_type(name, 'dv')

    @property
    def idv_column(self):
        """The idv column

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.idv_column.name
        'TIME'
        """
        return self.typeix['idv'][0]

    def set_idv_column(self, name):
        return self._set_column_type(name, 'idv')

    @property
    def names(self):
        """All column names

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.names
        ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
        """
        return [col.name for col in self._columns]

    @property
    def types(self):
        """All column types"""
        return [col.type for col in self._columns]

    def set_types(self, value):
        """Set types for all columns

        Parameters
        ----------
        value : list or str
            Types to set. If only one this will be broadcast

        Return
        ------
        DataInfo
            Updated datainfo
        """
        if isinstance(value, str):
            value = [value]
        if len(value) == 1:
            value *= len(self)
        if len(value) != len(self):
            raise ValueError(
                "Length mismatch. "
                "Can only set the same number of names as columns or 1 for broadcasting"
            )
        newcols = []
        for v, col in zip(value, self._columns):
            newcol = col.derive(type=v)
            newcols.append(newcol)
        return DataInfo(columns=newcols, path=self._path, separator=self._separator)

    def get_dtype_dict(self):
        """Create a dictionary from column names to pandas dtypes

        This can be used as input to some pandas functions to convert
        column to the correct pandas dtype.

        Returns
        -------
        dict
            Column name to pandas dtype

        Examples
        --------
        >>> from pharmpy.modeling import *
        >>> model = load_example_model("pheno")
        >>> model.datainfo.get_dtype_dict()
        {'ID': 'int32',
         'TIME': 'float64',
         'AMT': 'float64',
         'WGT': 'float64',
         'APGR': 'float64',
         'DV': 'float64',
         'FA1': 'float64',
         'FA2': 'float64'}
        """
        return {
            col.name: col.datatype
            if not col.drop and not col.datatype.startswith('nmtran')
            else 'str'
            for col in self
        }

    def _to_dict(self, path: Optional[str]):
        a = []
        for col in self._columns:
            d = {
                "name": col.name,
                "type": col.type,
                "scale": col.scale,
                "continuous": col.continuous,
                "categories": col.categories,
                "unit": str(col.unit),
                "datatype": col.datatype,
                "drop": col.drop,
            }
            if col.descriptor is not None:
                d["descriptor"] = col.descriptor
            a.append(d)

        return {
            "columns": a,
            "path": path,
            "separator": self._separator,
        }

    def to_json(self, path=None):
        if path is None:
            return json.dumps(self._to_dict(str(self.path) if self.path is not None else None))
        else:
            with open(path, 'w') as fp:
                json.dump(
                    self._to_dict(
                        str(path_relative_to(Path(path).parent, self.path))
                        if self.path is not None
                        else None
                    ),
                    fp,
                )

    @staticmethod
    def from_json(s):
        """Create DataInfo from JSON string

        Parameters
        ----------
        s : str
            JSON string

        Return
        ------
        DataInfo
            Created DataInfo object
        """
        d = json.loads(s)
        columns = []
        for col in d['columns']:
            ci = ColumnInfo(
                name=col['name'],
                type=col.get('type', 'unknown'),
                scale=col['scale'],
                continuous=col.get('continuous', None),
                unit=col.get('unit', sympy.Integer(1)),
                categories=col.get('categories', None),
                datatype=col.get('datatype', 'float64'),
                descriptor=col.get('descriptor', None),
                drop=col.get('drop', False),
            )
            columns.append(ci)
        path = d.get('path', None)
        if path:
            path = Path(path)
        separator = d.get('separator', ',')
        di = DataInfo(columns, path=path, separator=separator, force_absolute_path=False)
        return di

    @staticmethod
    def read_json(path):
        """Read DataInfo from JSON file

        Parameters
        ----------
        path : Path or str
            Path to JSON datainfo file

        Return
        ------
        DataInfo
            Created DataInfo object
        """
        with open(path, 'r') as fp:
            s = fp.read()
        di = DataInfo.from_json(s)
        return (
            di
            if di.path is None or di.path.is_absolute()
            else di.derive(path=path_absolute(Path(path).parent / di.path))
        )

    def __repr__(self):
        labels = [col.name for col in self._columns]
        types = [col.type for col in self._columns]
        scales = [col.scale for col in self._columns]
        cont = [col.continuous for col in self._columns]
        cats = [col.categories for col in self._columns]
        units = [col.unit for col in self._columns]
        drop = [col.drop for col in self._columns]
        datatype = [col.datatype for col in self._columns]
        descriptor = [col.descriptor for col in self._columns]
        df = pd.DataFrame(
            {
                'name': labels,
                'type': types,
                'scale': scales,
                'continuous': cont,
                'categories': cats,
                'unit': units,
                'drop': drop,
                'datatype': datatype,
                'descriptor': descriptor,
            }
        )
        return df.to_string(index=False)


class TypeIndexer:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, i):
        cols = [col for col in self._obj if col.type == i and not col.drop]
        if not cols:
            raise IndexError(f"No columns of type {i} available")
        return DataInfo(cols)


class DescriptorIndexer:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, i):
        cols = [col for col in self._obj if col.descriptor == i and not col.drop]
        if not cols:
            raise IndexError(f"No columns with descriptor {i} available")
        return DataInfo(cols)
