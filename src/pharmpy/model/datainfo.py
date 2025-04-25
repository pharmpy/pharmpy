"""DataInfo is a companion to the dataset. It contains metadata of the dataset"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union, cast, overload

from pharmpy import conf
from pharmpy.basic import Expr, TUnit, Unit
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.path import path_absolute, path_relative_to
from pharmpy.internals.immutable import Immutable, frozenmapping


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
    categories : Optional[Union[tuple, dict]]
        Tuple of all possible categories or dict from value to label for each category
    drop : bool
        Should column be dropped (i.e. barred from being used)
    datatype : str
        Pandas datatype or special Pharmpy datatype (see the "dtype" attribute)
    descriptor : str
        Descriptor (kind) of data
    """

    _all_types = (
        'id',
        'dv',
        'dvid',
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
        'admid',
        'lloq',
        'blq',
    )
    _all_scales = ('nominal', 'ordinal', 'interval', 'ratio')
    _all_dtypes = (
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
    )
    _all_descriptors = (
        None,
        'age',
        'body height',
        'body weight',
        'body surface area',
        'lean body mass',
        'fat free mass',
        'time after dose',
        'plasma concentration',
        'subject identifier',
        'observation identifier',
        'pk measurement',
        'pd measurement',
    )

    @staticmethod
    def convert_pd_dtype_to_datatype(dtype) -> str:
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
    def convert_datatype_to_pd_dtype(datatype) -> str:
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
        name: str,
        type: str = 'unknown',
        unit: Unit = Unit.unitless(),
        scale: str = 'ratio',
        continuous: Optional[bool] = None,
        categories: Optional[Union[frozenmapping[str, str], tuple[str, ...]]] = None,
        drop: bool = False,
        datatype: str = "float64",
        descriptor: Optional[str] = None,
    ):
        self._name = name
        self._type = type
        self._unit = unit
        self._scale = scale
        self._continuous = continuous
        self._categories = categories
        self._drop = drop
        self._datatype = datatype
        self._descriptor = descriptor

    @staticmethod
    def _canonicalize_categories(
        categories: Union[Mapping[str, str], Sequence[str], None],
    ) -> Union[frozenmapping[str, str], tuple[str, ...], None]:
        if isinstance(categories, dict):
            return frozenmapping(categories)
        elif isinstance(categories, frozenmapping):
            return categories
        elif isinstance(categories, tuple):
            return categories
        elif isinstance(categories, Sequence):
            return tuple(categories)
        elif categories is None:
            return categories
        else:
            raise TypeError("categories must be None, list-like or dict-like")

    @classmethod
    def create(
        cls,
        name: str,
        type: str = 'unknown',
        unit: Optional[TUnit] = None,
        scale: str = 'ratio',
        continuous: Optional[bool] = None,
        categories: Optional[Union[Mapping[str, str], Sequence[str]]] = None,
        drop: bool = False,
        datatype: str = "float64",
        descriptor: Optional[str] = None,
    ):
        if scale in ('nominal', 'ordinal'):
            if continuous is True:
                raise ValueError("A nominal or ordinal column cannot be continuous")
            else:
                continuous = False
        if continuous is None:
            continuous = True
        if not isinstance(name, str):
            raise TypeError("Column name must be a string")
        if type not in ColumnInfo._all_types:
            raise TypeError(f"Unknown column type {type}")
        if scale not in ColumnInfo._all_scales:
            raise TypeError(
                f"Unknown scale of measurement {scale}. Only {ColumnInfo._all_scales} are possible."
            )
        if unit is None:
            unit = Unit.unitless()
        else:
            unit = Unit(unit)
        if datatype not in ColumnInfo._all_dtypes:
            raise ValueError(
                f"{datatype} is not a valid datatype. Valid datatypes are {ColumnInfo._all_dtypes}"
            )
        if descriptor not in ColumnInfo._all_descriptors:
            raise TypeError(f"Unknown column descriptor {descriptor}")
        categories = ColumnInfo._canonicalize_categories(categories)

        return cls(
            name=name,
            type=type,
            unit=unit,
            scale=scale,
            continuous=continuous,
            categories=categories,
            drop=drop,
            datatype=datatype,
            descriptor=descriptor,
        )

    def replace(self, **kwargs) -> ColumnInfo:
        """Replace properties and create a new ColumnInfo"""
        d = {key[1:]: value for key, value in self.__dict__.items()}
        d.update(kwargs)
        new = ColumnInfo.create(**d)
        return new

    def __eq__(self, other: Any):
        if self is other:
            return True
        if not isinstance(other, ColumnInfo):
            return NotImplemented
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

    def __hash__(self):
        return hash(
            (
                self._name,
                self._type,
                self._unit,
                self._scale,
                self._continuous,
                # FIXME: What are categories really?
                # self._categories,
                self._drop,
                self._datatype,
                self._descriptor,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self._name,
            'type': self._type,
            'unit': self._unit.serialize(),
            'scale': self._scale,
            'continuous': self._continuous,
            'categories': self._categories,
            'drop': self._drop,
            'datatype': self._datatype,
            'descriptor': self._descriptor,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnInfo:
        return cls(
            name=d['name'],
            type=d['type'],
            unit=Unit.deserialize(d['unit']),
            scale=d['scale'],
            continuous=d['continuous'],
            categories=d['categories'],
            drop=d['drop'],
            datatype=d['datatype'],
            descriptor=d['descriptor'],
        )

    @property
    def name(self) -> str:
        """Column name"""
        return self._name

    @property
    def symbol(self) -> Expr:
        """Symbol having the column name"""
        return Expr.symbol(self._name)

    @property
    def type(self) -> str:
        """Type of column

        ============  =============
        type          Description
        ============  =============
        id            Individual identifier. Max one per DataFrame. All values have to be unique
        idv           Independent variable. Max one per DataFrame.
        dv            Observations of the dependent variable
        dvid          Dependent variable ID
        covariate     Covariate
        dose          Dose amount
        rate          Rate of infusion
        additional    Number of additional doses
        ii            Interdose interval
        ss            Steady state dosing
        event         0 = observation
        mdv           0 = DV is observation value, 1 = DV is missing
        admid         Administration ID
        compartment   Compartment information (not yet exactly specified)
        lloq          Lower limit of quantification
        blq           Below limit of quantification indicator
        unknown       Unkown type. This will be the default for columns that hasn't been
                      assigned a type
        ============  =============
        """
        return self._type

    @property
    def descriptor(self) -> Optional[str]:
        """Kind of data

        ====================== ============================================
        descriptor             Description
        ====================== ============================================
        age                    Age (since birth)
        body height            Human body height
        body surface area      Body surface area (calculated)
        body weight            Human body weight
        lean body mass         Lean body mass
        fat free mass          Fat free mass
        time after dose        Time after dose
        plasma concentration   Concentration of substance in blood plasma
        subject identifier     Unique integer identifier for a subject
        observation identifier Unique integer identifier for an observation
        pk measurement         Any kind of PK measurement
        pd measurement         Any kind of PD measurement
        ====================== ============================================
        """
        return self._descriptor

    @property
    def unit(self) -> Unit:
        """Unit of the column data

        Custom units are allowed, but units that are available in sympy.physics.units can be
        recognized. The default unit is 1, i.e. without unit.
        """
        return self._unit

    @property
    def scale(self) -> str:
        """Scale of measurement

        The statistical scale of measurement for the column data. Can be one of
        'nominal', 'ordinal', 'interval' and 'rational'.
        """
        return self._scale

    @property
    def continuous(self) -> Optional[bool]:
        """Is the column data continuous

        True for continuous data and False for discrete. Note that nominal and ordinal data have to
        be discrete.
        """
        return self._continuous

    @property
    def categories(self) -> Optional[Union[frozenmapping[str, str], tuple[str, ...]]]:
        """List or dict of allowed categories"""
        return self._categories

    @property
    def drop(self) -> bool:
        """Should this column be dropped"""
        return self._drop

    @property
    def datatype(self) -> str:
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

    def is_categorical(self) -> bool:
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
        >>> col1 = ColumnInfo.create("WGT", scale='ratio')
        >>> col1.is_categorical()
        False
        >>> col2 = ColumnInfo.create("ID", scale='nominal')
        >>> col2.is_categorical()
        True

        """
        return self.scale in ['nominal', 'ordinal']

    def is_numerical(self) -> bool:
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
        >>> col1 = ColumnInfo.create("WGT", scale='ratio')
        >>> col1.is_numerical()
        True
        >>> col2 = ColumnInfo.create("ID", scale='nominal')
        >>> col2.is_numerical()
        False

        """
        return self.scale in ['interval', 'ratio']

    def is_integer(self) -> bool:
        """Check if the column datatype is integral

        Returns
        -------
        bool
            True if of integral datatype

        See also
        --------
        is_categorical : Check if the column data is categorical

        Examples
        --------
        >>> from pharmpy.model import ColumnInfo
        >>> col1 = ColumnInfo.create("WGT", scale='ratio')
        >>> col1.is_integer()
        False
        """
        return self.datatype in [
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
        ]

    def get_all_categories(self) -> list[str]:
        """Get a list of all categories"""
        if isinstance(self._categories, tuple):
            return list(self._categories)
        elif self._categories is None:
            return []
        else:
            return list(self._categories.keys())

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
    missing_data_token : str
        Token for missing data
    """

    def __init__(
        self,
        columns: tuple[ColumnInfo, ...] = (),
        path: Optional[Path] = None,
        separator: str = ',',
        missing_data_token: Optional[str] = None,
    ):
        self._columns = columns
        self._path = path
        self._separator = separator
        if missing_data_token is None:
            self._missing_data_token = conf.missing_data_token
        else:
            self._missing_data_token = missing_data_token

    @classmethod
    def create(
        cls,
        columns: Optional[Union[Sequence[ColumnInfo], Sequence[str]]] = None,
        path: Optional[Union[str, Path]] = None,
        separator: str = ',',
        missing_data_token: Optional[str] = None,
    ):
        if columns:
            if not isinstance(columns, Sequence):
                raise TypeError('Argument `columns` must be iterable')
            if not all(isinstance(col, str) or isinstance(col, ColumnInfo) for col in columns):
                raise TypeError(
                    'Argument `columns` need to consist of either type `str` or `ColumnInfo`'
                )
        if columns is None or len(columns) == 0:
            cols = ()
        elif len(columns) > 0 and any(isinstance(col, str) for col in columns):
            cols = tuple(ColumnInfo.create(col) if isinstance(col, str) else col for col in columns)
        else:
            cols = cast(tuple[ColumnInfo, ...], tuple(columns))
        if path is not None:
            path = Path(path)
        if missing_data_token is None:
            missing_data_token = conf.missing_data_token
        return cls(
            columns=cols, path=path, separator=separator, missing_data_token=str(missing_data_token)
        )

    def replace(self, **kwargs) -> DataInfo:
        if 'columns' in kwargs:
            columns = tuple(kwargs['columns'])
        else:
            columns = self._columns
        if 'path' in kwargs:
            if kwargs['path'] is not None:
                path = Path(kwargs['path'])
            else:
                path = None
        else:
            path = self._path
        separator = kwargs.get('separator', self._separator)
        missing_data_token = kwargs.get('missing_data_token', self._missing_data_token)
        return DataInfo.create(
            columns=columns,
            path=path,
            separator=separator,
            missing_data_token=str(missing_data_token),
        )

    def __add__(self, other: DataInfo) -> DataInfo:
        if isinstance(other, DataInfo):
            return DataInfo.create(
                columns=self._columns + other._columns, path=self.path, separator=self.separator
            )
        elif isinstance(other, ColumnInfo):
            return DataInfo.create(
                columns=self._columns + (other,), path=self.path, separator=self.separator
            )
        else:
            return DataInfo.create(
                columns=self._columns + tuple(other), path=self.path, separator=self.separator
            )

    def __radd__(self, other: DataInfo) -> DataInfo:
        if isinstance(other, ColumnInfo):
            return DataInfo.create(
                columns=(other,) + self._columns, path=self.path, separator=self.separator
            )
        else:
            return DataInfo.create(
                columns=tuple(other) + self._columns, path=self.path, separator=self.separator
            )

    def __eq__(self, other: Any):
        if self is other:
            return True
        if not isinstance(other, DataInfo):
            return NotImplemented
        if len(self) != len(other):
            return False
        for col1, col2 in zip(self, other):
            if col1 != col2:
                return False
        return True

    def __hash__(self):
        return hash(self._columns)

    def __len__(self):
        return len(self._columns)

    def _getindex(self, i: Union[int, str]) -> int:
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
    def __getitem__(self, index: Union[list, slice]) -> DataInfo: ...

    @overload
    def __getitem__(self, index: Union[int, str]) -> ColumnInfo: ...

    def __getitem__(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, index: Union[list, slice, int, str]
    ) -> Union[DataInfo, ColumnInfo]:
        if isinstance(index, list):
            cols = []
            for ind in index:
                i = self._getindex(ind)
                cols.append(self._columns[i])
            return DataInfo.create(columns=cols)
        if isinstance(index, slice):
            return DataInfo.create(self._columns[index], path=self._path, separator=self._separator)

        return self._columns[self._getindex(index)]

    def __contains__(self, value: Any) -> bool:
        for col in self:
            if col == value or col.name == value:
                return True
        return False

    @property
    def path(self) -> Optional[Path]:
        r"""Path of dataset file

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> str(model.datainfo.path).replace('\\', '/')     # doctest: +ELLIPSIS
        '.../pharmpy/internals/example_models/pheno.dta'
        """
        return self._path

    @property
    def separator(self) -> str:
        """Separator for dataset file

        Can be a single character or a regular expression
        string.
        """
        return self._separator

    @property
    def missing_data_token(self) -> str:
        """Token for missing data"""
        return self._missing_data_token

    @property
    def typeix(self) -> TypeIndexer:
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
    def descriptorix(self) -> DescriptorIndexer:
        """Descriptor indexer

        Example
        -------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.descriptorix['body weight'].names
        ['WGT']
        """
        return DescriptorIndexer(self)

    def set_column(self, col: ColumnInfo) -> DataInfo:
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
        return self.replace(columns=newcols)

    @property
    def id_column(self) -> ColumnInfo:
        """The id column

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.id_column.name
        'ID'
        """
        return self.typeix['id'][0]

    def _set_column_type(self, name: str, type: str) -> DataInfo:
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

        newcol = mycol.replace(type=type)
        cols = self._columns[0:ind] + (newcol,) + self._columns[ind + 1 :]
        return DataInfo.create(cols, path=self._path, separator=self._separator)

    def set_id_column(self, name: str) -> DataInfo:
        return self._set_column_type(name, 'id')

    @property
    def dv_column(self) -> ColumnInfo:
        """The dv column

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.dv_column.name
        'DV'
        """
        return self.typeix['dv'][0]

    def set_dv_column(self, name: str) -> DataInfo:
        return self._set_column_type(name, 'dv')

    @property
    def idv_column(self) -> ColumnInfo:
        """The idv column

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.idv_column.name
        'TIME'
        """
        return self.typeix['idv'][0]

    def set_idv_column(self, name: str) -> DataInfo:
        return self._set_column_type(name, 'idv')

    @property
    def names(self) -> list[str]:
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
    def types(self) -> list[str]:
        """All column types"""
        return [col.type for col in self._columns]

    def set_types(self, value: Union[list[str], str]) -> DataInfo:
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
            newcol = col.replace(type=v)
            newcols.append(newcol)
        return DataInfo.create(columns=newcols, path=self._path, separator=self._separator)

    def find_single_column_name(self, type: str, default: Optional[str] = None) -> str:
        """Find name of single column given type

        Finds single column name with a given type, else provided default. Raises
        if more than one column is found or if no column is found and no default is
        given.

        Parameters
        ----------
        type : str
            Column type
        default : Optional[str]
            Default if column type is not found

        Return
        ------
        str
            Name of column
        """
        try:
            col = self.typeix[type]
        except IndexError:
            if default:
                return default
            raise ValueError(f'Colum of type {type} not found and no default given')
        if len(col) > 1:
            raise ValueError(f'More than one column found: {col.names}')
        return col[0].name

    def get_dtype_dict(self) -> dict[str, str]:
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
            col.name: (
                col.datatype if not col.drop and not col.datatype.startswith('nmtran') else 'str'
            )
            for col in self
        }

    def to_dict(self) -> dict[str, Any]:
        return self._to_dict(path=None)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataInfo:
        columns = tuple(ColumnInfo.from_dict(col) for col in d['columns'])
        # For backwards compatibility
        missing_data_token = d.get('missing_data_token', conf.missing_data_token)
        return cls(
            columns=columns,
            path=d['path'],
            separator=d['separator'],
            missing_data_token=missing_data_token,
        )

    def _to_dict(self, path: Optional[str]) -> dict[str, Any]:
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
                "descriptor": col.descriptor,
            }
            a.append(d)

        return {
            "columns": a,
            "path": path,
            "separator": self._separator,
            "missing_data_token": self._missing_data_token,
        }

    def to_json(self, path: Optional[Union[Path, str]] = None):
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
    def from_json(s: str) -> DataInfo:
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
            ci = ColumnInfo.create(
                name=col['name'],
                type=col.get('type', 'unknown'),
                scale=col['scale'],
                continuous=col.get('continuous', None),
                unit=col.get('unit', Unit.unitless()),
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
        di = DataInfo.create(columns, path=path, separator=separator)
        return di

    @staticmethod
    def read_json(path: Union[Path, str]) -> DataInfo:
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
            else di.replace(path=path_absolute(Path(path).parent / di.path))
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

    def __getitem__(self, i) -> DataInfo:
        cols = [col for col in self._obj if col.type == i and not col.drop]
        if not cols:
            raise IndexError(f"No columns of type {i} available")
        return DataInfo.create(cols)


class DescriptorIndexer:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, i) -> DataInfo:
        cols = [col for col in self._obj if col.descriptor == i and not col.drop]
        if not cols:
            raise IndexError(f"No columns with descriptor {i} available")
        return DataInfo.create(cols)
