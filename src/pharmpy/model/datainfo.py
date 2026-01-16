"""DataInfo is a companion to the dataset. It contains metadata of the dataset"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union, cast, overload

from pharmpy import conf
from pharmpy.basic import Expr, Unit
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.path import path_absolute, path_relative_to
from pharmpy.internals.immutable import Immutable, frozenmapping


class DataVariable(Immutable):
    """Information about one variable represented by data

    For long format datasets a data column can contain multiple data variables.

    Parameters
    ----------
    name : str
        Variable name. Not the same as the name of the column
    type : str
        Type of variable (see the "type" attribute)
    scale : str
        Scale of measurement (see the "scale" attribute)
    count : bool
        True if count data or False otherwise
    properties : dict
        Other properties of the variable (see the "properties" attribute)

    """

    _all_types = {
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
    }
    _all_scales = ('nominal', 'ordinal', 'interval', 'ratio')
    _all_descriptors = {
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
    }
    _all_properties = {'unit', 'categories', 'descriptor'}

    def __init__(
        self,
        name: str,
        type: str = 'unknown',
        scale: str = 'ratio',
        count: bool = False,
        properties: Mapping[str, Any] = frozenmapping({}),
    ):
        self._name = name
        self._type = type
        self._scale = scale
        self._count = count
        self._properties = properties

    @staticmethod
    def _canonicalize_properties(properties: Mapping[str, Any]) -> Mapping[str, Any]:
        new = dict(properties)
        for key, value in properties.items():
            if key == 'categories':
                new[key] = tuple(value)
            elif key == 'unit':
                new[key] = Unit(value)
            elif key == 'descriptor':
                if value not in DataVariable._all_descriptors:
                    raise ValueError(f"unknown descriptor {value}")
            else:
                raise ValueError(f'Unknown DataVariable property "{key}"')
        return frozenmapping(new)

    @classmethod
    def create(
        cls,
        name: str,
        type: str = 'unknown',
        scale: str = 'ratio',
        count: bool = False,
        properties: Mapping[str, Any] = frozenmapping({}),
    ) -> DataVariable:
        if not isinstance(name, str):
            raise TypeError("Data variable name must be a string")
        if type not in DataVariable._all_types:
            raise ValueError(f"Unknown column type {type}")
        if scale not in DataVariable._all_scales:
            raise ValueError(
                f"Unknown scale of measurement {scale}. Only {DataVariable._all_scales} are possible."
            )
        count = bool(count)
        if count and scale in {'nominal', 'ordinal'}:
            raise ValueError("A nominal or ordinal data variable cannot be count data")

        properties = DataVariable._canonicalize_properties(properties)

        return cls(
            name=name,
            type=type,
            scale=scale,
            count=count,
            properties=properties,
        )

    def replace(self, **kwargs) -> DataVariable:
        """Replace properties and create a new DataVariable"""
        d = {key[1:]: value for key, value in self.__dict__.items()}
        d.update(kwargs)
        new = DataVariable.create(**d)
        return new

    def __eq__(self, other: Any):
        if self is other:
            return True
        if not isinstance(other, DataVariable):
            return NotImplemented
        return (
            self._name == other._name
            and self._type == other._type
            and self._scale == other._scale
            and self._count == other._count
            and self._properties == other._properties
        )

    def __hash__(self):
        return hash(
            (
                self._name,
                self._type,
                self._scale,
                self._count,
                self._properties,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        properties = dict(self._properties)
        if 'unit' in properties:
            properties['unit'] = properties['unit'].serialize()

        return {
            'name': self._name,
            'type': self._type,
            'scale': self._scale,
            'count': self._count,
            'properties': properties,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataVariable:
        properties = d.get('properties', frozenmapping({}))
        if 'unit' in properties:
            properties['unit'] = Unit.deserialize(properties['unit'])
        if 'categories' in properties:
            properties['categories'] = tuple(properties['categories'])

        return cls.create(
            name=d['name'],
            type=d.get('type', 'unknown'),
            scale=d.get('scale', 'ratio'),
            count=d.get('count', False),
            properties=frozenmapping(properties),
        )

    @property
    def name(self) -> str:
        """Variable name"""
        return self._name

    @property
    def symbol(self) -> Expr:
        """Symbol having the variable name"""
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
    def scale(self) -> str:
        """Scale of measurement

        The statistical scale of measurement for the data variable. Can be one of
        'nominal', 'ordinal', 'interval' and 'rational'.
        """
        return self._scale

    @property
    def count(self) -> bool:
        """Does the data variable represent count data"""
        return self._count

    @property
    def properties(self) -> Mapping[str, Any]:
        """Other properties of the DataVariable

        descriptor

        Kind of data

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

        unit

        Unit of the data variable

        Custom units are allowed, but units that are available in sympy.physics.units can be
        recognized.

        categories

        All possible values of categorical data

        """
        return self._properties

    def is_categorical(self) -> bool:
        """Check if the data variable is categorical

        Returns
        -------
        bool
            True if categorical (nominal or ordinal) and False otherwise.

        See also
        --------
        is_numerical : Check if the data variable is numerical

        Examples
        --------
        >>> from pharmpy.model import DataVariable
        >>> var1 = DataVariable.create("WGT", scale='ratio')
        >>> var1.is_categorical()
        False
        >>> var2 = DataVariable.create("ID", scale='nominal')
        >>> var2.is_categorical()
        True

        """
        return self.scale in {'nominal', 'ordinal'}

    def is_numerical(self) -> bool:
        """Check if the data variable is numerical

        Returns
        -------
        bool
            True if numerical (interval or ratio) and False otherwise.

        See also
        --------
        is_categorical : Check if the data variable is categorical

        Examples
        --------
        >>> from pharmpy.model import DataVariable
        >>> var1 = DataVariable.create("WGT", scale='ratio')
        >>> var1.is_numerical()
        True
        >>> var2 = DataVariable.create("ID", scale='nominal')
        >>> var2.is_numerical()
        False

        """
        return self.scale in {'interval', 'ratio'}

    def get_property(self, property: str) -> Any:
        """Get a variable property with default if not defined

        Parameters
        ----------
        property : str
            The property to get

        Returns
        -------
        Any
            The value of the property or its default value

        Examples
        --------
        >>> from pharmpy.model import DataVariable
        >>> var1 = DataVariable.create("WGT", properties={"unit": "kg"})
        >>> var1.get_property("unit")
        kilogram
        >>> var2 = DataVariable.create("ID")
        >>> var2.get_property("unit")
        1
        """

        if property not in DataVariable._all_properties:
            raise ValueError(f"Unknown property {property}")

        if property == 'unit':
            default = Expr.integer(1)
        else:
            default = None
        value = self.properties.get(property, default)
        if value is None:
            raise KeyError(f"No value and no default value for property {property}")
        return value

    def set_property(self, property: str, value: Any) -> DataVariable:
        """Set the value for a property

        Parameters
        ----------
        property : str
            The property to set
        value : Any
            Value for the property

        Returns
        -------
        DataVariable
            The updated DataVariable

        Examples
        --------
        >>> from pharmpy.model import DataVariable
        >>> var1 = DataVariable.create("WGT")
        >>> var2 = var1.set_property("unit", "kg")
        >>> var2.get_property("unit")
        kilogram
        """
        props = dict(self._properties)
        props[property] = value
        return self.replace(properties=props)

    def remove_property(self, property: str) -> DataVariable:
        """Remove a property

        Parameters
        ----------
        property : str
            The property to remove

        Returns
        -------
        DataVariable
            The updated DataVariable

        Examples
        --------
        >>> from pharmpy.model import DataVariable
        >>> var1 = DataVariable.create("WGT", properties={"descriptor": "body weight"})
        >>> var2 = var1.remove_property("body weight")
        """
        props = dict(self._properties)
        props.pop(property, None)
        return self.replace(properties=props)

    def __repr__(self):
        return (
            f"DataVariable(name={self._name}, type={self._type}, scale={self._scale}, "
            f"count={self._count}, properties={self._properties})"
        )


class ColumnInfo(Immutable):
    """Information about one data column

    Parameters
    ----------
    name : str
        Column name
    variable_mapping : Mapping[int, DataVariable]
        A single DataVariable or a Mapping from identifier column to the DataVariable
    variable_id : str
        The DataVariable identifier column
    drop : bool
        Should column be dropped (i.e. barred from being used)
    datatype : str
        Pandas datatype or special Pharmpy datatype (see the "dtype" attribute)
    """

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
        variable_mapping: Union[Mapping[int, DataVariable], DataVariable],
        variable_id: Optional[str] = None,
        drop: bool = False,
        datatype: str = "float64",
    ):
        self._name = name
        self._variable_mapping = variable_mapping
        self._variable_id = variable_id
        self._drop = drop
        self._datatype = datatype

    @classmethod
    def create(
        cls,
        name: str,
        variable_mapping: Optional[Union[Mapping[int, DataVariable], DataVariable]] = None,
        variable_id: Optional[str] = None,
        drop: bool = False,
        datatype: str = "float64",
    ) -> ColumnInfo:
        if variable_mapping is None:
            variable_mapping = DataVariable(name)
        if not isinstance(variable_mapping, DataVariable):
            types = {var.type for var in variable_mapping.values()}
            if len(set(types)) != 1:
                raise ValueError("All data variables need to have the same type in a column")
            if variable_id is None:
                raise ValueError("Need a variable_id when mapping to multiple variables")

        if not isinstance(name, str):
            raise TypeError("Column name must be a string")
        if datatype not in ColumnInfo._all_dtypes:
            raise ValueError(
                f"{datatype} is not a valid datatype. Valid datatypes are {ColumnInfo._all_dtypes}"
            )
        if variable_id is not None and not isinstance(variable_id, str):
            raise TypeError("variable_id must be a string or None")
        if not isinstance(variable_mapping, DataVariable):
            for key, value in variable_mapping.items():
                if not isinstance(key, int) or not isinstance(value, DataVariable):
                    raise TypeError(
                        "The varaible_mapping must be either a single DataVariable"
                        " or a mapping from int to DataVariable"
                    )
            variable_mapping = frozenmapping(variable_mapping)

        return cls(
            name=name,
            datatype=datatype,
            drop=drop,
            variable_id=variable_id,
            variable_mapping=variable_mapping,
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
            and self._drop == other._drop
            and self._datatype == other._datatype
            and self._variable_id == other._variable_id
            and self._variable_mapping == other._variable_mapping
        )

    def __hash__(self):
        return hash(
            (
                self._name,
                self._drop,
                self._datatype,
                self._variable_id,
                self._variable_mapping,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self._variable_mapping, DataVariable):
            mapping = self._variable_mapping.to_dict()
        else:
            mapping = {str(key): value.to_dict() for key, value in self._variable_mapping.items()}
        return {
            'name': self._name,
            'drop': self._drop,
            'datatype': self._datatype,
            'variable_id': self._variable_id,
            'variable_mapping': mapping,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnInfo:
        variable_id = d.get('variable_id', None)
        if variable_id is None:
            mapping = DataVariable.from_dict(d['variable_mapping'])
        else:
            mapping = frozenmapping(
                {
                    int(key): DataVariable.from_dict(value)
                    for key, value in d['variable_mapping'].items()
                }
            )
        return cls.create(
            name=d['name'],
            drop=d.get('drop', False),
            datatype=d.get('datatype', 'float64'),
            variable_id=variable_id,
            variable_mapping=mapping,
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

    @property
    def variable_id(self) -> Optional[str]:
        """Name of identifier column (e.g. DVID or ADMID)"""
        return self._variable_id

    @property
    def variable_mapping(self) -> Union[Mapping[int, DataVariable], DataVariable]:
        """Mapping from value in identifier column to DataVariable"""
        return self._variable_mapping

    @property
    def variable(self) -> DataVariable:
        """If the column represent a single DataVariable return it else raise"""
        if not isinstance(self._variable_mapping, DataVariable):
            raise ValueError("This ColumnInfo represents more than one DataVariable. Use indexing")
        return self._variable_mapping

    @property
    def variables(self) -> tuple[DataVariable, ...]:
        """All datavariables defined in this column"""
        if isinstance(self._variable_mapping, DataVariable):
            return (self._variable_mapping,)
        else:
            return tuple(self._variable_mapping.values())

    @property
    def type(self) -> str:
        """The type of the column. See DataVariable.type
        Note that all variables in one column must have the same type
        """
        if isinstance(self._variable_mapping, DataVariable):
            return self._variable_mapping.type
        else:
            return next(iter(self._variable_mapping.values())).type

    def __len__(self) -> int:
        if isinstance(self._variable_mapping, DataVariable):
            return 1
        else:
            return len(self._variable_mapping)

    def __getitem__(self, index) -> DataVariable:
        if isinstance(self._variable_mapping, DataVariable):
            raise KeyError("This ColumnInfo represents a single DataVariable. Use .variable")
        if isinstance(index, int):
            return self._variable_mapping[index]
        else:
            for var in self._variable_mapping.values():
                if var.name == index:
                    return var
            raise KeyError(f"No DataVariable named {index}")

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
        >>> from pharmpy.model import ColumnInfo, DataVariable
        >>> var = DataVariable.create("WGT", scale='ratio')
        >>> col = ColumnInfo.create("WGT", var)
        >>> col.is_integer()
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

    def __repr__(self):
        variable_names = [var.name for var in self.variables]
        ser = pd.Series(
            [
                self._drop,
                self._datatype,
                self._variable_id,
                ', '.join(variable_names),
            ],
            index=[
                'drop',
                'datatype',
                'variable_id',
                'variables',
            ],
            name=self._name,
        )
        return ser.to_string(name=True)


class DataInfo(Sequence, Immutable):
    """Metadata for the dataset

    Can be indexed to get ColumnInfo for the columns.

    Parameters
    ----------
    columns : tuple
        Tuple of ColumnInfo
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
    ) -> DataInfo:
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
            cols = tuple(
                ColumnInfo.create(col, DataVariable.create(col)) if isinstance(col, str) else col
                for col in columns
            )
        else:
            cols = cast(tuple[ColumnInfo, ...], tuple(columns))
        if path is not None:
            path = Path(path)
        if missing_data_token is None:
            missing_data_token = conf.missing_data_token
        colnames = [col.name for col in cols]
        colnames_set = set(colnames)
        if len(colnames) != len(colnames_set):
            raise ValueError("Column names in a DataInfo need to be unique")
        variable_ids = {col.variable_id for col in cols if col.variable_id is not None}
        missing_ids = variable_ids - colnames_set
        if missing_ids:
            raise ValueError(f"All variable_ids must exist as columns. Missing: {missing_ids}")

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

    def __add__(self, other: Union[DataInfo, ColumnInfo, Sequence[ColumnInfo]]) -> DataInfo:
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

    def _getindex(self, i: Any) -> int:
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
    def __getitem__(self, index: Union[int, str]) -> ColumnInfo: ...

    @overload
    def __getitem__(self, index: Union[Sequence, slice]) -> DataInfo: ...

    def __getitem__(self, index: Union[Sequence, slice, int, str]) -> Union[DataInfo, ColumnInfo]:
        if isinstance(index, (int, str)):
            return self._columns[self._getindex(index)]
        elif isinstance(index, Sequence):
            cols = []
            for ind in index:
                i = self._getindex(ind)
                cols.append(self._columns[i])
            return DataInfo.create(columns=cols)
        elif isinstance(index, slice):
            return DataInfo.create(self._columns[index], path=self._path, separator=self._separator)
        else:
            # NOTE: To trigger the exception
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

        var_mapping = mycol.variable_mapping
        if isinstance(var_mapping, DataVariable):
            new_mapping = var_mapping.replace(type=type)
        else:
            new_mapping = dict(var_mapping)
            for key, value in new_mapping.items():
                new_variable = value.replace(type=type)
                new_mapping[key] = new_variable
        newcol = mycol.replace(variable_mapping=new_mapping)
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
    def symbols(self) -> list[Expr]:
        """Symbols for all columns

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.symbols
        [ID, TIME, AMT, WGT, APGR, DV, FA1, FA2]
        """
        return [col.symbol for col in self._columns]

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
            if isinstance(col._variable_mapping, DataVariable):
                newvar = col.variable.replace(type=v)
                newcol = col.replace(variable_mapping=newvar)
            else:
                new_mapping = {}
                for key, var in col._variable_mapping.items():
                    newvar = var.replace(type=v)
                    new_mapping[key] = newvar
                newcol = col.replace(variable_mapping=new_mapping)
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
        return cls.create(
            columns=columns,
            path=None if d['path'] is None else Path(d['path']),
            separator=d['separator'],
            missing_data_token=d['missing_data_token'],
        )

    def _to_dict(self, path: Optional[str]) -> dict[str, Any]:
        columns = [col.to_dict() for col in self._columns]

        return {
            "columns": columns,
            "path": None if path is None else str(path),
            "separator": self._separator,
            "missing_data_token": self._missing_data_token,
        }

    def to_json(self, path: Optional[Union[Path, str]] = None):
        if path is None:
            d = self._to_dict(str(self.path) if self.path is not None else None)
        else:
            d = self._to_dict(
                str(path_relative_to(Path(path).parent, self.path))
                if self.path is not None
                else None
            )
        d['__version__'] = 1
        if path is None:
            return json.dumps(d)
        else:
            with open(path, 'w') as fp:
                json.dump(d, fp)

    @staticmethod
    def _populate_dict_with_defaults(d: dict[str, Any]):
        def _defaults_in_data_variable(variable):
            if 'type' not in variable:
                variable['type'] = 'unknown'
            if 'count' not in variable:
                variable['count'] = False
            if 'properties' not in variable:
                variable['properties'] = {}
            if 'scale' not in variable:
                variable['scale'] = 'ratio'

        if 'path' not in d:
            d['path'] = None
        if 'missing_data_token' not in d:
            d['missing_data_token'] = None

        for col in d['columns']:
            if 'variable_id' not in col:
                col['variable_id'] = None
            if 'drop' not in col:
                col['drop'] = False
            if 'datatype' not in col:
                col['datatype'] = 'float64'
            variable_mapping = col['variable_mapping']
            first_key = next(iter(variable_mapping.keys()))
            try:
                int(first_key)
            except ValueError:
                is_single = True
            else:
                is_single = False
            if is_single:
                _defaults_in_data_variable(variable_mapping)
            else:
                for variable in variable_mapping.values():
                    _defaults_in_data_variable(variable)

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
        del d['__version__']
        DataInfo._populate_dict_with_defaults(d)
        di = DataInfo.from_dict(d)
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

    @property
    def variables(self) -> list[DataVariable]:
        """A list of all data variables in order"""
        variables = []
        for col in self._columns:
            if isinstance(col._variable_mapping, DataVariable):
                variables.append(col.variable)
            else:
                variables += list(col._variable_mapping.values())
        return variables

    def __repr__(self):
        colnames = []
        drop = []
        datatype = []
        for col in self._columns:
            colnames += [col.name] * len(col)
            drop += [col.drop] * len(col)
            datatype += [col.datatype] * len(col)

        variables = self.variables
        varnames = [var.name for var in variables]
        types = [var.type for var in variables]
        scales = [var.scale for var in variables]
        count = [var.count for var in variables]
        properties = [var.properties for var in variables]
        df = pd.DataFrame(
            {
                'name': colnames,
                'variable': varnames,
                'type': types,
                'scale': scales,
                'count': count,
                'drop': drop,
                'datatype': datatype,
                'properties': properties,
            }
        )
        return df.to_string(index=False)

    def find_column_by_property(self, property: str, value: Any) -> Optional[ColumnInfo]:
        """Find a single  column having a property/value pair

        Returns None if more than one column have the pair, if no column
        has the pair or if not all variables of a column have the pair.

        """
        found = None
        for col in self:
            for var in col.variables:
                if var.properties.get(property, None) != value:
                    break
            else:
                if found is None:
                    found = col
                else:
                    return None
        return found


class TypeIndexer:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, i) -> DataInfo:
        cols = [col for col in self._obj if col.type == i and not col.drop]
        if not cols:
            raise IndexError(f"No columns of type {i} available")
        return DataInfo.create(cols)
