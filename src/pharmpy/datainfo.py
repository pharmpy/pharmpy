"""DataInfo is a companion to the dataset. It contains metadata of the dataset
"""
import copy
import json
from collections.abc import MutableSequence
from pathlib import Path

import pandas as pd
import sympy

from pharmpy.utils import parse_units


class ColumnInfo:
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
        'additional',
        'ii',
        'ss',
        'event',
        'covariate',
        'mdv',
        'nmtran_date',
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
    ]
    _all_descriptors = [None, 'age', 'body weight', 'lean body mass', 'fat free mass']

    def __init__(
        self,
        name,
        type='unknown',
        unit=sympy.Integer(1),
        scale='ratio',
        continuous=True,
        categories=None,
        drop=False,
        datatype="float64",
        descriptor=None,
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
        self.datatype = datatype
        self.descriptor = descriptor

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.type == other.type
            and self.unit == other.unit
            and self.scale == other.scale
            and self.continuous == other.continuous
            and self.categories == other.categories
            and self.drop == other.drop
            and self.datatype == other.datatype
        )

    @property
    def name(self):
        """Column name"""
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Column name must be a string")
        self._name = value

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
        additional    Number of additional doses
        ii            Interdose interval
        ss            Steady state dosing
        event         0 = observation
        mdv           0 = DV is observation value, 1 = DV is missing
        unknown       Unkown type. This will be the default for columns that hasn't been
                      assigned a type
        ============  =============
        """
        return self._type

    @type.setter
    def type(self, value):
        if value not in ColumnInfo._all_types:
            raise TypeError(f"Unknown column type {value}")
        self._type = value

    @property
    def descriptor(self):
        """Kind of data

        ================ =================
        descriptor       Description
        ================ =================
        age              Age (since birth)
        body weight      Human body weight
        lean body mass   Lean body mass
        fat free mass    Fat free mass
        ================ =================
        """
        return self._descriptor

    @descriptor.setter
    def descriptor(self, value):
        if value not in ColumnInfo._all_descriptors:
            raise TypeError(f"Unknown column descriptor {value}")
        self._descriptor = value

    @property
    def unit(self):
        """Unit of the column data

        Custom units are allowed, but units that are available in sympy.physics.units can be
        recognized. The default unit is 1, i.e. without unit.
        """
        return self._unit

    @unit.setter
    def unit(self, value):
        a = parse_units(value)
        self._unit = a

    @property
    def scale(self):
        """Scale of measurement

        The statistical scale of measurement for the column data. Can be one of
        'nominal', 'ordinal', 'interval' and 'rational'.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        if value not in ColumnInfo._all_scales:
            raise TypeError(
                f"Unknown scale of measurement {value}. Only {ColumnInfo._all_scales} are possible."
            )
        self._scale = value
        if self.continuous and value in ['nominal', 'ordinal']:
            self.continuous = False

    @property
    def continuous(self):
        """Is the column data continuous

        True for continuous data and False for discrete. Note that nominal and ordinal data have to
        be discrete.
        """
        return self._continuous

    @continuous.setter
    def continuous(self, value):
        if value and self.is_categorical():
            raise ValueError(
                f"Cannot set variable on {self.scale} scale of measurement to continuous"
            )
        self._continuous = value

    @property
    def categories(self):
        """List of allowed categories"""
        return self._categories

    @categories.setter
    def categories(self, value):
        self._categories = value

    @property
    def drop(self):
        """Should this column be dropped"""
        return self._drop

    @drop.setter
    def drop(self, value):
        self._drop = bool(value)

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
        ============ ================ ========================================== ===========

        The default, and most common datatype, is float64.
        """
        return self._datatype

    @datatype.setter
    def datatype(self, value):
        if value not in ColumnInfo._all_dtypes:
            raise ValueError(
                f"{value} is not a valid datatype. Valid datatypes are {ColumnInfo._all_dtypes}"
            )
        self._datatype = value

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
        >>> from pharmpy.datainfo import ColumnInfo
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
        >>> from pharmpy.datainfo import ColumnInfo
        >>> col1 = ColumnInfo("WGT", scale='ratio')
        >>> col1.is_numerical()
        True
        >>> col2 = ColumnInfo("ID", scale='nominal')
        >>> col2.is_numerical()
        False

        """
        return self.scale in ['interval', 'ratio']

    def copy(self):
        """Create a deep copy of the ColumnInfo

        Returns
        -------
        ColumnInfo
            Copied object
        """
        return copy.deepcopy(self)

    def __repr__(self):
        di = DataInfo([self])
        return repr(di)


class DataInfo(MutableSequence):
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

    def __init__(self, columns=None, path=None, separator=','):
        if columns is None:
            self._columns = []
        elif len(columns) > 0 and isinstance(columns[0], str):
            self._columns = []
            for name in columns:
                colinf = ColumnInfo(name)
                self._columns.append(colinf)
        else:
            self._columns = columns
        self.path = path
        self.separator = separator

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for col1, col2 in zip(self, other):
            if col1 != col2:
                return False
        return self.path == other.path

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

    def __getitem__(self, i):
        if isinstance(i, list):
            cols = []
            for ind in i:
                index = self._getindex(ind)
                cols.append(self._columns[index])
            return DataInfo(columns=cols)
        return self._columns[self._getindex(i)]

    def __setitem__(self, i, value):
        self._columns[self._getindex(i)] = value

    def __delitem__(self, i):
        del self._columns[self._getindex(i)]

    def insert(self, i, value):
        self._columns.insert(self._getindex(i), value)

    @property
    def path(self):
        """Path of dataset file

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.path     # doctest: +ELLIPSIS
        ...pharmpy/modeling/example_models/pheno.dta')
        """
        return self._path

    @path.setter
    def path(self, value):
        if value is not None:
            self._path = Path(value)
        else:
            self._path = None

    @property
    def separator(self):
        """Separator for dataset file

        Can be a single character or a regular expression
        string.
        """
        return self._separator

    @separator.setter
    def separator(self, value):
        self._separator = value

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

    @id_column.setter
    def id_column(self, value):
        self[value].type = 'id'

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

    @dv_column.setter
    def dv_column(self, value):
        self[value].type = 'dv'

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

    @idv_column.setter
    def idv_column(self, value):
        self[value].type = 'idv'

    @property
    def names(self):
        """All column names

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.datainfo.names
        ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV']
        """
        return [col.name for col in self._columns]

    @property
    def types(self):
        """All column types"""
        return [col.type for col in self._columns]

    @types.setter
    def types(self, value):
        if isinstance(value, str):
            value = [value]
        if len(value) == 1:
            value *= len(self)
        if len(value) != len(self):
            raise ValueError(
                "Length mismatch. "
                "Can only set the same number of names as columns or 1 for broadcasting"
            )
        for v, col in zip(value, self._columns):
            col.type = v

    def to_json(self, path=None):
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
            }
            if col.descriptor is not None:
                d["descriptor"] = col.descriptor
            a.append(d)
        s = json.dumps({"columns": a, "path": str(self.path) if self.path is not None else None})
        if path is None:
            return s
        else:
            with open(path, 'w') as fp:
                fp.write(s)

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
                type=col['type'],
                scale=col['scale'],
                continuous=col.get('continuous', True),
                unit=col.get('unit', sympy.Integer(1)),
                categories=col.get('categories', None),
                datatype=col.get('datatype', 'float64'),
                descriptor=col.get('descriptor', None),
            )
            columns.append(ci)
        di = DataInfo(columns)
        path = d.get('path', None)
        if path:
            path = Path(path)
        di.path = path
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
        return DataInfo.from_json(s)

    def copy(self):
        """Create a deep copy of the datainfo

        Returns
        -------
        DataInfo
            Copied object
        """
        return copy.deepcopy(self)

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
        df = pd.DataFrame(columns=labels)
        df.loc['type'] = types
        df.loc['scale'] = scales
        df.loc['continuous'] = cont
        df.loc['categories'] = cats
        df.loc['unit'] = units
        df.loc['drop'] = drop
        df.loc['datatype'] = datatype
        df.loc['descriptor'] = descriptor
        return df.to_string()


class TypeIndexer:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, i):
        cols = [col for col in self._obj if col.type == i and not col.drop]
        if not cols:
            raise IndexError(f"No columns of type {i} available")
        return DataInfo(cols)
