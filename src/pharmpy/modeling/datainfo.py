from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union, overload

from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.path import normalize_user_given_path, path_absolute
from pharmpy.model import ColumnInfo, DataInfo, DataVariable, Model


def _read_dataset_header_and_separator(path) -> tuple[list[str], str]:
    with open(path) as file:
        first_line = file.readline()
        if ',' not in first_line:
            colnames = list(pd.read_csv(path, nrows=0, sep=r'\s+'))
            separator = r'\s+'
        else:
            colnames = list(pd.read_csv(path, nrows=0))
            separator = ','
    if len(colnames) > 0:
        colnames[0] = colnames[0].lstrip('#')
    return colnames, separator


def create_datainfo(path_or_df: Union[str, Path, pd.DataFrame]) -> DataInfo:
    """Create a DataInfo

    Will assume NONMEM names of columns

    Parameters
    ----------
    path_or_df : Path | str | pd.DataFrame
        A path to a dataset or a dataset

    Returns
    -------
    DataInfo
        DataInfo object
    """

    if not isinstance(path_or_df, pd.DataFrame):
        path = normalize_user_given_path(path_or_df)
        path = path_absolute(path)
        datainfo_path = path.with_suffix('.datainfo')
        try:
            di = read_datainfo(datainfo_path)
        except FileNotFoundError:
            pass
        else:
            di = di.replace(path=path)
            return di

        colnames, separator = _read_dataset_header_and_separator(path)

    else:
        colnames = path_or_df.columns
        separator = ","
        path = None

    column_info = []
    for colname in colnames:
        colname = colname.replace('.', '_')  # pandas uses . to name mangle
        if colname == 'ID' or colname == 'L1':
            var = DataVariable.create(colname, type='id', scale='nominal')
            info = ColumnInfo.create(colname, var, datatype='int32')
        elif colname == 'DV':
            var = DataVariable.create(colname, type='dv')
            info = ColumnInfo.create(colname, var)
        elif colname == 'TIME':
            if not set(colnames).isdisjoint({'DATE', 'DAT1', 'DAT2', 'DAT3'}):
                datatype = 'nmtran-time'
            else:
                datatype = 'float64'
            var = DataVariable.create(colname, type='idv', scale='ratio')
            info = ColumnInfo.create(colname, var, datatype=datatype)
        elif colname == 'EVID':
            var = DataVariable.create(colname, type='event', scale='nominal')
            info = ColumnInfo.create(colname, var)
        elif colname == 'MDV':
            if 'EVID' in colnames:
                var = DataVariable.create(colname, type='mdv')
                info = ColumnInfo.create(colname, var)
            else:
                var = DataVariable.create(colname, type='event', scale='nominal')
                info = ColumnInfo.create(colname, var, datatype='int32')
        elif colname == 'AMT':
            var = DataVariable.create(colname, type='dose', scale='ratio')
            info = ColumnInfo.create(colname, var)
        elif colname == 'RATE':
            var = DataVariable.create(colname, type='rate', scale='ratio')
            info = ColumnInfo.create(colname, var)
        elif colname == 'BLQ':
            var = DataVariable.create(colname, type='blq', scale='nominal')
            info = ColumnInfo.create(colname, var, datatype='int32')
        elif colname == 'LLOQ':
            var = DataVariable.create(colname, type='lloq', scale='ratio')
            info = ColumnInfo.create(colname, var)
        elif colname == 'DVID':
            var = DataVariable.create(colname, type='dvid', scale='nominal')
            info = ColumnInfo.create(colname, var, datatype='int32')
        elif colname == 'SS':
            var = DataVariable.create(colname, type='ss', scale='nominal')
            info = ColumnInfo.create(colname, var, datatype='int32')
        elif colname == 'II':
            var = DataVariable.create(colname, type='ii', scale='ratio')
            info = ColumnInfo.create(colname, var)
        else:
            info = ColumnInfo.create(colname)
        column_info.append(info)
    di = DataInfo.create(column_info, path=path, separator=separator)
    return di


def read_datainfo(path: Union[str, Path]) -> DataInfo:
    """Read a datainfo file

    Parameters
    ----------
    path : Path | str
        A path to a datainfo file

    Returns
    -------
    DataInfo
        DataInfo object
    """

    path = normalize_user_given_path(path)
    path = path_absolute(path)
    if path.is_file():
        di = DataInfo.read_json(path)
    else:
        raise FileNotFoundError("Could not find path to datainfo file")
    return di


def write_datainfo(di: DataInfo, path: Union[str, Path], force: bool = False) -> None:
    """Write a DataInfo object to a datainfo file

    Parameters
    ----------
    di : DataInfo
        DataInfo object
    path : Path | str
        Path to write the datainfo file
    force : bool
        Force overwrite if file already exists
    """

    path = normalize_user_given_path(path)
    if path.is_file() and not force:
        raise FileExistsError(
            f"A datainfo file already exists at {path}. " "Set force=True to overwrite"
        )
    di.to_json(path)


@overload
def annotate_unit(model_or_datainfo: Model, column: str, unit: str) -> Model: ...


@overload
def annotate_unit(model_or_datainfo: DataInfo, column: str, unit: str) -> DataInfo: ...


def annotate_unit(
    model_or_datainfo: Union[Model, DataInfo], column: str, unit: str
) -> Union[Model, DataInfo]:
    """Specify the unit of a data column

    Note that no conversion of units will happen if the unit was already set.

    Parameters
    ----------
    model_or_datainfo : Model | DataInfo
        Model object or DataInfo object
    column : str
        Name of a column. If the column contains multiple variables, e.g. DV
        with multiple DVs, the ID can be specified with a colon. For example "DV:1"
        will mean the DV column only when DVID is 1.
    unit : str
        The unit

    Returns
    -------
    Model | DataInfo
        An updated Model or DataInfo object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, annotate_unit
    >>> model = load_example_model("pheno")
    >>> model = annotate_unit(model, "WGT", "kg")

    See Also
    --------

    convert_unit - Convert between units for a variable

    """

    return set_property(model_or_datainfo, column, "unit", unit)


@overload
def set_property(model_or_datainfo: Model, column: str, property: str, value: Any) -> Model: ...


@overload
def set_property(
    model_or_datainfo: DataInfo, column: str, property: str, value: Any
) -> DataInfo: ...


def set_property(
    model_or_datainfo: Union[Model, DataInfo], column: str, property: str, value: Any
) -> Union[Model, DataInfo]:
    """Specify a property of a column

    See :py:attr:`pharmpy.DataInfo.properties` for documentation on data properties.

    Parameters
    ----------
    model_or_datainfo : Model | DataInfo
        Model object or DataInfo object
    column : str
        Name of a column. If the column contains multiple variables, e.g. DV
        with multiple DVs, the ID can be specified with a colon. For example "DV:1"
        will mean the DV column only when DVID is 1.
    property : str
        Name of the property to set
    value : Any
        Value of the property to set

    Returns
    -------
    Model | DataInfo
        An updated Model or DataInfo object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, set_property
    >>> model = load_example_model("pheno")
    >>> model = set_property(model, "APGR", "categories", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    See Also
    --------

    annotate_unit - Annotate the unit of a data variable

    """

    if isinstance(model_or_datainfo, Model):
        di = model_or_datainfo.datainfo
    else:
        di = model_or_datainfo

    a = column.split(":")
    name = a[0]
    if len(a) == 2:
        n = int(a[1])
    else:
        n = None

    col = di[name]
    if n is not None:
        var = col[n]
        new_var = var.set_property(property, value)
        old_mapping = col.variable_mapping
        assert isinstance(old_mapping, Mapping)
        new_mapping = old_mapping.replace(n, new_var)
    elif not isinstance(col.variable_mapping, DataVariable):
        new_mapping = {}
        for key, var in col.variable_mapping.items():
            new_var = var.set_property(property, value)
            new_mapping[key] = new_var
    else:
        var = col.variable
        new_mapping = var.set_property(property, value)

    new_col = col.replace(variable_mapping=new_mapping)
    new_di = di.set_column(new_col)

    if isinstance(model_or_datainfo, Model):
        new_model = model_or_datainfo.replace(datainfo=new_di)
        return new_model
    else:
        return new_di
