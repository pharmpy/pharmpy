from typing import Any, Union, overload

from pharmpy.model import DataInfo, DataVariable, Model


@overload
def set_unit(model_or_datainfo: Model, column: str, unit: str) -> Model: ...


@overload
def set_unit(model_or_datainfo: DataInfo, column: str, unit: str) -> DataInfo: ...


def set_unit(
    model_or_datainfo: Union[Model, DataInfo], column: str, unit: str
) -> Union[Model, DataInfo]:
    """Specify the unit of a column

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
    >>> from pharmpy.modeling import load_example_model, set_unit
    >>> model = load_example_model("pheno")
    >>> model = set_unit(model, "WGT", "kg")

    See also
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

    See also
    --------

    set_unit - Set unit of a data variable

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
        new_mapping = col.variable_mapping.replace(n, new_var)
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
