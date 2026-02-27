from typing import Union, overload

from pharmpy.model import DataInfo, Model


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
        new_var = var.set_property("unit", unit)
        new_mapping = col.variable_mapping.replace(n, new_var)
    elif len(col) > 1:
        new_mapping = {}
        for key, var in col.variable_mapping:
            new_var = var.set_property("unit", unit)
            new_mapping[key] = new_var
    else:
        var = col.variable
        new_mapping = var.set_property("unit", unit)

    new_col = col.replace(variable_mapping=new_mapping)
    new_di = di.set_column(new_col)

    if isinstance(model_or_datainfo, Model):
        new_model = model_or_datainfo.replace(datainfo=new_di)
        return new_model
    else:
        return new_di
