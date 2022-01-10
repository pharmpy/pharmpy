import sympy

from pharmpy.statements import Assignment, ODESystem


def _extract_minus(expr):
    if expr.is_Mul and expr.args[0] == -1:
        return sympy.Mul(*expr.args[1:])
    else:
        return expr


def get_unit_of(model, variable):
    """Derive the physical unit of a variable in the model

    Unit information for the dataset needs to be available.
    The variable can be defined in the code, a dataset olumn, a parameter
    or a random variable.

    Parameters
    ----------
    model : Model
        Pharmpy model object
    variable : str or Symbol
        Find physical unit of this variable

    Returns
    -------
    unit expression
        A sympy physics.units expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_unit_of
    >>> model = load_example_model("pheno")
    >>> get_unit_of(model, "Y")
    milligram/liter
    >>> get_unit_of(model, "V")
    liter
    >>> get_unit_of(model, "WGT")
    kilogram
    """
    if isinstance(variable, str):
        symbol = sympy.Symbol(variable)
    else:
        symbol = variable
        variable = variable.name

    di = model.datainfo
    if variable in di.names:
        return di[variable].unit

    y = model.dependent_variable
    input_units = {sympy.Symbol(col.name): col.unit for col in di}
    input_units[sympy.exp] = 1
    unit_eqs = []
    unit_eqs.append(y - di[di.dv_column.name].unit)

    for s in model.statements:
        if isinstance(s, Assignment):
            expr = sympy.expand(s.expression.subs(input_units))
            if expr.is_Add:
                for term in expr.args:
                    unit_eqs.append(s.symbol - _extract_minus(term))
            else:
                unit_eqs.append(s.symbol - _extract_minus(expr))
        elif isinstance(s, ODESystem):
            amt_unit = di[di.typeix['dose'][0].name].unit
            time_unit = di[di.idv_column.name].unit
            for e in s.compartmental_matrix.diagonal():
                if e.is_Add:
                    for term in e.args:
                        unit_eqs.append(amt_unit / time_unit - _extract_minus(term))
                elif e == 0:
                    pass
                else:
                    unit_eqs.append(amt_unit / time_unit - _extract_minus(e))
            for a in s.amounts:
                unit_eqs.append(amt_unit - a)
    sol = sympy.solve(unit_eqs)
    return sol[symbol]
