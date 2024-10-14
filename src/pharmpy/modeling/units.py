from __future__ import annotations

from typing import TypeVar, Union

from pharmpy.basic import Unit
from pharmpy.deps import sympy
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.expr.tree import prune
from pharmpy.model import Assignment, CompartmentalSystem, Model

T = TypeVar('T')


def _extract_minus(expr):
    if expr.is_Mul and expr.args[0] == -1:
        return sympy.Mul(*expr.args[1:])
    else:
        return expr


def get_unit_of(model: Model, variable: Union[str, sympy.Symbol]) -> Unit:
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
    Unit
        A unit expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_unit_of
    >>> model = load_example_model("pheno")
    >>> get_unit_of(model, "Y")
    milligram/liter
    >>> get_unit_of(model, "VC")
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

    # FIXME: Handle other DVs?
    y = sympy.sympify(list(model.dependent_variables.keys())[0])
    input_units = {sympy.Symbol(col.name): col.unit._expr for col in di}
    pruned_nodes = {sympy.exp}

    def pruning_predicate(e: sympy.Basic) -> bool:
        return e.func in pruned_nodes

    unit_eqs = []
    # FIXME: Using private _expr in some places. sympify doesn't work for some reason.
    a = di[di.dv_column.name].unit._expr
    unit_eqs.append(y - a)
    d = {}

    for s in model.statements:
        if isinstance(s, Assignment):
            expr = sympy.expand(
                subs(
                    prune(pruning_predicate, sympy.sympify(s.expression)),
                    input_units,
                    simultaneous=True,
                )
            )
            if expr.is_Add:
                for term in expr.args:
                    unit_eqs.append(sympy.sympify(s.symbol) - _extract_minus(term))
            else:
                unit_eqs.append(sympy.sympify(s.symbol) - _extract_minus(expr))
        elif isinstance(s, CompartmentalSystem):
            amt_unit = di[di.typeix['dose'][0].name].unit._expr
            time_unit = di[di.idv_column.name].unit._expr
            for e in s.compartmental_matrix.diagonal():
                e = sympy.sympify(e)
                if e.is_Add:
                    for term in e.args:
                        unit_eqs.append(amt_unit / time_unit - _extract_minus(term))
                elif e == 0:
                    pass
                else:
                    unit_eqs.append(amt_unit / time_unit - _extract_minus(e))
            for a in s.amounts:
                sy = sympy.Symbol(a.name)
                d[a] = sy
                unit_eqs.append(amt_unit - sy)

    filtered_unit_eqs = [eq.subs(d) for eq in unit_eqs]
    # NOTE: For some reason telling sympy to solve for "symbol" does not work
    sol = sympy.solve(filtered_unit_eqs, dict=True)
    return Unit(sol[0][symbol])
