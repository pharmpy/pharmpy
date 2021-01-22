"""
:meta private:
"""

from pharmpy.random_variables import RandomVariables, VariabilityLevel
from pharmpy.statements import Assignment
from pharmpy.symbols import symbol as S


def remove_iiv(model, list_to_remove=None):
    """
    Removes all IIV omegas given a list with eta names and/or parameter names.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_to_remove : list
        List of etas and/or parameter name to remove. If None, all etas that are IIVs will
        be removed. None is default.
    """
    rvs, sset = model.random_variables, model.statements
    etas = _get_etas(model, list_to_remove)

    for eta in etas:
        statement = sset.find_assignment(eta.name, is_symbol=False)
        statement.expression = statement.expression.subs(S(eta.name), 0)
        rvs.discard(eta)

    model.random_variables = rvs
    model.statements = sset

    model.modelfit_results = None

    return model


def _get_etas(model, list_to_remove):
    rvs = model.random_variables
    sset = model.statements
    symbols_all = [s.symbol.name for s in model.statements if isinstance(s, Assignment)]

    if list_to_remove is None:
        list_to_remove = [rv for rv in rvs.etas]

    etas = []
    symbols = []
    for variable in list_to_remove:
        try:
            eta = rvs[variable]
        except KeyError:
            if variable in symbols_all:
                symbols.append(variable)
                continue
            else:
                raise KeyError(f'Random variable does not exist: {variable}')

        if eta.variability_level == VariabilityLevel.IOV:
            if list_to_remove:
                raise ValueError(f'Random variable cannot be IOV: {variable}')
            continue

        etas.append(eta)

    for symbol in symbols:
        terms = sset.find_assignment(symbol).free_symbols
        eta_set = terms.intersection(rvs.free_symbols)
        etas += [rvs[eta.name] for eta in eta_set]

    return RandomVariables(etas)
