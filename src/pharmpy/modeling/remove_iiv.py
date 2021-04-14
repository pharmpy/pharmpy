"""
:meta private:
"""

from pharmpy.modeling.help_functions import _format_input_list, _get_etas
from pharmpy.symbols import symbol as S


def remove_iiv(model, list_to_remove=None):
    """
    Removes all IIV omegas given a list with eta names and/or parameter names.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_to_remove : str, list
        Name/names of etas and/or name/names of individual parameters to remove.
        If None, all etas that are IIVs will be removed. None is default.
    """
    rvs, sset = model.random_variables, model.statements
    list_to_remove = _format_input_list(list_to_remove)
    etas = _get_etas(model, list_to_remove, include_symbols=True)

    for eta in etas:
        statement = sset.find_assignment(eta.name, is_symbol=False)
        statement.expression = statement.expression.subs(S(eta.name), 0)
        del rvs[eta]

    model.random_variables = rvs
    model.statements = sset

    model.modelfit_results = None

    return model
