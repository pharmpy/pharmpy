"""
:meta private:
"""

import warnings

from pharmpy.random_variables import RandomVariables, VariabilityLevel
from pharmpy.symbols import symbol as S


def remove_iov(model):
    """
    Removes all IOV omegas.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    """
    rvs, sset = model.random_variables, model.statements
    etas = _get_etas(rvs)

    if not etas:
        warnings.warn('No IOVs present')
        return model

    for eta in etas:
        statement = sset.find_assignment(eta.name, is_symbol=False)
        statement.expression = statement.expression.subs(S(eta.name), 0)
        rvs.discard(eta)

    model.random_variables = rvs
    model.statements = sset

    model.modelfit_results = None

    return model


def _get_etas(rvs):
    etas = [eta for eta in rvs if eta.variability_level == VariabilityLevel.IOV]

    return RandomVariables(etas)
