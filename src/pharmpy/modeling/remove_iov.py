"""
:meta private:
"""

import warnings

from pharmpy.random_variables import RandomVariables


def remove_iov(model):
    """Removes all IOV etas

    Parameters
    ----------
    model : Model
        Pharmpy model to remove IOV from.

    Return
    ------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_iov(model)       # doctest: +ELLIPSIS
    <...>

    See also
    --------
    add_iiv
    add_iov
    remove_iiv

    """
    rvs, sset = model.random_variables, model.statements
    etas = _get_etas(rvs)

    if not etas:
        warnings.warn('No IOVs present')
        return model

    for eta in etas:
        sset.subs({eta.symbol: 0})
        del rvs[eta]

    model.random_variables = rvs
    model.statements = sset

    model.modelfit_results = None

    return model


def _get_etas(rvs):
    etas = [eta for eta in rvs if eta.level == 'IOV']

    return RandomVariables(etas)
