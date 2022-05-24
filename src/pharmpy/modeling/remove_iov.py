"""
:meta private:
"""

import warnings

from sympy.functions.elementary.piecewise import Piecewise

from pharmpy.model import Model
from pharmpy.modeling import remove_unused_parameters_and_rvs
from pharmpy.modeling.help_functions import _format_input_list
from pharmpy.random_variables import RandomVariable
from pharmpy.statements import Assignment


def remove_iov(model, to_remove=None):
    """Removes all IOV etas given a list with eta names.

    Parameters
    ----------
    model : Model
        Pharmpy model to remove IOV from.
    to_remove : str, list
        Name/names of IOV etas to remove, e.g. 'ETA_IOV_11'.
        If None, all etas that are IOVs will be removed. None is default.
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
    add_pk_iiv

    """
    rvs, sset = model.random_variables, model.statements
    etas = _get_iov_etas(model, to_remove)
    if not etas:
        warnings.warn('No IOVs present')
        return model

    for eta in etas:
        sset.subs({eta.symbol: 0})
        del rvs[eta]

    model.random_variables = rvs
    model.statements = sset

    model.modelfit_results = None
    remove_unused_parameters_and_rvs(model)
    model.update_source()
    return model


def _get_iov_etas(model: Model, list_of_etas):
    list_of_etas = _format_input_list(list_of_etas)
    rvs = model.random_variables
    if list_of_etas is None:
        return set(rvs.iov)

    # NOTE Include all directly referenced ETAs
    direct_etas = set()
    for eta_str in list_of_etas:
        eta = rvs[eta_str.upper()]
        assert isinstance(eta, RandomVariable)
        direct_etas.add(eta)

    # NOTE Include all ETAs that assign to the same IOV variables
    # as the one directly referenced
    indirect_etas = set()
    for expression_rvs in _get_iov_piecewise_assignments_rvs(model):
        if not direct_etas.isdisjoint(expression_rvs):
            indirect_etas.update(expression_rvs)

    etas = direct_etas | indirect_etas

    # NOTE Check that we got the closure
    for expression_rvs in _get_iov_piecewise_assignments_rvs(model):
        if not etas.isdisjoint(expression_rvs):
            assert etas.issuperset(expression_rvs)

    return etas


def _get_iov_piecewise_assignments_rvs(model: Model):
    iovs = set(rv.symbol for rv in model.random_variables.iov)
    for statement in model.statements:
        if isinstance(statement, Assignment) and isinstance(statement.expression, Piecewise):
            expression_symbols = [p[0] for p in statement.expression.as_expr_set_pairs()]
            if all(s in iovs for s in expression_symbols):
                yield model.random_variables[expression_symbols]
