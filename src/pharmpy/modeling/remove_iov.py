"""
:meta private:
"""

import warnings
from collections import defaultdict

from pharmpy.deps import sympy
from pharmpy.model import Model
from pharmpy.modeling import remove_unused_parameters_and_rvs
from pharmpy.modeling.help_functions import _format_input_list


def remove_iov(model, to_remove=None):
    """Removes all IOV etas given a list with eta names.

    Parameters
    ----------
    model : Model
        Pharmpy model to remove IOV from.
    to_remove : str, list
        Name/names of IOV etas to remove, e.g. 'ETA_IOV_1_1'.
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

    keep = [name for name in rvs.names if name not in etas]
    model.random_variables = rvs[keep]

    d = {sympy.Symbol(name): 0 for name in etas}
    model.statements = sset.subs(d)

    remove_unused_parameters_and_rvs(model)
    model.update_source()
    return model


def _get_iov_etas(model: Model, list_of_etas):
    list_of_etas = _format_input_list(list_of_etas)
    rvs = model.random_variables
    if list_of_etas is None:
        return set(rvs.iov.names)

    # NOTE Include all directly referenced ETAs
    direct_etas = set(list_of_etas)

    # NOTE Include all IOV ETAs that are identically distributed to the ones
    # directly referenced
    indirect_etas = set()
    for group in _get_iov_groups(model):
        if not direct_etas.isdisjoint(group):
            indirect_etas.update(group)

    return direct_etas | indirect_etas


def _get_iov_groups(model: Model):
    iovs = model.random_variables.iov
    same = defaultdict(set)
    for dist in iovs:
        for i, name in enumerate(dist.names):
            key = (i, dist.variance)
            same[key].add(name)

    return same.values()
