"""
:meta private:
"""

import warnings

from pharmpy.modeling import remove_unused_parameters_and_rvs
from pharmpy.modeling.help_functions import _format_input_list
from pharmpy.random_variables import RandomVariables


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
    etas = _get_etas(rvs, to_remove)
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


def _get_etas(rvs, list_of_etas):
    list_of_etas = _format_input_list(list_of_etas)
    etas_all = [eta for eta in rvs if eta.level == 'IOV']
    etas = []
    if list_of_etas is None:
        etas = etas_all
    else:
        for eta_str in list_of_etas:
            eta_remove = rvs[eta_str.upper()]
            if eta_remove not in etas:
                etas.append(eta_remove)
                for eta in etas_all:
                    if eta.parameter_names[0] == eta_remove.parameter_names[0] and eta not in etas:
                        etas.append(eta)

    return RandomVariables(etas)
