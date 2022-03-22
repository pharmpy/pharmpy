"""
:meta private:
"""
from pharmpy.modeling import remove_unused_parameters_and_rvs
from pharmpy.modeling.help_functions import _format_input_list, _get_etas


def remove_iiv(model, to_remove=None):
    """
    Removes all IIV etas given a list with eta names and/or parameter names.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    to_remove : str, list
        Name/names of etas and/or name/names of individual parameters to remove.
        If None, all etas that are IIVs will be removed. None is default.

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_iiv(model)  # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("CL")
    CL := TVCL

    >>> model = load_example_model("pheno")
    >>> remove_iiv(model, "V")  # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("V")
    V := TVV

    See also
    --------
    remove_iov
    add_iiv
    add_iov
    add_pk_iiv

    """
    rvs, sset = model.random_variables, model.statements
    to_remove = _format_input_list(to_remove)
    etas = _get_etas(model, to_remove, include_symbols=True)

    for eta in etas:
        sset.subs({eta.symbol: 0})
        del rvs[eta]

    model.random_variables = rvs
    model.statements = sset

    model.modelfit_results = None
    remove_unused_parameters_and_rvs(model)
    model.update_source()
    return model
