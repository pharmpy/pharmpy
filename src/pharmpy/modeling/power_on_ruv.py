"""
:meta private:
"""

import sympy

from pharmpy.modeling import has_proportional_error_model
from pharmpy.modeling.help_functions import _format_input_list
from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment
from pharmpy.symbols import symbol as S


def set_power_on_ruv(model, list_of_eps=None, ipred=None):
    """Applies a power effect to provided epsilons.

    Initial estimates for new thetas are 1 if the error
    model is proportional, otherwise they are 0.1.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_of_eps : str, list
        Name/names of epsilons to apply power effect. If None, all epsilons will be used.
        None is default.
    ipred : Symbol
        Symbol to use as IPRED. Default is to autodetect expression for IPRED.

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_power_on_ruv(model)   # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
                 power₁
    Y := EPS(1)⋅F       + F

    See also
    --------
    set_iiv_on_ruv

    """
    list_of_eps = _format_input_list(list_of_eps)
    eps = model.random_variables.epsilons
    if list_of_eps is not None:
        eps = eps[list_of_eps]
    pset, sset = model.parameters, model.statements

    if ipred is None:
        ipred = get_ipred(model)
    else:
        ipred = sympy.sympify(ipred)

    if has_proportional_error_model(model):
        theta_init = 1
    else:
        theta_init = 0.1

    # Find for example W = IPRED
    for s in sset:
        if isinstance(s, Assignment) and s.expression == ipred:
            alternative = s.symbol
            break
    else:
        alternative = None

    for i, e in enumerate(eps):
        theta_name = str(model.create_symbol(stem='power', force_numbering=True))
        theta = Parameter(theta_name, theta_init, lower=0.01)
        pset.append(theta)
        sset.subs({e.symbol * ipred: e.symbol})  # To avoid getting F*EPS*F**THETA
        if alternative:  # To avoid getting W*EPS*F**THETA
            sset.subs({e.symbol * alternative: e.symbol})
        sset.subs({e.name: ipred ** S(theta.name) * e.symbol})

    model.parameters = pset
    model.statements = sset

    return model


def get_ipred(model):
    expr = model.statements.after_odes.full_expression(model.dependent_variable)
    for rv in model.random_variables:
        expr = expr.subs(rv.symbol, 0)
    ipred = expr
    for s in model.statements:
        if isinstance(s, Assignment) and s.expression == ipred:
            ipred = s.symbol
            break
    return ipred
