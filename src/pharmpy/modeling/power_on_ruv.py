"""
:meta private:
"""
from __future__ import annotations

from typing import Optional, Union

from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.model import Assignment, Model, Parameter, Parameters

from .error import has_proportional_error_model
from .expressions import create_symbol
from .help_functions import _format_input_list


def set_power_on_ruv(
    model: Model,
    list_of_eps: Optional[Union[str, list]] = None,
    lower_limit: Optional[float] = 0.01,
    ipred: Optional[Union[str, sympy.Symbol]] = None,
    zero_protection: bool = False,
):
    """Applies a power effect to provided epsilons.

    Initial estimates for new thetas are 1 if the error
    model is proportional, otherwise they are 0.1.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_of_eps : str or list or None
        Name/names of epsilons to apply power effect. If None, all epsilons will be used.
        None is default.
    lower_limit : float or None
        Lower limit of power (theta). None for no limit.
    ipred : Symbol
        Symbol to use as IPRED. Default is to autodetect expression for IPRED.
    zero_protection : bool
        Set to True to add code protecting from IPRED=0

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
    Y = EPS(1)⋅F       + F

    See also
    --------
    set_iiv_on_ruv

    """
    list_of_eps = _format_input_list(list_of_eps)
    eps = model.random_variables.epsilons
    if list_of_eps is not None:
        eps = eps[list_of_eps]
    pset, sset = list(model.parameters), model.statements

    if ipred is None:
        ipred = get_ipred(model)
    else:
        ipred = parse_expr(ipred)

    if has_proportional_error_model(model):
        theta_init = 1
    else:
        theta_init = 0.1

    # Find for example W = IPRED
    for s in sset:
        if isinstance(s, Assignment) and s.expression == ipred:
            alternative = s.symbol
            if zero_protection:
                guard_expr = sympy.Piecewise(
                    (2.225e-307, sympy.Eq(s.expression, 0)), (s.expression, True)
                )
                guard_assignment = Assignment(ipred, guard_expr)
                ind = sset.find_assignment_index('Y')
                sset = sset[0:ind] + guard_assignment + sset[ind:]
            break
    else:
        alternative = None

    for e in eps.names:
        theta_name = str(create_symbol(model, stem='power', force_numbering=True))
        if lower_limit is None:
            theta = Parameter(theta_name, theta_init)
        else:
            theta = Parameter(theta_name, theta_init, lower=lower_limit)
        pset.append(theta)
        sset = sset.subs(
            {sympy.Symbol(e) * ipred: sympy.Symbol(e)}
        )  # To avoid getting F*EPS*F**THETA
        if alternative:  # To avoid getting W*EPS*F**THETA
            sset = sset.subs({sympy.Symbol(e) * alternative: sympy.Symbol(e)})
        sset = sset.subs({sympy.Symbol(e): ipred ** sympy.Symbol(theta.name) * sympy.Symbol(e)})
        model.statements = sset

    model.parameters = Parameters(pset)
    model.statements = sset

    return model


def get_ipred(model):
    expr = model.statements.after_odes.full_expression(model.dependent_variable)
    ipred = subs(
        expr, {sympy.Symbol(rv): 0 for rv in model.random_variables.names}, simultaneous=True
    )
    for s in model.statements:
        if isinstance(s, Assignment) and s.expression == ipred:
            ipred = s.symbol
            break
    return ipred
