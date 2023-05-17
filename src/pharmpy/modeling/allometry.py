from __future__ import annotations

from typing import List, Optional, Union

from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.model import Assignment, Model, Parameter, Parameters

from .expressions import _create_symbol
from .odes import find_clearance_parameters, find_volume_parameters


def add_allometry(
    model: Model,
    allometric_variable: Union[str, sympy.Expr] = 'WT',
    reference_value: Union[str, int, float, sympy.Expr] = 70,
    parameters: Optional[List[Union[str, sympy.Expr]]] = None,
    initials: Optional[List[Union[int, float]]] = None,
    lower_bounds: Optional[List[Union[int, float]]] = None,
    upper_bounds: Optional[List[Union[int, float]]] = None,
    fixed: bool = True,
):
    """Add allometric scaling of parameters

    Add an allometric function to each listed parameter. The function will be
    P=P*(X/Z)**T where P is the parameter, X the allometric_variable, Z the reference_value
    and T is a theta. Default is to automatically use clearance and volume parameters.

    Parameters
    ----------
    model : Model
        Pharmpy model
    allometric_variable : str or sympy.Expr
        Value to use for allometry (X above)
    reference_value : str, int, float or sympy.Expr
        Reference value (Z above)
    parameters : list
        Parameters to use or None (default) for all available CL, Q and V parameters
    initials : list
        Initial estimates for the exponents. Default is to use 0.75 for CL and Qs and 1 for Vs
    lower_bounds : list
        Lower bounds for the exponents. Default is 0 for all parameters
    upper_bounds : list
        Upper bounds for the exponents. Default is 2 for all parameters
    fixed : bool
        Whether the exponents should be fixed

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, add_allometry
    >>> model = load_example_model("pheno")
    >>> model = add_allometry(model, allometric_variable='WGT')
    >>> model.statements.before_odes
            ⎧TIME  for AMT > 0
            ⎨
    BTIME = ⎩ 0     otherwise
    TAD = -BTIME + TIME
    TVCL = PTVCL⋅WGT
    TVV = PTVV⋅WGT
          ⎧TVV⋅(THETA₃ + 1)  for APGR < 5
          ⎨
    TVV = ⎩      TVV           otherwise
               ETA₁
    CL = TVCL⋅ℯ
                 ALLO_CL
            ⎛WGT⎞
         CL⋅⎜───⎟
    CL =    ⎝ 70⎠
             ETA₂
    V = TVV⋅ℯ
               ALLO_V
          ⎛WGT⎞
        V⋅⎜───⎟
    V =   ⎝ 70⎠
    S₁ = V

    """
    variable = parse_expr(allometric_variable)
    reference = parse_expr(reference_value)

    parsed_parameters = []

    if parameters is not None:
        parsed_parameters = [parse_expr(p) for p in parameters]

    if parameters is None or initials is None:
        cls = find_clearance_parameters(model)
        vcs = find_volume_parameters(model)

        if parameters is None:
            parsed_parameters = cls + vcs

        if initials is None:
            # Need to understand which parameter is CL or Q and which is V
            initials = []
            for p in parsed_parameters:
                if p in cls:
                    initials.append(0.75)
                elif p in vcs:
                    initials.append(1.0)

    if not parsed_parameters:
        raise ValueError("No parameters provided")

    if lower_bounds is None:
        lower_bounds = [0.0] * len(parsed_parameters)
    if upper_bounds is None:
        upper_bounds = [2.0] * len(parsed_parameters)

    if not (len(parsed_parameters) == len(initials) == len(lower_bounds) == len(upper_bounds)):
        raise ValueError("The number of parameters, initials and bounds must be the same")
    sset = model.statements
    params = list(model.parameters)
    for p, init, lower, upper in zip(parsed_parameters, initials, lower_bounds, upper_bounds):
        symb = _create_symbol(
            sset, params, model.random_variables, model.datainfo, f'ALLO_{p.name}', False
        )
        param = Parameter(symb.name, init=init, lower=lower, upper=upper, fix=fixed)
        params.append(param)
        expr = p * (variable / reference) ** param.symbol
        new_ass = Assignment(p, expr)
        ind = sset.find_assignment_index(p)
        sset = sset[0 : ind + 1] + new_ass + sset[ind + 1 :]
    parameters = Parameters.create(params)
    model = model.replace(statements=sset, parameters=parameters)
    model = model.update_source()

    return model
