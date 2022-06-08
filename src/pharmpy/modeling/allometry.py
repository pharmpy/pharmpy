from pharmpy.parameter import Parameter, Parameters
from pharmpy.statements import Assignment, sympify

from . import find_clearance_parameters, find_volume_parameters
from .expressions import create_symbol


def add_allometry(
    model,
    allometric_variable='WT',
    reference_value=70,
    parameters=None,
    initials=None,
    lower_bounds=None,
    upper_bounds=None,
    fixed=True,
):
    """Add allometric scaling of parameters

    Add an allometric function to each listed parameter. The function will be
    P=P*(X/Z)**T where P is the parameter, X the allometric_variable, Z the reference_value
    and T is a theta. Default is to automatically use clearance and volume parameters.

    Parameters
    ----------
    model : Model
        Pharmpy model
    allometric_variable : str or Symbol
        Variable to use for allometry (X above)
    reference_value : str, int, float or expression
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
    >>> add_allometry(model, allometric_variable='WGT')     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.before_odes
            ⎧TIME  for AMT > 0
            ⎨
    BTIME = ⎩ 0     otherwise
    TAD = -BTIME + TIME
    TVCL = THETA(1)⋅WGT
    TVV = THETA(2)⋅WGT
          ⎧TVV⋅(THETA(3) + 1)  for APGR < 5
          ⎨
    TVV = ⎩       TVV           otherwise
               ETA(1)
    CL = TVCL⋅ℯ
                 ALLO_CL
            ⎛WGT⎞
         CL⋅⎜───⎟
    CL =    ⎝ 70⎠
             ETA(2)
    V = TVV⋅ℯ
               ALLO_V
          ⎛WGT⎞
        V⋅⎜───⎟
    V =   ⎝ 70⎠
    S₁ = V

    """
    allometric_variable = sympify(allometric_variable)
    reference_value = sympify(reference_value)
    cls = find_clearance_parameters(model)
    vcs = find_volume_parameters(model)

    if parameters is None:
        parameters = cls + vcs
        if not parameters:
            raise ValueError("No parameters found")
    else:
        if not parameters:
            raise ValueError("No parameters provided")
        parameters = [sympify(p) for p in parameters]
    if initials is None:
        # Need to understand which parameter is CL or Q and which is V
        initials = []
        for p in parameters:
            if p in cls:
                initials.append(0.75)
            elif p in vcs:
                initials.append(1.0)

    if lower_bounds is None:
        lower_bounds = [0] * len(parameters)
    if upper_bounds is None:
        upper_bounds = [2] * len(parameters)

    if not (len(parameters) == len(initials) == len(lower_bounds) == len(upper_bounds)):
        raise ValueError("The number of parameters, initials and bounds must be the same")

    params = [p for p in model.parameters]
    for p, init, lower, upper in zip(parameters, initials, lower_bounds, upper_bounds):
        symb = create_symbol(model, f'ALLO_{p.name}')
        param = Parameter(symb.name, init=init, lower=lower, upper=upper, fix=fixed)
        params.append(param)
        expr = p * (allometric_variable / reference_value) ** param.symbol
        new_ass = Assignment(p, expr)
        p_ass = model.statements.find_assignment(p)
        model.statements.insert_after(p_ass, new_ass)
    model.parameters = Parameters(params)

    return model
