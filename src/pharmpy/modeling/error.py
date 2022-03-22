"""
:meta private:
"""

import sympy

import pharmpy.symbols as symbols
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariable
from pharmpy.statements import Assignment, sympify

from .common import remove_unused_parameters_and_rvs
from .data import get_observations
from .expressions import create_symbol


def _preparations(model):
    stats = model.statements
    y = model.dependent_variable
    f = model.statements.find_assignment(y.name).expression
    for eps in model.random_variables.epsilons:
        f = f.subs({symbols.symbol(eps.name): 0})
    return stats, y, f


def _canonicalize_data_transformation(model, value):
    if value is None:
        value = model.dependent_variable
    else:
        value = sympify(value)
        if value.free_symbols != {model.dependent_variable}:
            raise ValueError(
                f"Expression for data transformation must contain the dependent variable "
                f"{model.dependent_variable} and no other variables"
            )
    return value


def remove_error_model(model):
    """Remove error model.

    Parameters
    ----------
    model : Model
        Remove error model for this model

    Return
    ------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import remove_error_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.statements.find_assignment("Y")
    Y := EPS(1)⋅W + F
    >>> remove_error_model(model)    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
    Y := F

    Warnings
    --------
    Removing the error model will make the model unrunable for some tools.

    """
    stats, y, f = _preparations(model)
    stats.reassign(y, f)
    remove_unused_parameters_and_rvs(model)
    return model


def set_additive_error_model(model, data_trans=None, series_terms=2):
    r"""Set an additive error model. Initial estimate for new sigma is :math:`(min(DV)/2)²`.

    The error function being applied depends on the data transformation. The table displays
    some examples.

    +------------------------+----------------------------------------+
    | Data transformation    | Additive error                         |
    +========================+========================================+
    | :math:`y`              | :math:`f + \epsilon_1`                 |
    +------------------------+----------------------------------------+
    | :math:`log(y)`         | :math:`\log(f) + \frac{\epsilon_1}{f}` |
    +------------------------+----------------------------------------+

    Parameters
    ----------
    model : Model
        Set error model for this model
    data_trans : str or expression
        A data transformation expression or None (default) to use the transformation
        specified by the model. Series expansion will be used for approximation.
    series_terms : int
        Number of terms to use for the series expansion approximation for data
        transformation.

    Return
    ------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import set_additive_error_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.statements.find_assignment("Y")
    Y := EPS(1)⋅W + F
    >>> set_additive_error_model(model)    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
    Y := F + εₐ

    >>> from pharmpy.modeling import set_additive_error_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.statements.find_assignment("Y")
    Y := EPS(1)⋅W + F
    >>> set_additive_error_model(model, data_trans="log(Y)")    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
                  εₐ
         log(F) + ──
    Y :=          F

    See Also
    --------
    set_proportional_error_model : Proportional error model
    set_combined_error_model : Combined error model

    """
    if has_additive_error_model(model):
        return model
    stats, y, f = _preparations(model)
    ruv = create_symbol(model, 'epsilon_a')

    data_trans = _canonicalize_data_transformation(model, data_trans)
    expr = f + ruv
    if data_trans != model.dependent_variable:
        expr = data_trans.subs(model.dependent_variable, expr).series(ruv, n=series_terms).removeO()

    stats.reassign(y, expr)
    remove_unused_parameters_and_rvs(model)

    sigma = create_symbol(model, 'sigma')
    sigma_par = Parameter(sigma.name, init=_get_prop_init(model))
    model.parameters.append(sigma_par)

    eps = RandomVariable.normal(ruv.name, 'RUV', 0, sigma)
    model.random_variables.append(eps)
    return model


def _get_prop_init(model):
    dv_min = get_observations(model).min()
    if dv_min == 0:
        return 0.01
    else:
        return (dv_min / 2) ** 2


def set_proportional_error_model(model, data_trans=None, zero_protection=False):
    r"""Set a proportional error model. Initial estimate for new sigma is 0.09.

    The error function being applied depends on the data transformation.

    +------------------------+----------------------------------------+
    | Data transformation    | Proportional error                     |
    +========================+========================================+
    | :math:`y`              | :math:`f + f \epsilon_1`               |
    +------------------------+----------------------------------------+
    | :math:`log(y)`         | :math:`\log(f) + \epsilon_1`           |
    +------------------------+----------------------------------------+

    Parameters
    ----------
    model : Model
        Set error model for this model
    data_trans : str or expression
        A data transformation expression or None (default) to use the transformation
        specified by the model.
    zero_protection : bool
        Set to True to add code protecting from IPRED=0

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> set_proportional_error_model(model)    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
    Y := F⋅εₚ + F

    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> set_proportional_error_model(model, data_trans="log(Y)", zero_protection=True)    # doctest: +ELLIPSIS  # noqa: E501
    <...>
    >>> model.statements.after_odes
         A_CENTRAL
         ─────────
    F :=     S₁
    W := F
                ⎧2.225e-16  for F = 0
                ⎨
    IPREDADJ := ⎩    F       otherwise
    Y := εₚ + log(IPREDADJ)
    IPRED := F
    IRES := DV - IPRED
             IRES
             ────
    IWRES :=  W

    See Also
    --------
    set_additive_error_model : Additive error model
    set_combined_error_model : Combined error model

    """
    if has_proportional_error_model(model):
        return model
    stats, y, f = _preparations(model)
    ruv = create_symbol(model, 'epsilon_p')

    data_trans = _canonicalize_data_transformation(model, data_trans)
    if zero_protection:
        ipred = create_symbol(model, 'IPREDADJ')
        guard_expr = sympy.Piecewise((2.225e-16, sympy.Eq(f, 0)), (f, True))
        guard_assignment = Assignment(ipred, guard_expr)
    else:
        ipred = f

    if data_trans == sympy.log(model.dependent_variable):
        if zero_protection:
            stats.insert_before(stats.find_assignment(y), guard_assignment)
        expr = sympy.log(ipred) + ruv
    elif data_trans == model.dependent_variable:
        if zero_protection:
            stats.insert_before(stats.find_assignment(y), guard_assignment)
        expr = f + ipred * ruv
    else:
        raise ValueError(f"Not supported data transformation {data_trans}")

    stats.reassign(y, expr)
    remove_unused_parameters_and_rvs(model)

    sigma = create_symbol(model, 'sigma')
    sigma_par = Parameter(sigma.name, init=0.09)
    model.parameters.append(sigma_par)

    eps = RandomVariable.normal(ruv.name, 'RUV', 0, sigma)
    model.random_variables.append(eps)
    return model


def set_combined_error_model(model, data_trans=None):
    r"""Set a combined error model. Initial estimates for new sigmas are :math:`(min(DV)/2)²` for
    proportional and 0.09 for additive.

    The error function being applied depends on the data transformation.

    +------------------------+-----------------------------------------------------+
    | Data transformation    | Combined error                                      |
    +========================+=====================================================+
    | :math:`y`              | :math:`f + f \epsilon_1 + \epsilon_2`               |
    +------------------------+-----------------------------------------------------+
    | :math:`log(y)`         | :math:`\log(f) + \epsilon_1 + \frac{\epsilon_2}{f}` |
    +------------------------+-----------------------------------------------------+

    Parameters
    ----------
    model : Model
        Set error model for this model
    data_trans : str or expression
        A data transformation expression or None (default) to use the transformation
        specified by the model.

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> set_combined_error_model(model)    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
    Y := F⋅εₚ + F + εₐ

    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> set_combined_error_model(model, data_trans="log(Y)")    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
                      εₐ
        εₚ + log(F) + ──
    Y :=              F

    See Also
    --------
    set_additive_error_model : Additive error model
    set_proportional_error_model: Proportional error model

    """
    if has_combined_error_model(model):
        return model
    stats, y, f = _preparations(model)

    expr = stats.find_assignment(y.name).expression

    ruv_prop = create_symbol(model, 'epsilon_p')
    ruv_add = create_symbol(model, 'epsilon_a')

    eta_ruv = symbols.symbol('ETA_RV1')
    theta_time = symbols.symbol('time_varying')

    data_trans = _canonicalize_data_transformation(model, data_trans)
    if data_trans == sympy.log(model.dependent_variable):
        expr_combined = sympy.log(f) + ruv_prop + ruv_add / f
    elif data_trans == model.dependent_variable:
        if isinstance(expr, sympy.Piecewise):
            expr_0 = expr.args[0][0]
            expr_1 = expr.args[1][0]
            cond_0 = expr.args[0][1]
            for eps in model.random_variables.epsilons:
                expr_0 = expr_0.subs({symbols.symbol(eps.name): ruv_prop})
                expr_1 = expr_1.subs({symbols.symbol(eps.name): ruv_prop})
                if (
                    eta_ruv in model.random_variables.free_symbols
                    and theta_time in model.parameters.symbols
                ):
                    expr_combined = sympy.Piecewise(
                        (expr_0 + ruv_add * theta_time * sympy.exp(eta_ruv), cond_0),
                        (expr_1 + ruv_add * sympy.exp(eta_ruv), True),
                    )
                elif (
                    eta_ruv not in model.random_variables.free_symbols
                    and theta_time in model.parameters.symbols
                ):
                    expr_combined = sympy.Piecewise(
                        (expr_0 + ruv_add * theta_time, cond_0), (expr_1 + ruv_add, True)
                    )
        elif (
            eta_ruv in model.random_variables.free_symbols
            and theta_time not in model.parameters.symbols
        ):
            expr_combined = f + f * ruv_prop * sympy.exp(eta_ruv) + ruv_add * sympy.exp(eta_ruv)
        else:
            expr_combined = f + f * ruv_prop + ruv_add
    else:
        raise ValueError(f"Not supported data transformation {data_trans}")

    stats.reassign(y, expr_combined)
    remove_unused_parameters_and_rvs(model)

    sigma_prop = create_symbol(model, 'sigma_prop')
    sigma_par1 = Parameter(sigma_prop.name, init=0.09)
    model.parameters.append(sigma_par1)
    sigma_add = create_symbol(model, 'sigma_add')
    sigma_par2 = Parameter(sigma_add.name, init=_get_prop_init(model))
    model.parameters.append(sigma_par2)

    eps_prop = RandomVariable.normal(ruv_prop.name, 'RUV', 0, sigma_prop)
    model.random_variables.append(eps_prop)
    eps_add = RandomVariable.normal(ruv_add.name, 'RUV', 0, sigma_add)
    model.random_variables.append(eps_add)
    return model


def has_additive_error_model(model):
    """Check if a model has an additive error model

    Parameters
    ----------
    model : Model
        The model to check

    Return
    ------
    bool
        True if the model has an additive error model and False otherwise

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, has_additive_error_model
    >>> model = load_example_model("pheno")
    >>> has_additive_error_model(model)
    False

    See Also
    --------
    has_proportional_error_model : Check if a model has a proportional error model
    has_combined_error_model : Check if a model has a combined error model
    """
    y = model.dependent_variable
    expr = model.statements.after_odes.full_expression(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {
        symbols.symbol(rv.name) for rv in rvs if symbols.symbol(rv.name) in expr.free_symbols
    }
    if len(rvs_in_y) != 1:
        return False
    eps = rvs_in_y.pop()
    return eps not in (expr - eps).simplify().free_symbols


def has_proportional_error_model(model):
    """Check if a model has a proportional error model

    Parameters
    ----------
    model : Model
        The model to check

    Return
    ------
    bool
        True if the model has a proportional error model and False otherwise

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, has_proportional_error_model
    >>> model = load_example_model("pheno")
    >>> has_proportional_error_model(model)
    True

    See Also
    --------
    has_additive_error_model : Check if a model has an additive error model
    has_combined_error_model : Check if a model has a combined error model
    """
    y = model.dependent_variable
    expr = model.statements.after_odes.full_expression(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {
        symbols.symbol(rv.name) for rv in rvs if symbols.symbol(rv.name) in expr.free_symbols
    }
    if len(rvs_in_y) != 1:
        return False
    eps = rvs_in_y.pop()
    return eps not in (expr / (1 + eps)).simplify().free_symbols


def has_combined_error_model(model):
    """Check if a model has a combined additive and proportinal error model

    Parameters
    ----------
    model : Model
        The model to check

    Return
    ------
    bool
        True if the model has a combined error model and False otherwise

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, has_combined_error_model
    >>> model = load_example_model("pheno")
    >>> has_combined_error_model(model)
    False

    See Also
    --------
    has_additive_error_model : Check if a model has an additive error model
    has_proportional_error_model : Check if a model has a proportional error model
    """
    y = model.dependent_variable
    expr = model.statements.after_odes.full_expression(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {
        symbols.symbol(rv.name) for rv in rvs if symbols.symbol(rv.name) in expr.free_symbols
    }
    if len(rvs_in_y) != 2:
        return False
    eps1 = rvs_in_y.pop()
    eps2 = rvs_in_y.pop()
    canc1 = ((expr - eps1) / (eps2 + 1)).simplify()
    canc2 = ((expr - eps2) / (eps1 + 1)).simplify()
    return (
        eps1 not in canc1.free_symbols
        and eps2 not in canc1.free_symbols
        or eps1 not in canc2.free_symbols
        and eps2 not in canc2.free_symbols
    )


def use_thetas_for_error_stdev(model):
    """Use thetas to estimate standard deviation of error

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, use_thetas_for_error_stdev
    >>> model = load_example_model("pheno")
    >>> use_thetas_for_error_stdev(model)    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
    Y := EPS(1)⋅SD_EPS(1)⋅W + F

    See also
    --------
    set_weighted_error_model : Encode error model with one epsilon and weight
    """
    rvs = model.random_variables.epsilons
    for eps in rvs:
        sigmas = eps.parameter_names
        if len(sigmas) > 1:
            raise ValueError('use_thetas_for_error_stdev only supports non-correlated sigmas')
        sigma = sigmas[0]
        param = model.parameters[sigma]
        param.fix = True
        theta_init = param.init**0.5
        param.init = 1
        theta = Parameter(f'SD_{eps.name}', theta_init, lower=0)
        model.parameters.append(theta)
        symb = sympy.Symbol(eps.name)
        model.statements.subs({symb: theta.symbol * symb})
    return model


def set_weighted_error_model(model):
    """Encode error model with one epsilon and W as weight

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_weighted_error_model
    >>> model = load_example_model("pheno")
    >>> set_weighted_error_model(model)    # doctest: +ELLIPSIS
    <...>

    See also
    --------
    use_thetas_for_error_stdev : Use thetas to estimate error
    """
    stats, y, f = _preparations(model)
    epsilons = model.random_variables.epsilons
    expr = stats.find_assignment(y.name).expression
    ssum = 0
    q = sympy.Q.real(y)  # Dummy predicate
    for term in expr.args:
        eps = [x for x in term.free_symbols if x.name in epsilons.names]
        if len(eps) > 0:
            eps = eps[0]
            remaining = term / eps
            ssum += remaining**2
            for symb in remaining.free_symbols:
                q &= sympy.Q.positive(symb)
    w = sympy.sqrt(ssum)
    w = sympy.refine(w, q)

    for i, s in enumerate(stats):
        if isinstance(s, Assignment) and s.symbol == y:
            stats.insert(i, Assignment('W', w))
            break

    stats.reassign(y, f + sympy.Symbol('W') * sympy.Symbol(epsilons[0].name))
    remove_unused_parameters_and_rvs(model)
    return model


def set_dtbs_error_model(model, fix_to_log=False):
    """Dynamic transform both sides

    Parameters
    ----------
    model : Model
        Pharmpy model
    fix_to_log : Boolean
        Set to True to fix lambda and zeta to 0, i.e. emulating log-transformed data

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_dtbs_error_model
    >>> model = load_example_model("pheno")
    >>> set_dtbs_error_model(model)    # doctest: +ELLIPSIS
    <...>

    """
    use_thetas_for_error_stdev(model)
    set_weighted_error_model(model)
    stats, y, f = _preparations(model)
    tbs_lambda = Parameter('tbs_lambda', 1)
    tbs_zeta = Parameter('tbs_zeta', 0.001)
    if fix_to_log:
        tbs_lambda.fix = True
        tbs_lambda.init = 0
        tbs_zeta.fix = True
        tbs_zeta.init = 0
    model.parameters.append(tbs_lambda)
    model.parameters.append(tbs_zeta)
    lam = tbs_lambda.symbol
    zeta = tbs_zeta.symbol

    for i, s in enumerate(stats):
        if isinstance(s, Assignment) and s.symbol == sympy.Symbol('W'):
            break

    stats.insert(i + 1, Assignment('W', (f**zeta) * sympy.Symbol('W')))
    ipred = sympy.Piecewise(
        ((f**lam - 1) / lam, sympy.And(sympy.Ne(lam, 0), sympy.Ne(f, 0))),
        (sympy.log(f), sympy.And(sympy.Eq(lam, 0), sympy.Ne(f, 0))),
        (-1 / lam, sympy.And(sympy.Eq(lam, 0), sympy.Eq(f, 0))),
        (-1000000000, True),
    )
    stats.insert(i + 2, Assignment('IPRED', ipred))
    yexpr = stats.find_assignment(y.name)
    yexpr.subs({f: sympy.Symbol('IPRED')})

    obs = sympy.Piecewise(
        (sympy.log(y), sympy.Eq(lam, 0)), ((y**lam - 1) / lam, sympy.Ne(lam, 0))
    )
    model.observation_transformation = obs

    return model


def set_time_varying_error_model(model, cutoff, idv='TIME'):
    """Set a time varying error model per time cutoff

    Parameters
    ----------
    model : Model
        Pharmpy model
    cutoff : float
        A value at the given quantile over idv column
    idv : str
        Time or time after dose, default is Time

    Return
    ------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_time_varying_error_model
    >>> model = load_example_model("pheno")
    >>> set_time_varying_error_model(model, cutoff=1.0)    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
         ⎧EPS(1)⋅W⋅time_varying + F  for TIME < 1.0
         ⎨
    Y := ⎩      EPS(1)⋅W + F           otherwise

    """
    stats = model.statements
    y = stats.find_assignment('Y')
    idv = sympify(idv)
    theta = create_symbol(model, 'time_varying')
    eps = model.random_variables.epsilons
    expr = sympy.Piecewise(
        (y.expression.subs({e.symbol: e.symbol * theta for e in eps}), idv < cutoff),
        (y.expression, True),
    )
    stats.reassign(y.symbol, expr)

    theta_tvar = Parameter(theta.name, init=0.1)
    model.parameters.append(theta_tvar)

    return model
