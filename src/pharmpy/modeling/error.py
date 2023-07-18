"""
:meta private:
"""
from __future__ import annotations

from typing import List, Optional, Union

from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.model import Assignment, Model, NormalDistribution, Parameter, Parameters, Statements

from .blq import get_blq_symb_and_type, get_sd_expr, has_blq_transformation
from .common import _get_unused_parameters_and_rvs, remove_unused_parameters_and_rvs
from .data import get_observations
from .expressions import _create_symbol, create_symbol, get_dv_symbol
from .help_functions import _format_input_list, _get_epsilons
from .parameters import add_population_parameter, fix_parameters, set_initial_estimates


def _preparations(model, y=None):
    stats = model.statements
    # FIXME: handle other DVs?
    if y is None:
        y = list(model.dependent_variables.keys())[0]
    f = subs(
        model.statements.find_assignment(y.name).expression,
        {sympy.Symbol(eps): 0 for eps in model.random_variables.epsilons.names},
        simultaneous=True,
    )
    return stats, y, f


def _canonicalize_data_transformation(model, value, dv):
    if value is None:
        value = dv
    else:
        value = parse_expr(value)
        if value.free_symbols != {dv}:
            raise ValueError(
                f"Expression for data transformation must contain the dependent variable "
                f"{dv} and no other variables"
            )
    return value


def remove_error_model(model: Model):
    """Remove error model.

    Parameters
    ----------
    model : Model
        Remove error model for this model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import remove_error_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.statements.find_assignment("Y")
    Y = EPS₁⋅W + F
    >>> model = remove_error_model(model)
    >>> model.statements.find_assignment("Y")
    Y = F

    Warnings
    --------
    Removing the error model will make the model unrunable for some tools.

    """
    stats, y, f = _preparations(model)
    model = model.replace(statements=stats.reassign(y, f))
    model = remove_unused_parameters_and_rvs(model)
    return model.update_source()


def set_additive_error_model(
    model: Model,
    dv: Union[sympy.Symbol, str, int, None] = None,
    data_trans: Optional[Union[str, sympy.Expr]] = None,
    series_terms: int = 2,
):
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
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)
    data_trans : str or expression
        A data transformation expression or None (default) to use the transformation
        specified by the model. Series expansion will be used for approximation.
    series_terms : int
        Number of terms to use for the series expansion approximation for data
        transformation.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import set_additive_error_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.statements.find_assignment("Y")
    Y = EPS₁⋅W + F
    >>> model = set_additive_error_model(model)
    >>> model.statements.find_assignment("Y")
    Y = F + εₐ

    >>> from pharmpy.modeling import set_additive_error_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.statements.find_assignment("Y")
    Y = EPS₁⋅W + F
    >>> model = set_additive_error_model(model, data_trans="log(Y)")
    >>> model.statements.find_assignment("Y")
                 εₐ
        log(F) + ──
    Y =          F

    See Also
    --------
    set_proportional_error_model : Proportional error model
    set_combined_error_model : Combined error model

    """
    dv = get_dv_symbol(model, dv)
    if has_additive_error_model(model, dv):
        return model
    stats, y, f = _preparations(model, dv)
    ruv = create_symbol(model, 'epsilon_a')

    data_trans = _canonicalize_data_transformation(model, data_trans, dv)
    expr = f + ruv

    if data_trans != dv:
        expr = subs(data_trans, {dv: expr}, simultaneous=True).series(ruv, n=series_terms).removeO()

    sigma = create_symbol(model, 'sigma')
    model = add_population_parameter(model, sigma.name, _get_prop_init(model))

    eps = NormalDistribution.create(ruv.name, 'RUV', 0, sigma)

    rvs_new, params_new = _get_unused_parameters_and_rvs(
        stats.reassign(y, expr), model.parameters, model.random_variables + eps
    )

    model = model.replace(
        statements=stats.reassign(y, expr), random_variables=rvs_new, parameters=params_new
    )
    return model.update_source()


def _get_prop_init(model):
    dv_min = get_observations(model).min()
    if dv_min == 0:
        return 0.01
    else:
        return (dv_min / 2) ** 2


def set_proportional_error_model(
    model: Model,
    dv: Union[sympy.Symbol, str, int, None] = None,
    data_trans: Optional[Union[str, sympy.Expr]] = None,
    zero_protection: bool = True,
):
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
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)
    data_trans : str or expression
        A data transformation expression or None (default) to use the transformation
        specified by the model.
    zero_protection : bool
        Set to True to add code protecting from IPRED=0

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> model = set_proportional_error_model(model)
    >>> model.statements.after_odes
        A_CENTRAL
        ─────────
    F =     S₁
    W = F
               ⎧2.225e-16  for F = 0
               ⎨
    IPREDADJ = ⎩    F      otherwise
    Y = F + IPREDADJ⋅εₚ
    IPRED = F
    IRES = DV - IPRED
            IRES
            ────
    IWRES =  W

    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> model = set_proportional_error_model(
    ...     model,
    ...     data_trans="log(Y)"
    ... )
    >>> model.statements.after_odes
        A_CENTRAL
        ─────────
    F =     S₁
    W = F
               ⎧2.225e-16  for F = 0
               ⎨
    IPREDADJ = ⎩    F       otherwise
    Y = εₚ + log(IPREDADJ)
    IPRED = F
    IRES = DV - IPRED
            IRES
            ────
    IWRES =  W

    See Also
    --------
    set_additive_error_model : Additive error model
    set_combined_error_model : Combined error model

    """
    dv = get_dv_symbol(model, dv)
    if has_proportional_error_model(model, dv):
        return model

    stats, y, f = _preparations(model, dv)
    ruv = create_symbol(model, 'epsilon_p')

    data_trans = _canonicalize_data_transformation(model, data_trans, dv)
    ipred = create_symbol(model, 'IPREDADJ') if zero_protection else f

    sigma = create_symbol(model, 'sigma')
    model = add_population_parameter(model, sigma.name, 0.09)

    eps = NormalDistribution.create(ruv.name, 'RUV', 0, sigma)

    f_dummy = sympy.Dummy('x')
    if data_trans == sympy.log(dv):
        error_expr = sympy.log(ipred) + ruv if zero_protection else sympy.log(f_dummy) + ruv
    elif data_trans == dv:
        error_expr = f_dummy + ipred * ruv if zero_protection else f_dummy + f_dummy * ruv
    else:
        raise ValueError(f"Not supported data transformation {data_trans}")

    if has_blq_transformation(model):
        f, stats_new = _get_updated_blq_statements(model, error_expr, f, f_dummy, eps)
    else:
        expr = error_expr.subs({f_dummy: f})
        stats_new = stats.reassign(y, expr)

    if zero_protection:
        guard_expr = sympy.Piecewise((2.225e-16, sympy.Eq(f, 0)), (f, True))
        guard_assignment = Assignment(ipred, guard_expr)
        ind = 0
        # Find first occurrence of IPREDADJ
        for i, s in enumerate(stats_new):
            if ipred in s.free_symbols:
                ind = i
                break
        stats_new = stats_new[0:ind] + guard_assignment + stats_new[ind:]

    rvs_new, params_new = _get_unused_parameters_and_rvs(
        stats_new, model.parameters, model.random_variables + eps
    )

    model = model.replace(statements=stats_new, random_variables=rvs_new, parameters=params_new)

    return model.update_source()


def _get_updated_blq_statements(model, expr_dummy, f, f_dummy, eps_new):
    y = list(model.dependent_variables.keys())[0]
    f_above_lloq = _get_f_above_lloq(model, f)
    expr_above_lloq = expr_dummy.subs({f_dummy: f_above_lloq})
    expr = f.subs({f_above_lloq: expr_above_lloq})
    # FIXME: make more general
    sd = model.statements.find_assignment('SD')
    sd_new = get_sd_expr(expr_above_lloq, model.random_variables + eps_new, model.parameters)
    stats_new = model.statements.reassign(sd.symbol, sd_new)
    stats_new = stats_new.reassign(y, expr)
    return f_above_lloq, stats_new


def _get_f_above_lloq(model, f):
    blq_symb, _ = get_blq_symb_and_type(model)
    for expr, cond in f.args:
        if blq_symb in cond.free_symbols:
            return expr
    else:
        raise AssertionError('BLQ symbol not found')


def set_combined_error_model(
    model: Model,
    dv: Union[sympy.Symbol, str, int, None] = None,
    data_trans: Optional[Union[str, sympy.Expr]] = None,
):
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
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)
    data_trans : str or expression
        A data transformation expression or None (default) to use the transformation
        specified by the model.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> model = set_combined_error_model(model)
    >>> model.statements.find_assignment("Y")
    Y = F⋅εₚ + F + εₐ

    >>> from pharmpy.modeling import *
    >>> model = remove_error_model(load_example_model("pheno"))
    >>> model = set_combined_error_model(model, data_trans="log(Y)")
    >>> model.statements.find_assignment("Y")
                     εₐ
       εₚ + log(F) + ──
    Y =              F

    See Also
    --------
    set_additive_error_model : Additive error model
    set_proportional_error_model: Proportional error model

    """
    dv = get_dv_symbol(model, dv)
    if has_combined_error_model(model, dv):
        return model
    stats, y, f = _preparations(model, dv)

    expr = stats.find_assignment(y.name).expression

    ruv_prop = create_symbol(model, 'epsilon_p')
    ruv_add = create_symbol(model, 'epsilon_a')

    eta_ruv = sympy.Symbol('ETA_RV1')
    theta_time = sympy.Symbol('time_varying')

    data_trans = _canonicalize_data_transformation(model, data_trans, dv)

    sigma_prop = create_symbol(model, 'sigma_prop')
    model = add_population_parameter(model, sigma_prop.name, 0.09)
    sigma_add = create_symbol(model, 'sigma_add')
    model = add_population_parameter(model, sigma_add.name, _get_prop_init(model))

    eps_prop = NormalDistribution.create(ruv_prop.name, 'RUV', 0, sigma_prop)
    eps_add = NormalDistribution.create(ruv_add.name, 'RUV', 0, sigma_add)

    # FIXME: handle other DVs
    dv = list(model.dependent_variables.keys())[0]
    f_dummy = sympy.Dummy('x')
    if data_trans == sympy.log(dv):
        error_expr = sympy.log(f_dummy) + ruv_prop + ruv_add / f_dummy
    elif data_trans == dv:
        # Time varying
        if isinstance(expr, sympy.Piecewise) and not has_blq_transformation(model):
            expr_0 = expr.args[0][0]
            expr_1 = expr.args[1][0]
            cond_0 = expr.args[0][1]
            error_expr = None
            for eps in model.random_variables.epsilons.names:
                expr_0 = subs(expr_0, {sympy.Symbol(eps): ruv_prop}, simultaneous=True)
                expr_1 = subs(expr_1, {sympy.Symbol(eps): ruv_prop}, simultaneous=True)
                if (
                    eta_ruv in model.random_variables.free_symbols
                    and theta_time in model.parameters.symbols
                ):
                    error_expr = sympy.Piecewise(
                        (expr_0 + ruv_add * theta_time * sympy.exp(eta_ruv), cond_0),
                        (expr_1 + ruv_add * sympy.exp(eta_ruv), True),
                    )
                elif (
                    eta_ruv not in model.random_variables.free_symbols
                    and theta_time in model.parameters.symbols
                ):
                    error_expr = sympy.Piecewise(
                        (expr_0 + ruv_add * theta_time, cond_0), (expr_1 + ruv_add, True)
                    )
            assert error_expr is not None
        elif (
            eta_ruv in model.random_variables.free_symbols
            and theta_time not in model.parameters.symbols
        ):
            if has_blq_transformation(model):
                raise ValueError('Currently not supported to change from IIV on RUV model with BLQ')
            error_expr = (
                f_dummy + f_dummy * ruv_prop * sympy.exp(eta_ruv) + ruv_add * sympy.exp(eta_ruv)
            )
        else:
            error_expr = f_dummy + f_dummy * ruv_prop + ruv_add
    else:
        raise ValueError(f"Not supported data transformation {data_trans}")

    if has_blq_transformation(model):
        _, stats_new = _get_updated_blq_statements(
            model, error_expr, f, f_dummy, [eps_prop, eps_add]
        )
    else:
        expr = error_expr.subs({f_dummy: f})
        stats_new = stats.reassign(y, expr)

    rvs_new, params_new = _get_unused_parameters_and_rvs(
        stats_new, model.parameters, model.random_variables + [eps_prop, eps_add]
    )
    model = model.replace(statements=stats_new, random_variables=rvs_new, parameters=params_new)

    return model.update_source()


def has_additive_error_model(model: Model, dv: Union[sympy.Symbol, str, int, None] = None):
    """Check if a model has an additive error model

    Multiple dependent variables are supported. By default the only (in case of one) or the
    first (in case of many) dependent variable is going to be checked.

    Parameters
    ----------
    model : Model
        The model to check
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)

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
    has_weighted_error_model : Check if a model has a weighted error model
    """
    y = get_dv_symbol(model, dv)
    expr = model.statements.error.full_expression(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {sympy.Symbol(name) for name in rvs.names if sympy.Symbol(name) in expr.free_symbols}
    if len(rvs_in_y) != 1:
        return False
    eps = rvs_in_y.pop()
    return eps not in (expr - eps).simplify().free_symbols


def has_proportional_error_model(model: Model, dv: Union[sympy.Symbol, str, int, None] = None):
    """Check if a model has a proportional error model

    Multiple dependent variables are supported. By default the only (in case of one) or the
    first (in case of many) dependent variable is going to be checked.

    Parameters
    ----------
    model : Model
        The model to check
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)

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
    has_weighted_error_model : Check if a model has a weighted error model
    """
    y = get_dv_symbol(model, dv)
    expr = model.statements.error.full_expression(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {sympy.Symbol(name) for name in rvs.names if sympy.Symbol(name) in expr.free_symbols}
    if len(rvs_in_y) != 1:
        return False
    eps = rvs_in_y.pop()
    return eps not in (expr / (1 + eps)).simplify().free_symbols


def has_combined_error_model(model: Model, dv: Union[sympy.Symbol, str, int, None] = None):
    """Check if a model has a combined additive and proportional error model

    Multiple dependent variables are supported. By default the only (in case of one) or the
    first (in case of many) dependent variable is going to be checked.

    Parameters
    ----------
    model : Model
        The model to check
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)

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
    has_weighted_error_model : Check if a model has a weighted error model
    """
    y = get_dv_symbol(model, dv)
    expr = model.statements.error.full_expression(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {sympy.Symbol(name) for name in rvs.names if sympy.Symbol(name) in expr.free_symbols}
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


def use_thetas_for_error_stdev(model: Model):
    """Use thetas to estimate standard deviation of error

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, use_thetas_for_error_stdev
    >>> model = load_example_model("pheno")
    >>> model = use_thetas_for_error_stdev(model)
    >>> model.statements.find_assignment("W")
    W = F⋅SD_EPS_1
    >>> model.statements.find_assignment("Y")
    Y = EPS₁⋅W + F

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
        theta_init = param.init**0.5
        model = fix_parameters(model, [sigma])
        model = set_initial_estimates(model, {sigma: 1})

        sdsymb = create_symbol(model, f'SD_{eps.names[0]}')
        model = add_population_parameter(model, sdsymb.name, theta_init, lower=0)
        symb = sympy.Symbol(eps.names[0])
        if has_weighted_error_model(model) and (
            has_additive_error_model(model) or has_proportional_error_model(model)
        ):
            w = get_weighted_error_model_weight(model)
            w_ass = model.statements.find_assignment(w)
            model = model.replace(
                statements=model.statements.reassign(w_ass.symbol, w_ass.expression * sdsymb)
            )
        else:
            model = model.replace(statements=model.statements.subs({symb: sdsymb * symb}))
    return model.update_source()


def set_weighted_error_model(model: Model):
    """Encode error model with one epsilon and W as weight

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_weighted_error_model
    >>> model = load_example_model("pheno")
    >>> model = set_weighted_error_model(model)

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

    i = _index_of_first_assignment(stats, y)

    model = model.replace(statements=stats[0:i] + Assignment(sympy.Symbol('W'), w) + stats[i:])
    model = model.replace(
        statements=model.statements.reassign(
            y, f + sympy.Symbol('W') * sympy.Symbol(epsilons[0].names[0])
        )
    )
    model = remove_unused_parameters_and_rvs(model)
    return model.update_source()


def has_weighted_error_model(model: Model):
    """Check if a model has a weighted error model

    Parameters
    ----------
    model : Model
        The model to check

    Return
    ------
    bool
        True if the model has a weighted error model and False otherwise

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, has_weighted_error_model
    >>> model = load_example_model("pheno")
    >>> has_weighted_error_model(model)
    True

    See Also
    --------
    has_additive_error_model : Check if a model has an additive error model
    has_combined_error_model : Check if a model has a combined error model
    has_proportional_error_model : Check if a model has a proportional error model
    """
    w = get_weighted_error_model_weight(model)

    if w:
        return True
    return False


def get_weighted_error_model_weight(model: Model):
    # Defines weighted error model as e.g.: Y = F + EPS(1)*W
    stats, y, f = _preparations(model)
    # FIXME: handle multiple DVs? Handle piecewise?
    y_expr = stats.error.find_assignment(y).expression
    rvs = model.random_variables.epsilons
    rvs_in_y = {
        sympy.Symbol(name) for name in rvs.names if sympy.Symbol(name) in y_expr.free_symbols
    }

    if len(rvs_in_y) > 1:
        return None

    f = y_expr.subs({rv: 0 for rv in rvs_in_y})
    w = None
    eps_expr = {arg for arg in y_expr.args if arg.free_symbols.intersection(rvs_in_y)}

    if len(eps_expr) == 1:
        eps_expr = eps_expr.pop()
        if len(eps_expr.args) == 2 and eps_expr.func is sympy.Mul:
            a, b = eps_expr.args
            w_cand = a if a not in rvs_in_y else b
            if w_cand != f:
                w = w_cand

    return w


def _index_of_first_assignment(statements: Statements, symbol: sympy.Symbol) -> int:
    return next(
        (i for i, s in enumerate(statements) if isinstance(s, Assignment) and s.symbol == symbol)
    )


def set_dtbs_error_model(model: Model, fix_to_log: bool = False):
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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_dtbs_error_model
    >>> model = load_example_model("pheno")
    >>> model = set_dtbs_error_model(model)

    """
    model = use_thetas_for_error_stdev(model)
    model = set_weighted_error_model(model)
    stats, y, f = _preparations(model)
    lam = create_symbol(model, 'tbs_lambda')
    zeta = create_symbol(model, 'tbs_zeta')
    if fix_to_log:
        model = add_population_parameter(model, lam.name, 0, fix=True)
        model = add_population_parameter(model, zeta.name, 0, fix=True)
    else:
        model = add_population_parameter(model, lam.name, 1)
        model = add_population_parameter(model, zeta.name, 0.001)

    i = _index_of_first_assignment(stats, sympy.Symbol('W'))

    wass = Assignment(sympy.Symbol('W'), (f**zeta) * sympy.Symbol('W'))
    ipred = sympy.Piecewise(
        ((f**lam - 1) / lam, sympy.And(sympy.Ne(lam, 0), sympy.Ne(f, 0))),
        (sympy.log(f), sympy.And(sympy.Eq(lam, 0), sympy.Ne(f, 0))),
        (-1 / lam, sympy.And(sympy.Eq(lam, 0), sympy.Eq(f, 0))),
        (-1000000000, True),
    )
    ipredass = Assignment(sympy.Symbol('IPRED'), ipred)
    yexpr_ind = stats.find_assignment_index(y.name)
    yexpr = stats[yexpr_ind].subs({f: sympy.Symbol('IPRED')})

    statements = (
        stats[0 : i + 1]
        + wass
        + ipredass
        + stats[i + 1 : yexpr_ind]
        + yexpr
        + stats[yexpr_ind + 1 :]
    )

    obs = model.observation_transformation
    obs = obs.replace(
        y,
        sympy.Piecewise((sympy.log(y), sympy.Eq(lam, 0)), ((y**lam - 1) / lam, sympy.Ne(lam, 0))),
    )
    model = model.replace(observation_transformation=obs, statements=statements)

    return model.update_source()


def set_time_varying_error_model(
    model: Model, cutoff: float, idv: str = 'TIME', dv: Union[sympy.Symbol, str, int, None] = None
):
    """Set a time varying error model per time cutoff

    Parameters
    ----------
    model : Model
        Pharmpy model
    cutoff : float
        A value at the given quantile over idv column
    idv : str
        Time or time after dose, default is Time
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_time_varying_error_model
    >>> model = load_example_model("pheno")
    >>> model = set_time_varying_error_model(model, cutoff=1.0)
    >>> model.statements.find_assignment("Y")
        ⎧EPS₁⋅W⋅time_varying + F  for TIME < 1.0
        ⎨
    Y = ⎩      EPS₁⋅W + F           otherwise

    """
    dv = get_dv_symbol(model, dv)
    y = model.statements.find_assignment(dv)
    idv = parse_expr(idv)
    theta = create_symbol(model, 'time_varying')
    eps = model.random_variables.epsilons
    expr = sympy.Piecewise(
        (
            subs(
                y.expression,
                {sympy.Symbol(e): sympy.Symbol(e) * theta for e in eps.names},
                simultaneous=True,
            ),
            idv < cutoff,
        ),
        (y.expression, True),
    )
    model = add_population_parameter(model, theta.name, 0.1)
    model = model.replace(statements=model.statements.reassign(y.symbol, expr))

    return model.update_source()


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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_power_on_ruv(model)
    >>> model.statements.find_assignment("Y")
              power₁
    Y = EPS₁⋅F       + F

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
    ipredadj = None
    alternative = None
    for s in sset:
        if isinstance(s, Assignment):
            if s.expression == ipred:
                alternative = s.symbol
                if zero_protection:
                    guard_expr = sympy.Piecewise(
                        (2.225e-307, sympy.Eq(s.expression, 0)), (s.expression, True)
                    )
                    guard_assignment = Assignment(ipred, guard_expr)
                    ind = sset.find_assignment_index('Y')
                    sset = sset[0:ind] + guard_assignment + sset[ind:]
                break
            if isinstance(s.expression, sympy.Piecewise):
                args = s.expression.args
                for expr, cond in args:
                    if expr == ipred:
                        ipredadj = s.symbol
                        break

        if has_blq_transformation(model):
            _, _, f = _preparations(model)
            ipred = _get_f_above_lloq(model, f)

    for e in eps.names:
        theta_name = str(
            _create_symbol(
                sset, pset, model.random_variables, model.datainfo, 'power', force_numbering=True
            )
        )
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

        if ipredadj:
            sset = sset.subs(
                {sympy.Symbol(e) * ipredadj: ipredadj ** sympy.Symbol(theta.name) * sympy.Symbol(e)}
            )
        else:
            sset = sset.subs({sympy.Symbol(e): ipred ** sympy.Symbol(theta.name) * sympy.Symbol(e)})

        if has_blq_transformation(model):
            # FIXME: make more general
            y_above_lloq, _ = sset.find_assignment('Y').expression.args[0]
            sd = model.statements.find_assignment('SD')
            sd_new = get_sd_expr(y_above_lloq, model.random_variables, Parameters.create(pset))
            sset = sset.reassign(sd.symbol, sd_new)

    model = model.replace(parameters=Parameters.create(pset), statements=sset)

    return model.update_source()


def get_ipred(model):
    # FIXME: handle other DVs?
    dv = list(model.dependent_variables.keys())[0]
    expr = model.statements.after_odes.full_expression(dv)
    ipred = subs(
        expr, {sympy.Symbol(rv): 0 for rv in model.random_variables.names}, simultaneous=True
    )
    for s in model.statements:
        if isinstance(s, Assignment) and s.expression == ipred:
            ipred = s.symbol
            break
    return ipred


def set_iiv_on_ruv(
    model: Model,
    dv: Union[sympy.Symbol, str, int, None] = None,
    list_of_eps: Optional[Union[List[str], str]] = None,
    same_eta: bool = True,
    eta_names: Optional[Union[List[str], str]] = None,
):
    """
    Multiplies epsilons with exponential (new) etas.

    Initial variance for new etas is 0.09.

    Parameters
    ----------
    model : Model
        Pharmpy model to apply IIV on epsilons.
    list_of_eps : str, list
        Name/names of epsilons to multiply with exponential etas. If None, all epsilons will
        be chosen. None is default.
    same_eta : bool
        Boolean of whether all RUVs from input should use the same new ETA or if one ETA
        should be created for each RUV. True is default.
    eta_names : str, list
        Custom names of new etas. Must be equal to the number epsilons or 1 if same eta.
    dv : Union[sympy.Symbol, str, int, None]
        Name or DVID of dependent variable. None for the default (first or only)

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_iiv_on_ruv(model)
    >>> model.statements.find_assignment("Y")
                  ETA_RV1
    Y = EPS₁⋅W⋅ℯ        + F

    See also
    --------
    set_power_on_ruv

    """
    list_of_eps = _format_input_list(list_of_eps)
    eps = _get_epsilons(model, list_of_eps)

    if eta_names and len(eta_names) != len(eps):
        raise ValueError(
            'The number of provided eta names must be equal to the number of epsilons.'
        )

    rvs, pset, sset = model.random_variables, list(model.parameters), model.statements

    if same_eta:
        eta = _create_eta(pset, 1, eta_names)
        rvs = rvs + eta
        eta_dict = {e: eta for e in eps}
    else:
        etas = [_create_eta(pset, i + 1, eta_names) for i in range(len(eps))]
        rvs = rvs + etas
        eta_dict = dict(zip(eps, etas))

    dv_symb = get_dv_symbol(model, dv)
    y = model.statements.find_assignment(dv_symb)

    for e in eps:
        subs_dict = {
            sympy.Symbol(e.names[0]): sympy.Symbol(e.names[0])
            * sympy.exp(sympy.Symbol(eta_dict[e].names[0]))
        }
        # FIXME: this is needed if you e.g. have Y and IPRED, with multiple DVs, how should this be handled?
        if not dv:
            sset = sset.subs(subs_dict)
        else:
            y = y.subs(subs_dict)
            sset = sset.reassign(y.symbol, y.expression)

    model = model.replace(random_variables=rvs, parameters=Parameters.create(pset), statements=sset)
    return model.update_source()


def _create_eta(pset, number, eta_names):
    omega = sympy.Symbol(f'IIV_RUV{number}')
    pset.append(Parameter(str(omega), 0.09))

    if eta_names:
        eta_name = eta_names[number - 1]
    else:
        eta_name = f'ETA_RV{number}'

    eta = NormalDistribution.create(eta_name, 'iiv', 0, omega)
    return eta
