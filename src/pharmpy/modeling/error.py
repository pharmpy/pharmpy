"""
:meta private:
"""

import sympy

import pharmpy.model
import pharmpy.symbols as symbols
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariable
from pharmpy.statements import Assignment


def _preparations(model):
    stats = model.statements
    y = model.dependent_variable
    f = model.statements.find_assignment(y.name).expression
    for eps in model.random_variables.epsilons:
        f = f.subs({symbols.symbol(eps.name): 0})
    return stats, y, f


def remove_error(model):
    """Remove error model.

    Parameters
    ----------
    model : Model
        Remove error model for this model
    """
    stats, y, f = _preparations(model)
    stats.reassign(y, f)
    model.remove_unused_parameters_and_rvs()
    return model


def additive_error(model, data_trans=None):
    r"""Set an additive error model. Initial estimate for new sigma is :math:`(min(DV)/2)²`.

    The error function being applied depends on the data transformation.

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
        specified by the model.
    """
    if has_additive_error(model):
        return model
    stats, y, f = _preparations(model)
    ruv = model.create_symbol('epsilon_a')

    data_trans = pharmpy.model.canonicalize_data_transformation(model, data_trans)
    if data_trans == sympy.log(model.dependent_variable):
        expr = sympy.log(f) + ruv / f
    elif data_trans == model.dependent_variable:
        expr = f + ruv
    else:
        raise ValueError(f"Not supported data transformation {data_trans}")

    stats.reassign(y, expr)
    model.remove_unused_parameters_and_rvs()

    # FIXME: Refactor to model.add_parameter
    sigma = model.create_symbol('sigma')
    sigma_par = Parameter(sigma.name, init=_get_prop_init(model.dataset))
    model.parameters.append(sigma_par)

    eps = RandomVariable.normal(ruv.name, 'RUV', 0, sigma)
    model.random_variables.append(eps)
    return model


def _get_prop_init(dt):
    dv_min = dt.pharmpy.observations.min()
    if dv_min == 0:
        return 0.01
    else:
        return (dv_min / 2) ** 2


def proportional_error(model, data_trans=None):
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
    """
    if has_proportional_error(model):
        return model
    stats, y, f = _preparations(model)
    ruv = model.create_symbol('epsilon_p')

    data_trans = pharmpy.model.canonicalize_data_transformation(model, data_trans)
    if data_trans == sympy.log(model.dependent_variable):
        expr = sympy.log(f) + ruv
    elif data_trans == model.dependent_variable:
        expr = f + f * ruv
    else:
        raise ValueError(f"Not supported data transformation {data_trans}")

    stats.reassign(y, expr)
    model.remove_unused_parameters_and_rvs()

    # FIXME: Refactor to model.add_parameter
    sigma = model.create_symbol('sigma')
    sigma_par = Parameter(sigma.name, init=0.09)
    model.parameters.append(sigma_par)

    eps = RandomVariable.normal(ruv.name, 'RUV', 0, sigma)
    model.random_variables.append(eps)
    return model


def combined_error(model, data_trans=None):
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
    """
    if has_combined_error(model):
        return model
    stats, y, f = _preparations(model)
    ruv_prop = model.create_symbol('epsilon_p')
    ruv_add = model.create_symbol('epsilon_a')

    data_trans = pharmpy.model.canonicalize_data_transformation(model, data_trans)
    if data_trans == sympy.log(model.dependent_variable):
        expr = sympy.log(f) + ruv_prop + ruv_add / f
    elif data_trans == model.dependent_variable:
        expr = f + f * ruv_prop + ruv_add
    else:
        raise ValueError(f"Not supported data transformation {data_trans}")

    stats.reassign(y, expr)
    model.remove_unused_parameters_and_rvs()

    # FIXME: Refactor to model.add_parameter
    sigma_prop = model.create_symbol('sigma_prop')
    sigma_par1 = Parameter(sigma_prop.name, init=0.09)
    model.parameters.append(sigma_par1)
    sigma_add = model.create_symbol('sigma_add')
    sigma_par2 = Parameter(sigma_add.name, init=_get_prop_init(model.dataset))
    model.parameters.append(sigma_par2)

    eps_prop = RandomVariable.normal(ruv_prop.name, 'RUV', 0, sigma_prop)
    model.random_variables.append(eps_prop)
    eps_add = RandomVariable.normal(ruv_add.name, 'RUV', 0, sigma_add)
    model.random_variables.append(eps_add)
    return model


def has_additive_error(model):
    """Check if a model has an additive error model

    Parameters
    ----------
    model : Model
        The model to check
    """
    y = model.dependent_variable
    expr = model.statements.full_expression_after_odes(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {
        symbols.symbol(rv.name) for rv in rvs if symbols.symbol(rv.name) in expr.free_symbols
    }
    if len(rvs_in_y) != 1:
        return False
    eps = rvs_in_y.pop()
    return eps not in (expr - eps).simplify().free_symbols


def has_proportional_error(model):
    """Check if a model has a proportional error model

    Parameters
    ----------
    model : Model
        The model to check
    """
    y = model.dependent_variable
    expr = model.statements.full_expression_after_odes(y)
    rvs = model.random_variables.epsilons
    rvs_in_y = {
        symbols.symbol(rv.name) for rv in rvs if symbols.symbol(rv.name) in expr.free_symbols
    }
    if len(rvs_in_y) != 1:
        return False
    eps = rvs_in_y.pop()
    return eps not in (expr / (1 + eps)).simplify().free_symbols


def has_combined_error(model):
    """Check if a model has a combined additive and proportinal error model

    Parameters
    ----------
    model : Model
        The model to check
    """
    y = model.dependent_variable
    expr = model.statements.full_expression_after_odes(y)
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


def theta_as_stdev(model):
    """Use thetas to estimate standard deviation of error"""
    rvs = model.random_variables.epsilons
    for eps in rvs:
        sigmas = eps.parameter_names
        if len(sigmas) > 1:
            raise ValueError('theta_as_stdev only supports non-correlated sigmas')
        sigma = sigmas[0]
        param = model.parameters[sigma]
        param.fix = True
        theta_init = param.init ** 0.5
        param.init = 1
        theta = Parameter(f'SD_{eps.name}', theta_init, lower=0)
        model.parameters.append(theta)
        symb = sympy.Symbol(eps.name)
        model.statements.subs({symb: theta.symbol * symb})
    return model


def set_weighted_error_model(model):
    """Encode error model with one epsilon and W as weight"""
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
            ssum += remaining ** 2
            for symb in remaining.free_symbols:
                q &= sympy.Q.positive(symb)
    w = sympy.sqrt(ssum)
    w = sympy.refine(w, q)

    for i, s in enumerate(stats):
        if isinstance(s, Assignment) and s.symbol == y:
            stats.insert(i, Assignment('W', w))
            break

    stats.reassign(y, f + sympy.Symbol('W') * sympy.Symbol(epsilons[0].name))
    model.remove_unused_parameters_and_rvs()
    return model


def set_dtbs_error(model):
    """Dynamic transform both sides"""
    theta_as_stdev(model)
    set_weighted_error_model(model)
    stats, y, f = _preparations(model)
    tbs_lambda = Parameter('tbs_lambda', 1)
    tbs_zeta = Parameter('tbs_zeta', 0.001)
    model.parameters.append(tbs_lambda)
    model.parameters.append(tbs_zeta)
    lam = tbs_lambda.symbol
    zeta = tbs_zeta.symbol

    for i, s in enumerate(stats):
        if isinstance(s, Assignment) and s.symbol == sympy.Symbol('W'):
            break

    stats.insert(i + 1, Assignment('W', (f ** zeta) * sympy.Symbol('W')))
    ipred = sympy.Piecewise(
        ((f ** lam - 1) / lam, sympy.And(sympy.Ne(lam, 0), sympy.Ne(f, 0))),
        (sympy.log(f), sympy.And(sympy.Eq(lam, 0), sympy.Ne(f, 0))),
        (-1 / lam, sympy.And(sympy.Eq(lam, 0), sympy.Eq(f, 0))),
        (-1000000000, True),
    )
    stats.insert(i + 2, Assignment('IPRED', ipred))
    yexpr = stats.find_assignment(y.name)
    yexpr.subs({f: sympy.Symbol('IPRED')})

    obs = sympy.Piecewise(
        (sympy.log(y), sympy.Eq(lam, 0)), ((y ** lam - 1) / lam, sympy.Ne(lam, 0))
    )
    model.observation_transformation = obs

    return model
