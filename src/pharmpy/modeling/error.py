"""
:meta private:
"""

import sympy

import pharmpy.symbols as symbols
from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel


def _preparations(model):
    stats = model.statements
    y = model.dependent_variable_symbol
    f = model.statements.find_assignment(y.name).expression
    for eps in model.random_variables.ruv_rvs:
        f = f.subs({symbols.symbol(eps.name): 0})
    return stats, y, f


def remove_error(model):
    """Remove error model.

    Parameters
    ----------
    model
        Remove error model for this model
    """
    stats, y, f = _preparations(model)
    stats.reassign(y, f)
    model.remove_unused_parameters_and_rvs()
    return model


def additive_error(model):
    """Set an additive error model. Initial estimate for new sigma is :math:`(min(DV)/2)²`.

    Parameters
    ----------
    model
        Set error model for this model
    """
    if has_additive_error(model):
        return model
    stats, y, f = _preparations(model)
    ruv = model.create_symbol('RUV')
    expr = f + ruv
    stats.reassign(y, expr)
    model.remove_unused_parameters_and_rvs()

    # FIXME: Refactor to model.add_parameter
    sigma = model.create_symbol('sigma')
    sigma_par = Parameter(sigma.name, init=_get_prop_init(model.dataset))
    model.parameters.add(sigma_par)

    eps = sympy.stats.Normal(ruv.name, 0, sympy.sqrt(sigma))
    eps.variability_level = VariabilityLevel.RUV
    model.random_variables.add(eps)
    return model


def _get_prop_init(dt):
    dv_min = dt.pharmpy.observations.min()
    if dv_min == 0:
        return 0.01
    else:
        return (dv_min / 2) ** 2


def proportional_error(model):
    """Set a proportional error model. Initial estimate for new sigma is 0.09.

    Parameters
    ----------
    model
        Set error model for this model
    """
    if has_proportional_error(model):
        return model
    stats, y, f = _preparations(model)
    ruv = model.create_symbol('RUV')
    expr = f + f * ruv
    stats.reassign(y, expr)
    model.remove_unused_parameters_and_rvs()

    # FIXME: Refactor to model.add_parameter
    sigma = model.create_symbol('sigma')
    sigma_par = Parameter(sigma.name, init=0.09)
    model.parameters.add(sigma_par)

    eps = sympy.stats.Normal(ruv.name, 0, sympy.sqrt(sigma))
    eps.variability_level = VariabilityLevel.RUV
    model.random_variables.add(eps)
    return model


def combined_error(model):
    """Set a combined error model. Initial estimates for new sigmas are :math:`(min(DV)/2)²` for
    proportional and 0.09 for additive.

    Parameters
    ----------
    model
        Set error model for this model
    """
    if has_combined_error(model):
        return model
    stats, y, f = _preparations(model)
    ruv_prop = model.create_symbol('RUV_PROP')
    ruv_add = model.create_symbol('RUV_ADD')
    expr = f + f * ruv_prop + ruv_add
    stats.reassign(y, expr)
    model.remove_unused_parameters_and_rvs()

    # FIXME: Refactor to model.add_parameter
    sigma_prop = model.create_symbol('sigma_prop')
    sigma_par1 = Parameter(sigma_prop.name, init=0.09)
    model.parameters.add(sigma_par1)
    sigma_add = model.create_symbol('sigma_add')
    sigma_par2 = Parameter(sigma_add.name, init=_get_prop_init(model.dataset))
    model.parameters.add(sigma_par2)

    eps_prop = sympy.stats.Normal(ruv_prop.name, 0, sympy.sqrt(sigma_prop))
    eps_prop.variability_level = VariabilityLevel.RUV
    model.random_variables.add(eps_prop)
    eps_add = sympy.stats.Normal(ruv_add.name, 0, sympy.sqrt(sigma_add))
    eps_add.variability_level = VariabilityLevel.RUV
    model.random_variables.add(eps_add)
    return model


def has_additive_error(model):
    """Check if a model has an additive error model

    Parameters
    ----------
    model
        The model to check
    """
    y = model.dependent_variable_symbol
    expr = model.statements.full_expression_after_odes(y)
    rvs = model.random_variables.ruv_rvs
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
    model
        The model to check
    """
    y = model.dependent_variable_symbol
    expr = model.statements.full_expression_after_odes(y)
    rvs = model.random_variables.ruv_rvs
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
    model
        The model to check
    """
    y = model.dependent_variable_symbol
    expr = model.statements.full_expression_after_odes(y)
    rvs = model.random_variables.ruv_rvs
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
