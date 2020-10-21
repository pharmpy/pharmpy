import sympy

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel


def error_model(model, error_model):
    """Set a predefined error model

    Parameters
    ----------
    model
        Set error model for this model
    error_model
        'none', 'additive', 'combined' or 'proportional'
    """
    stats = model.statements
    y = model.dependent_variable_symbol
    f = model.prediction_symbol
    if error_model == 'none':
        stats.reassign(y, f)
        # FIXME: Would want a clean function here
        model.remove_unused_parameters_and_rvs()
    elif error_model == 'additive':
        ruv = model.create_symbol('RUV')
        expr = f + ruv
        stats.reassign(y, expr)
        model.remove_unused_parameters_and_rvs()

        # FIXME: Refactor to model.add_parameter
        sigma = model.create_symbol('sigma')
        sigma_par = Parameter(sigma.name, init=0.1)
        model.parameters.add(sigma_par)

        eps = sympy.stats.Normal(ruv.name, 0, sympy.sqrt(sigma))
        eps.variability_level = VariabilityLevel.RUV
        model.random_variables.add(eps)
    elif error_model == 'proportional':
        ruv = model.create_symbol('RUV')
        expr = f + f * ruv
        stats.reassign(y, expr)
        model.remove_unused_parameters_and_rvs()

        # FIXME: Refactor to model.add_parameter
        sigma = model.create_symbol('sigma')
        sigma_par = Parameter(sigma.name, init=0.1)
        model.parameters.add(sigma_par)

        eps = sympy.stats.Normal(ruv.name, 0, sympy.sqrt(sigma))
        eps.variability_level = VariabilityLevel.RUV
        model.random_variables.add(eps)
    elif error_model == 'combined':
        ruv_prop = model.create_symbol('RUV_PROP')
        ruv_add = model.create_symbol('RUV_ADD')
        expr = f + f * ruv_prop + ruv_add
        stats.reassign(y, expr)
        model.remove_unused_parameters_and_rvs()

        # FIXME: Refactor to model.add_parameter
        sigma_prop = model.create_symbol('sigma_prop')
        sigma_par1 = Parameter(sigma_prop.name, init=0.1)
        model.parameters.add(sigma_par1)
        sigma_add = model.create_symbol('sigma_add')
        sigma_par2 = Parameter(sigma_add.name, init=0.1)
        model.parameters.add(sigma_par2)

        eps_prop = sympy.stats.Normal(ruv_prop.name, 0, sympy.sqrt(sigma_prop))
        eps_prop.variability_level = VariabilityLevel.RUV
        model.random_variables.add(eps_prop)
        eps_add = sympy.stats.Normal(ruv_add.name, 0, sympy.sqrt(sigma_add))
        eps_add.variability_level = VariabilityLevel.RUV
        model.random_variables.add(eps_add)
    else:
        raise ValueError(f'Requested error_model {error_model} but only ' f'none are supported')
