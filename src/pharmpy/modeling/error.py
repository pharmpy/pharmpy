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
        'none' or 'additive'
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
    else:
        raise ValueError(f'Requested error_model {error_model} but only '
                         f'none are supported')
