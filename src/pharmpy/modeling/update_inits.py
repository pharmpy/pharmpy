"""
:meta private:
"""

from pharmpy.random_variables import RandomVariables, VariabilityLevel


def update_inits(model, force_update_individual_estimates=False):
    try:
        model.parameters = model.modelfit_results.parameter_estimates
    except AttributeError:
        pass

    try:
        model.initial_individual_estimates = model.modelfit_results.individual_estimates
    except AttributeError:
        if force_update_individual_estimates:
            print('force')
        else:
            pass

    return model


def _get_etas(rvs):
    etas = [eta for eta in rvs if eta.variability_level == VariabilityLevel.IOV]

    return RandomVariables(etas)
