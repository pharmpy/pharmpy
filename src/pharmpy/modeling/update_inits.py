"""
:meta private:
"""


class ModelfitResultsException(Exception):
    pass


def update_inits(model, force_individual_estimates=False):
    """
    Updates initial estimates from previous output. Can be forced if no initial
    individual estimates have been read.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    force_individual_estimates : bool
        Whether update of initial individual estimates should be forced.
    """
    try:
        res = model.modelfit_results
    except AttributeError:
        raise ModelfitResultsException('No modelfit results available')

    model.parameters = res.parameter_estimates

    if model.initial_individual_estimates is not None or force_individual_estimates:
        model.initial_individual_estimates = res.individual_estimates

    return model
