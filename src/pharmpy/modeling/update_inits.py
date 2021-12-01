"""
:meta private:
"""


def update_inits(model, force_individual_estimates=False):
    """Update initial parameter estimate for a model

    Updates initial estimates of population parameters for a model from
    its modelfit_results. If the model has used initial estimates for
    individual estimates these will also be updated. If the new initial estimates
    are out of bounds or NaN this function will raise.

    Parameters
    ----------
    model : Model
        Pharmpy model to update initial estimates
    force_individual_estimates : bool
        Update initial individual estimates even if model din't use them previously.

    Returns
    -------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, update_inits
    >>> model = load_example_model("pheno")   # This model was previously fitted to its data
    >>> model.parameters.inits  # doctest:+ELLIPSIS
    {'THETA(1)': 0.00469307, 'THETA(2)': 1.00916, 'THETA(3)': 0.1, 'OMEGA(1,1)': 0.0309626...}
    >>> update_inits(model)  # doctest:+ELLIPSIS
    <...>
    >>> model.parameters.inits  # doctest:+ELLIPSIS
    {'THETA(1)': 0.00469555, 'THETA(2)': 0.984258, 'THETA(3)': 0.15892, 'OMEGA(1,1)': 0.0293508...}

    """
    if isinstance(model, list) and len(model) == 1:
        model = model[0]

    res = model.modelfit_results
    if not res:
        raise ValueError('No modelfit results available')

    try:
        model.parameters = res.parameter_estimates
    except ValueError:
        raise ValueError('One or more parameter estimates are NaN')

    if model.initial_individual_estimates is not None or force_individual_estimates:
        model.initial_individual_estimates = res.individual_estimates

    return model
