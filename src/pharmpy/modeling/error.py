def error_model(model, error_model):
    """Set a predefined error model

    Parameters
    ----------
    model
        Set error model for this model
    error_model
        'none'
    """
    if error_model == 'none':
        stats = model.statements
        y = model.dependent_variable_symbol
        f = model.prediction_symbol
        stats.reassign(y, f)
        # FIXME: Would want a clean function here
        model.remove_unused_parameters_and_rvs()
    else:
        raise ValueError(f'Requested error_model {error_model} but only '
                         f'none are supported')
