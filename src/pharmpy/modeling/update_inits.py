"""
:meta private:
"""


def update_inits(model, force_individual_estimates=False, move_est_close_to_bounds=False):
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
    move_est_close_to_bounds : bool
        Move estimates that are close to bounds. If correlation >0.99 the correlation will
        be set to 0.9, if variance is <0.001 the variance will be set to 0.01.

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
        raise ValueError(
            'Cannot update initial parameter estimates since no modelfit results are available'
        )

    if move_est_close_to_bounds:
        param_est = _move_est_close_to_bounds(model)
    else:
        param_est = res.parameter_estimates

    model.parameters = model.parameters.set_initial_estimates(param_est)

    if model.initial_individual_estimates is not None or force_individual_estimates:
        model.initial_individual_estimates = res.individual_estimates

    return model


def _move_est_close_to_bounds(model):
    rvs, pset = model.random_variables, model.parameters
    res = model.modelfit_results
    est = res.parameter_estimates.to_dict()
    sdcorr = rvs.parameters_sdcorr(est)
    newdict = est.copy()
    for rvs, dist in rvs.distributions():
        if len(rvs) > 1:
            sigma_sym = dist.sigma
            for i in range(sigma_sym.rows):
                for j in range(sigma_sym.cols):
                    param_name = sigma_sym[i, j].name
                    if i != j:
                        if sdcorr[param_name] > 0.99:
                            name_i, name_j = sigma_sym[i, i].name, sigma_sym[j, j].name
                            # From correlation to covariance
                            corr_new = 0.9
                            sd_i, sd_j = sdcorr[name_i], sdcorr[name_j]
                            newdict[param_name] = corr_new * sd_i * sd_j
                    else:
                        if not _is_zero_fix(pset[param_name]) and est[param_name] < 0.001:
                            newdict[param_name] = 0.01
        else:
            param_name = (dist.std**2).name
            if not _is_zero_fix(pset[param_name]) and est[param_name] < 0.001:
                newdict[param_name] = 0.01
    return newdict


def _is_zero_fix(param):
    return param.init == 0 and param.fix
