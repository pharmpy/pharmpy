"""
:meta private:
"""

from pharmpy.model import Model


def update_inits(model: Model, parameter_estimates, move_est_close_to_bounds=False):
    """Update initial parameter estimate for a model

    Updates initial estimates of population parameters for a model.
    If the new initial estimates are out of bounds or NaN this function will raise.

    Parameters
    ----------
    model : Model
        Pharmpy model to update initial estimates
    parameter_estimates : pd.Series
        Parameter estimates to update
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
    >>> update_inits(model, model.modelfit_results.parameter_estimates)  # doctest:+ELLIPSIS
    <...>
    >>> model.parameters.inits  # doctest:+ELLIPSIS
    {'THETA(1)': 0.00469555, 'THETA(2)': 0.984258, 'THETA(3)': 0.15892, 'OMEGA(1,1)': 0.0293508...}

    """
    if move_est_close_to_bounds:
        parameter_estimates = _move_est_close_to_bounds(model, parameter_estimates)

    model.parameters = model.parameters.set_initial_estimates(parameter_estimates)

    return model


def _move_est_close_to_bounds(model: Model, pe):
    rvs, pset = model.random_variables, model.parameters
    est = pe.to_dict()
    sdcorr = rvs.parameters_sdcorr(est)
    newdict = est.copy()
    for dist in rvs:
        rvs = dist.names
        if len(rvs) > 1:
            sigma_sym = dist.variance
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
            param_name = dist.variance.name
            if not _is_zero_fix(pset[param_name]) and est[param_name] < 0.001:
                newdict[param_name] = 0.01
    return newdict


def _is_zero_fix(param):
    return param.init == 0 and param.fix


def update_initial_individual_estimates(model, individual_estimates, force=True):
    """Update initial individual estimates for a model

    Updates initial individual estimates for a model.

    Parameters
    ----------
    model : Model
        Pharmpy model to update initial estimates
    individual_estimates : pd.DataFrame
        Individual estimates to use
    force : bool
        Set to False to only update if the model had initial individual estimates before

    Returns
    -------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, update_initial_individual_estimates
    >>> model = load_example_model("pheno")
    >>> ie = model.modelfit_results.individual_estimates
    >>> model = update_initial_individual_estimates(model, ie)
    """
    if not force and model.initial_individual_estimates is None:
        return model

    model.initial_individual_estimates = individual_estimates
    return model
