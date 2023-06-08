from typing import Dict, List, Optional, Union

from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Parameter, Parameters


def get_thetas(model: Model):
    """Get all thetas (structural parameters) of a model

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Parameters
        A copy of all theta parameters

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_thetas(model)
                value  lower      upper    fix
    PTVCL    0.004693   0.00  1000000.0  False
    PTVV     1.009160   0.00  1000000.0  False
    THETA_3  0.100000  -0.99  1000000.0  False

    See also
    --------
    get_omegas : Get omega parameters
    get_sigmas : Get sigma parameters
    """
    rvs_fs = model.random_variables.free_symbols
    thetas = [p for p in model.parameters if p.symbol not in rvs_fs]
    return Parameters(tuple(thetas))


def get_omegas(model: Model):
    """Get all omegas (variability parameters) of a model

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Parameters
        A copy of all omega parameters

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_omegas(model)
             value  lower upper    fix
    IVCL  0.030963    0.0     ∞  False
    IVV   0.031128    0.0     ∞  False

    See also
    --------
    get_thetas : Get theta parameters
    get_sigmas : Get sigma parameters
    """
    omegas = [p for p in model.parameters if p.symbol in model.random_variables.etas.free_symbols]
    return Parameters(tuple(omegas))


def get_sigmas(model: Model):
    """Get all sigmas (residual error variability parameters) of a model

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Parameters
        A copy of all sigma parameters

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_sigmas(model)
                  value  lower upper    fix
    SIGMA_1_1  0.013241    0.0     ∞  False

    See also
    --------
    get_thetas : Get theta parameters
    get_omegas : Get omega parameters
    """
    sigmas = [
        p for p in model.parameters if p.symbol in model.random_variables.epsilons.free_symbols
    ]
    return Parameters(tuple(sigmas))


def set_initial_estimates(model: Model, inits: Dict[str, float]):
    """Set initial estimates

    Parameters
    ----------
    model : Model
        Pharmpy model
    inits : dict
        A dictionary of parameter init for parameters to change

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_initial_estimates
    >>> model = load_example_model("pheno")
    >>> model = set_initial_estimates(model, {'PTVCL': 2})
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 2, lower=0.0, upper=1000000.0, fix=False)

    See also
    --------
    fix_parameters_to : Fixing and setting parameter initial estimates in the same function
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
    """
    new = model.parameters.set_initial_estimates(inits)
    model = model.replace(parameters=new)
    return model.update_source()


def update_inits(
    model: Model, parameter_estimates: pd.Series, move_est_close_to_bounds: bool = False
):
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
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, update_inits
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> model.parameters.inits  # doctest:+ELLIPSIS
    {'PTVCL': 0.00469307, 'PTVV': 1.00916, 'THETA_3': 0.1, 'IVCL': 0.0309626, 'IVV': 0.031128, 'SIGMA_1_1': 0.013241}
    >>> model = update_inits(model, results.parameter_estimates)
    >>> model.parameters.inits  # doctest:+ELLIPSIS
    {'PTVCL': 0.00469555, 'PTVV': 0.984258, 'THETA_3': 0.15892, 'IVCL': 0.0293508, 'IVV': 0.027906, ...}

    """
    # FIXME: can be combined with set_initial_estimates
    if move_est_close_to_bounds:
        parameter_estimates = _move_est_close_to_bounds(model, parameter_estimates)

    model = model.replace(parameters=model.parameters.set_initial_estimates(parameter_estimates))

    return model.update_source()


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


def set_upper_bounds(model: Model, bounds: Dict[str, float]):
    """Set parameter upper bounds

    Parameters
    ----------
    model : Model
        Pharmpy model
    bounds : dict
        A dictionary of parameter bounds for parameters to change

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_upper_bounds
    >>> model = load_example_model("pheno")
    >>> model = set_upper_bounds(model, {'PTVCL': 10})
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=0.0, upper=10, fix=False)

    See also
    --------
    set_lower_bounds : Set parameter lower bounds
    unconstrain_parameters : Remove all constraints of parameters
    """
    new = []
    for p in model.parameters:
        if p.name in bounds:
            newparam = Parameter(
                name=p.name, init=p.init, lower=p.lower, upper=bounds[p.name], fix=p.fix
            )
        else:
            newparam = p
        new.append(newparam)
    model = model.replace(parameters=Parameters(tuple(new)))
    return model.update_source()


def set_lower_bounds(model: Model, bounds: Dict[str, float]):
    """Set parameter lower bounds

    Parameters
    ----------
    model : Model
        Pharmpy model
    bounds : dict
        A dictionary of parameter bounds for parameters to change

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_lower_bounds
    >>> model = load_example_model("pheno")
    >>> model = set_lower_bounds(model, {'PTVCL': -10})
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=-10, upper=1000000.0, fix=False)

    See also
    --------
    set_upper_bounds : Set parameter upper bounds
    unconstrain_parameters : Remove all constraints of parameters
    """
    new = []
    for p in model.parameters:
        if p.name in bounds:
            newparam = Parameter(
                name=p.name, init=p.init, lower=bounds[p.name], upper=p.upper, fix=p.fix
            )
        else:
            newparam = p
        new.append(newparam)
    model = model.replace(parameters=Parameters(tuple(new)))
    model = model.update_source()
    return model


def fix_parameters(model: Model, parameter_names: Union[List[str], str]):
    """Fix parameters

    Fix all listed parameters

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list or str
        one parameter name or a list of parameter names

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import fix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> model = fix_parameters(model, 'PTVCL')
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=0.0, upper=1000000.0, fix=True)

    See also
    --------
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    fix_parameters_to : Fixing and setting parameter initial estimates in the same function
    unfix_paramaters : Unfixing parameters
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function

    """
    if isinstance(parameter_names, str):
        parameter_names = [parameter_names]
    params = model.parameters
    new = []
    for p in params:
        if p.name in parameter_names:
            new_param = p.replace(fix=True)
        else:
            new_param = p
        new.append(new_param)
    model = model.replace(parameters=Parameters(tuple(new)))
    return model.update_source()


def unfix_parameters(model: Model, parameter_names: Union[List[str], str]):
    """Unfix parameters

    Unfix all listed parameters

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list or str
        one parameter name or a list of parameter names

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters, unfix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model = fix_parameters(model, ['PTVCL', 'PTVV', 'THETA_3'])
    >>> model.parameters.fix    # doctest: +ELLIPSIS
    {'PTVCL': True, 'PTVV': True, 'THETA_3': True, ...}
    >>> model = unfix_parameters(model, 'PTVCL')
    >>> model.parameters.fix        # doctest: +ELLIPSIS
    {'PTVCL': False, 'PTVV': True, 'THETA_3': True, ...}

    See also
    --------
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function
    fix_parameters : Fix parameters
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    fix_parameters_to : Fixing and setting parameter initial estimates in the same function
    unconstrain_parameters : Remove all constraints of parameters

    """
    if isinstance(parameter_names, str):
        parameter_names = [parameter_names]
    params = model.parameters
    new = []
    for p in params:
        if p.name in parameter_names:
            new_param = p.replace(fix=False)
        else:
            new_param = p
        new.append(new_param)
    model = model.replace(parameters=Parameters(tuple(new)))
    model = model.update_source()
    return model


def fix_parameters_to(model: Model, inits: Dict[str, float]):
    """Fix parameters to

    Fix all listed parameters to specified value/values

    Parameters
    ----------
    model : Model
        Pharmpy model
    inits : dict
        Inits for all parameters to fix and set init

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters_to, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> model = fix_parameters_to(model, {'PTVCL': 0.5})
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.5, lower=0.0, upper=1000000.0, fix=True)

    See also
    --------
    fix_parameters : Fix parameters
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    unfix_paramaters : Unfixing parameters
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function

    """
    model = fix_parameters(model, inits.keys())
    model = set_initial_estimates(model, inits)
    return model


def unfix_parameters_to(model: Model, inits: Dict[str, float]):
    """Unfix parameters to

    Unfix all listed parameters to specified value/values

    Parameters
    ----------
    model : Model
        Pharmpy model
    inits : dict
        Inits for all parameters to unfix and change init

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters, unfix_parameters_to, load_example_model
    >>> model = load_example_model("pheno")
    >>> model = fix_parameters(model, ['PTVCL', 'PTVV', 'THETA_3'])
    >>> model.parameters.fix    # doctest: +ELLIPSIS
    {'PTVCL': True, 'PTVV': True, 'THETA_3': True, 'IVCL': False, 'IVV': False, 'SIGMA_1_1': False}
    >>> model = unfix_parameters_to(model, {'PTVCL': 0.5})
    >>> model.parameters.fix        # doctest: +ELLIPSIS
    {'PTVCL': False, 'PTVV': True, 'THETA_3': True, ...}
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.5, lower=0.0, upper=1000000.0, fix=False)

    Returns
    -------
    Model
        Pharmpy model object

    See also
    --------
    fix_parameters : Fix parameters
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    unfix_paramaters : Unfixing parameters
    fix_paramaters_to : Fixing parameters and setting a new initial estimate in the same
        function
    """
    model = unfix_parameters(model, inits.keys())
    model = set_initial_estimates(model, inits)
    return model


def fix_or_unfix_parameters(model: Model, parameters: Dict[str, bool]):
    """Fix or unfix parameters

    Set fixedness of parameters to specified values

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameters : dict
        Set fix/unfix for these parameters

    Examples
    --------
    >>> from pharmpy.modeling import fix_or_unfix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> model = fix_or_unfix_parameters(model, {'PTVCL': True})
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=0.0, upper=1000000.0, fix=True)

    Returns
    -------
    Model
        Pharmpy model object

    See also
    --------
    fix_parameters : Fix parameters
    unfix_paramaters : Unfixing parameters
    fix_paramaters_to : Fixing parameters and setting a new initial estimate in the same
        function
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function
    """
    params = model.parameters
    new = []
    for p in params:
        if p.name in parameters:
            new_param = p.replace(fix=parameters[p.name])
        else:
            new_param = p
        new.append(new_param)
    model = model.replace(parameters=Parameters(tuple(new)))
    model = model.update_source()
    return model


def unconstrain_parameters(model: Model, parameter_names: List[str]):
    """Remove all constraints from parameters

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list
        Remove all constraints for the listed parameters

    Examples
    --------
    >>> from pharmpy.modeling import unconstrain_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> model = unconstrain_parameters(model, ['PTVCL'])
    >>> model.parameters['PTVCL']
    Parameter("PTVCL", 0.00469307, lower=-∞, upper=∞, fix=False)

    Returns
    -------
    Model
        Pharmpy model object

    See also
    --------
    set_lower_bounds : Set parameter lower bounds
    set_upper_bounds : Set parameter upper bounds
    unfix_parameters : Unfix parameters
    """
    if isinstance(parameter_names, str):
        parameter_names = [parameter_names]
    new = []
    for p in model.parameters:
        if p.name in parameter_names:
            newparam = Parameter(name=p.name, init=p.init)
        else:
            newparam = p
        new.append(newparam)
    model = model.replace(parameters=Parameters(tuple(new)))
    model = model.update_source()
    return model


def add_population_parameter(
    model: Model,
    name: str,
    init: float,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    fix: bool = False,
):
    """Add a new population parameter to the model

    Parameters
    ----------
    model : Model
        Pharmpy model
    name : str
        Name of the new parameter
    init : float
        Initial estimate of the new parameter
    lower : float
        Lower bound of the new parameter
    upper : float
        Upper bound of the new parameter
    fix : bool
        Should the new parameter be fixed?

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import add_population_parameter, load_example_model
    >>> model = load_example_model("pheno")
    >>> model = add_population_parameter(model, 'POP_KA', 2)
    >>> model.parameters
                  value lower      upper    fix
    PTVCL      0.004693   0.0  1000000.0  False
    PTVV       1.009160   0.0  1000000.0  False
    THETA_3    0.100000 -0.99  1000000.0  False
    IVCL       0.030963   0.0          ∞  False
    IVV        0.031128   0.0          ∞  False
    SIGMA_1_1  0.013241   0.0          ∞  False
    POP_KA     2.000000    -∞          ∞  False
    """

    param = Parameter.create(name, init, lower=lower, upper=upper, fix=fix)
    params = model.parameters + param
    model = model.replace(parameters=params)
    return model.update_source()
