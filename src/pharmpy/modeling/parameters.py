from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Literal, Optional, Union

from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.model import (
    Assignment,
    JointNormalDistribution,
    Model,
    Parameter,
    Parameters,
    RandomVariables,
)


def get_thetas(model: Model):
    """Get all thetas (structural parameters) of a model

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Parameters
        A copy of all theta parameters

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_thetas(model)
                value  lower upper    fix
    POP_CL   0.004693   0.00     ∞  False
    POP_VC   1.009160   0.00     ∞  False
    COVAPGR  0.100000  -0.99     ∞  False

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
        Pharmpy model

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
    IIV_CL  0.030963    0.0     ∞  False
    IIV_VC  0.031128    0.0     ∞  False

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
        Pharmpy model

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
    SIGMA  0.013086    0.0     ∞  False

    See also
    --------
    get_thetas : Get theta parameters
    get_omegas : Get omega parameters
    """
    sigmas = [
        p for p in model.parameters if p.symbol in model.random_variables.epsilons.free_symbols
    ]
    return Parameters(tuple(sigmas))


def set_initial_estimates(
    model: Model,
    inits: Mapping[str, float],
    move_est_close_to_bounds: bool = False,
    strict: bool = True,
):
    """Update initial parameter estimate for a model

    Updates initial estimates of population parameters for a model.
    If the new initial estimates are out of bounds or NaN this function will raise.

    Parameters
    ----------
    model : Model
        Pharmpy model
    inits : pd.Series or dict
        Initial parameter estimates to update
    move_est_close_to_bounds : bool
        Move estimates that are close to bounds. If correlation >0.99 the correlation will
        be set to 0.9, if variance is <0.001 the variance will be set to 0.01.
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True
        Setting strict to False will also disregard any initial estimate being NaN and keep
        the original value for these parameters.

    Returns
    -------
    Model
        Updated Pharmpy model

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, set_initial_estimates
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> model.parameters.inits  # doctest:+ELLIPSIS
    {'POP_CL': 0.00469307, 'POP_VC': 1.00916, 'COVAPGR': 0.1, 'IIV_CL': 0.0309626, 'IIV_VC': ...}
    >>> model = set_initial_estimates(model, results.parameter_estimates)
    >>> model.parameters.inits  # doctest:+ELLIPSIS
    {'POP_CL': 0.00469555, 'POP_VC': 0.984258, 'COVAPGR': 0.15892, 'IIV_CL': 0.0293508, ...}
    >>> model = load_example_model("pheno")
    >>> model = set_initial_estimates(model, {'POP_CL': 2.0})
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 2.0, lower=0.0, upper=∞, fix=False)

    See also
    --------
    fix_parameters_to : Fixing and setting parameter initial estimates in the same function
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same

    """
    if strict:
        _check_input_params(model, inits.keys())
    else:
        inits = _remove_nan(inits)
    if move_est_close_to_bounds:
        inits = _move_est_close_to_bounds(model, inits)

    new = model.parameters.set_initial_estimates(inits)
    model = model.replace(parameters=new)

    return model.update_source()


def _remove_nan(inits):
    return {key: value for key, value in inits.items() if not np.isnan(value)}


def _get_nonfixed_rvs(model):
    fixed_omegas = get_omegas(model).fixed.names
    rvs = model.random_variables
    nonfixed_rvs = [rv for rv in rvs if str(list(rv.variance.free_symbols)[0]) not in fixed_omegas]
    return RandomVariables.create(nonfixed_rvs)


def _move_est_close_to_bounds(model: Model, est_new):
    rvs, pset = _get_nonfixed_rvs(model), model.parameters

    newdict = _move_omega_ests(rvs, pset, est_new)
    newdict = _move_theta_est(rvs, pset, est_new, newdict)

    return newdict


def _move_omega_ests(rvs, pset, est_new):
    parameter_estimates = pset.inits
    parameter_estimates.update(est_new)  # Need all numerical values for sdcorr
    sdcorr = rvs.parameters_sdcorr(parameter_estimates)
    newdict = est_new.copy()
    for dist in rvs:
        if isinstance(dist, JointNormalDistribution):
            sigma_sym = dist.variance
            for i in range(sigma_sym.rows):
                for j in range(sigma_sym.cols):
                    param_name = sigma_sym[i, j].name
                    if param_name not in est_new.keys():
                        continue
                    if i != j:
                        if -0.99 < sdcorr[param_name] < 0.99:
                            init = est_new[param_name]
                        else:
                            name_i, name_j = sigma_sym[i, i].name, sigma_sym[j, j].name
                            # From correlation to covariance
                            corr_new = 0.9 if sdcorr[param_name] > 0.99 else -0.9
                            sd_i, sd_j = sdcorr[name_i], sdcorr[name_j]
                            init = corr_new * sd_i * sd_j
                    else:
                        init = _get_diag_init(pset[param_name], est_new[param_name])
                    newdict[param_name] = init
        else:
            param_name = dist.variance.name
            if param_name not in est_new.keys():
                continue
            newdict[param_name] = _get_diag_init(pset[param_name], est_new[param_name])
    return newdict


def _move_theta_est(rvs, pset, est_new, newdict):
    for param, init in est_new.items():
        if param in rvs.parameter_names:
            continue
        if param not in pset.names:  # Has been checked via strict option
            continue

        upper_limit = 0.95 * pset[param].upper
        lower_limit = 0.95 * pset[param].lower

        if init == 0:
            init = 0.01

        init = max(init, lower_limit)
        init = min(init, upper_limit)

        if init < lower_limit or init > upper_limit:
            init = (upper_limit - lower_limit) / 2

        newdict[param] = init
    return newdict


def _is_zero_fix(param):
    return param.init == 0 and param.fix


def _get_diag_init(param, init):
    if not param.fix and not _is_zero_fix(param) and init < 0.001:
        return 0.01
    else:
        return init


def set_upper_bounds(model: Model, bounds: Mapping[str, float], strict: bool = True):
    """Set parameter upper bounds

    Parameters
    ----------
    model : Model
        Pharmpy model
    bounds : dict
        A dictionary of parameter bounds for parameters to change
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_upper_bounds
    >>> model = load_example_model("pheno")
    >>> model = set_upper_bounds(model, {'POP_CL': 10})
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=0.0, upper=10, fix=False)

    See also
    --------
    set_lower_bounds : Set parameter lower bounds
    unconstrain_parameters : Remove all constraints of parameters
    """
    if strict:
        _check_input_params(model, bounds.keys())
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


def set_lower_bounds(model: Model, bounds: Mapping[str, float], strict: bool = True):
    """Set parameter lower bounds

    Parameters
    ----------
    model : Model
        Pharmpy model
    bounds : dict
        A dictionary of parameter bounds for parameters to change
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_lower_bounds
    >>> model = load_example_model("pheno")
    >>> model = set_lower_bounds(model, {'POP_CL': -10})
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=-10, upper=∞, fix=False)

    See also
    --------
    set_upper_bounds : Set parameter upper bounds
    unconstrain_parameters : Remove all constraints of parameters
    """
    if strict:
        _check_input_params(model, bounds.keys())
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


def fix_parameters(model: Model, parameter_names: Union[Iterable[str], str], strict: bool = True):
    """Fix parameters

    Fix all listed parameters

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list or str
        one parameter name or a list of parameter names
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Returns
    -------
    Model
        Updated Pharmpy model

    Example
    -------
    >>> from pharmpy.modeling import fix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=0.0, upper=∞, fix=False)
    >>> model = fix_parameters(model, 'POP_CL')
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=0.0, upper=∞, fix=True)

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
    if strict:
        _check_input_params(model, parameter_names)
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


def _check_input_params(model, parameter_names):
    params_not_in_model = [
        p_name for p_name in parameter_names if p_name not in model.parameters.names
    ]
    if params_not_in_model:
        raise ValueError(f'Parameters not found in model: {params_not_in_model}')


def unfix_parameters(model: Model, parameter_names: Union[Iterable[str], str], strict: bool = True):
    """Unfix parameters

    Unfix all listed parameters

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list or str
        one parameter name or a list of parameter names
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters, unfix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model = fix_parameters(model, ['POP_CL', 'POP_VC'])
    >>> model.parameters.fix    # doctest: +ELLIPSIS
    {'POP_CL': True, 'POP_VC': True, ...}
    >>> model = unfix_parameters(model, 'POP_CL')
    >>> model.parameters.fix        # doctest: +ELLIPSIS
    {'POP_CL': False, 'POP_VC': True, ...}

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
    if strict:
        _check_input_params(model, parameter_names)
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


def fix_parameters_to(model: Model, inits: Mapping[str, float], strict: bool = True):
    """Fix parameters to

    Fix all listed parameters to specified value/values

    Parameters
    ----------
    model : Model
        Pharmpy model
    inits : dict
        Inits for all parameters to fix and set init
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters_to, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=0.0, upper=∞, fix=False)
    >>> model = fix_parameters_to(model, {'POP_CL': 0.5})
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.5, lower=0.0, upper=∞, fix=True)

    See also
    --------
    fix_parameters : Fix parameters
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    unfix_paramaters : Unfixing parameters
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function

    """
    model = fix_parameters(model, inits.keys(), strict=strict)
    model = set_initial_estimates(model, inits, strict=strict)
    return model


def unfix_parameters_to(model: Model, inits: Mapping[str, float], strict: bool = True):
    """Unfix parameters to

    Unfix all listed parameters to specified value/values

    Parameters
    ----------
    model : Model
        Pharmpy model
    inits : dict
        Inits for all parameters to unfix and change init
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters, unfix_parameters_to, load_example_model
    >>> model = load_example_model("pheno")
    >>> model = fix_parameters(model, ['POP_CL', 'POP_VC'])
    >>> model.parameters.fix    # doctest: +ELLIPSIS
    {'POP_CL': True, 'POP_VC': True, 'COVAPGR': False, 'IIV_CL': False, 'IIV_VC': False, 'SIGMA': False}
    >>> model = unfix_parameters_to(model, {'POP_CL': 0.5})
    >>> model.parameters.fix        # doctest: +ELLIPSIS
    {'POP_CL': False, 'POP_VC': True, 'COVAPGR': False, ...}
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.5, lower=0.0, upper=∞, fix=False)

    Returns
    -------
    Model
        Updated Pharmpy model

    See also
    --------
    fix_parameters : Fix parameters
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    unfix_paramaters : Unfixing parameters
    fix_paramaters_to : Fixing parameters and setting a new initial estimate in the same
        function
    """
    model = unfix_parameters(model, inits.keys(), strict=strict)
    model = set_initial_estimates(model, inits, strict=strict)
    return model


def fix_or_unfix_parameters(model: Model, parameters: Mapping[str, bool], strict: bool = True):
    """Fix or unfix parameters

    Set fixedness of parameters to specified values

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameters : dict
        Set fix/unfix for these parameters
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Examples
    --------
    >>> from pharmpy.modeling import fix_or_unfix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=0.0, upper=∞, fix=False)
    >>> model = fix_or_unfix_parameters(model, {'POP_CL': True})
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=0.0, upper=∞, fix=True)

    Returns
    -------
    Model
        Updated Pharmpy model

    See also
    --------
    fix_parameters : Fix parameters
    unfix_paramaters : Unfixing parameters
    fix_paramaters_to : Fixing parameters and setting a new initial estimate in the same
        function
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function
    """
    if strict:
        _check_input_params(model, parameters.keys())
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


def unconstrain_parameters(model: Model, parameter_names: Iterable[str], strict: bool = True):
    """Remove all constraints from parameters

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list
        Remove all constraints for the listed parameters
    strict : bool
        Whether all parameters in input need to exist in the model. Default is True

    Examples
    --------
    >>> from pharmpy.modeling import unconstrain_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=0.0, upper=∞, fix=False)
    >>> model = unconstrain_parameters(model, ['POP_CL'])
    >>> model.parameters['POP_CL']
    Parameter("POP_CL", 0.00469307, lower=-∞, upper=∞, fix=False)

    Returns
    -------
    Model
        Updated Pharmpy model

    See also
    --------
    set_lower_bounds : Set parameter lower bounds
    set_upper_bounds : Set parameter upper bounds
    unfix_parameters : Unfix parameters
    """
    if isinstance(parameter_names, str):
        parameter_names = [parameter_names]
    if strict:
        _check_input_params(model, parameter_names)
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
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import add_population_parameter, load_example_model
    >>> model = load_example_model("pheno")
    >>> model = add_population_parameter(model, 'POP_KA', 2)
    >>> model.parameters
                value lower upper    fix
    POP_CL   0.004693   0.0     ∞  False
    POP_VC   1.009160   0.0     ∞  False
    COVAPGR  0.100000 -0.99     ∞  False
    POP_KA   2.000000    -∞     ∞  False
    IIV_CL   0.030963   0.0     ∞  False
    IIV_VC   0.031128   0.0     ∞  False
    SIGMA    0.013086   0.0     ∞  False
    """

    param = Parameter.create(name, init, lower=lower, upper=upper, fix=fix)
    params = model.parameters + param
    model = model.replace(parameters=params)
    return model.update_source()


def replace_fixed_thetas(model: Model):
    """Replace all fixed thetas with constants in the model statements

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Updated Pharmpy model
    """

    keep = []
    new_assignments = []

    for p in model.parameters:
        if p.fix:
            ass = Assignment(p.symbol, Expr.float(p.init))
            new_assignments.append(ass)
        else:
            keep.append(p)

    model = model.replace(
        parameters=Parameters(tuple(keep)), statements=new_assignments + model.statements
    )
    model = model.update_source()
    return model


def map_eta_parameters(
    model: Model,
    keys: Literal['parameters', 'omegas', 'etas'],
    values: Literal['parameters', 'omegas', 'etas'],
    level: Literal['iiv', 'iov'] = 'iiv',
) -> dict[str, list[str]]:
    """Create a map with the connections from and to individual parameters, omegas and/or etas

    The mapping will always be one to many.

    Parameters
    ----------
    model : Model
        Pharmpy model
    keys : str
        What to map from. Either parameters, omegas or etas
    values : str
        What to map to. Either parameters, omegas or etas
    level : str
        Which variability level to consider. Either iiv or iov. iiv is the default.

    Returns
    -------
    dict
        A dictionary from parameters, omegas or etas to a list of connected parameters, omegas or etas.

    Examples
    --------
    >>> from pharmpy.modeling import map_eta_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> map_eta_parameters(model, "parameters", "omegas")
    {'CL': ['IIV_CL'], 'VC': ['IIV_VC']}
    """

    if keys == values:
        raise ValueError("keys and values are the same in map_eta_parameters")
    if level.lower() not in ('iiv', 'iov'):
        raise ValueError(f"level can only be iiv or iov in map_eta_parameters. Was {level}")

    def get_etas_and_omegas(model, level):
        if level == 'iiv':
            etas = model.random_variables.iiv
        else:  # level == 'iov':
            etas = model.random_variables.iov
        omegas = [str(symbol) for symbol in etas.covariance_matrix.diagonal()]
        eta_names = etas.names
        return eta_names, omegas

    def get_omega_to_eta(model, level):
        eta_to_omega = get_eta_to_omega(model, level)
        d = reverse_and_explode_dict(eta_to_omega)
        return d

    def get_eta_to_omega(model, level):
        etas, omegas = get_etas_and_omegas(model, level)
        d = {}
        for eta, omega in zip(etas, omegas):
            if eta in d:
                d[eta].append(omega)
            else:
                d[eta] = [omega]
        return d

    def get_param_to_etas(model, level):
        from .expressions import get_individual_parameters, get_parameter_rv

        indpars = get_individual_parameters(model, level=level)
        d = {param: get_parameter_rv(model, param, var_type=level) for param in indpars}
        return d

    def get_param_to_omegas(model, level):
        param_to_etas = get_param_to_etas(model, level)
        eta_to_omega = get_eta_to_omega(model, level)
        d = {}
        for param, eta in param_to_etas.items():
            a = []
            for e in eta:
                a.extend(eta_to_omega[e])
            d[param] = a
        return d

    def reverse_and_explode_dict(d):
        new = {}
        for key, value in d.items():
            for e in value:
                if e in new:
                    new[e].append(key)
                else:
                    new[e] = [key]
        return new

    if keys == 'parameters' and values == 'etas':
        d = get_param_to_etas(model, level)
    elif keys == 'parameters' and values == 'omegas':
        d = get_param_to_omegas(model, level)
    elif keys == 'etas' and values == 'parameters':
        param_to_etas = get_param_to_etas(model, level)
        d = reverse_and_explode_dict(param_to_etas)
    elif keys == 'omegas' and values == 'parameters':
        param_to_omegas = get_param_to_omegas(model, level)
        d = reverse_and_explode_dict(param_to_omegas)
    elif keys == 'omegas' and values == 'etas':
        d = get_omega_to_eta(model, level)
    elif keys == 'etas' and values == 'omegas':
        d = get_eta_to_omega(model, level)
    else:
        raise ValueError(
            f"Bad combination of values and keys to map_eta_parameters. "
            f"(keys={keys} and values={values})"
        )
    return d
