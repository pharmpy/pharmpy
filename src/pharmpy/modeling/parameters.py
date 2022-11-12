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
    THETA(1)  0.004693   0.00  1000000.0  False
    THETA(2)  1.009160   0.00  1000000.0  False
    THETA(3)  0.100000  -0.99  1000000.0  False

    See also
    --------
    get_omegas : Get omega parameters
    get_sigmas : Get sigma parameters
    """
    rvs_fs = model.random_variables.free_symbols
    thetas = [p for p in model.parameters if p.symbol not in rvs_fs]
    return Parameters(thetas)


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
    OMEGA(1,1)  0.030963    0.0     ∞  False
    OMEGA(2,2)  0.031128    0.0     ∞  False

    See also
    --------
    get_thetas : Get theta parameters
    get_sigmas : Get sigma parameters
    """
    omegas = [p for p in model.parameters if p.symbol in model.random_variables.etas.free_symbols]
    return Parameters(omegas)


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
    SIGMA(1,1)  0.013241    0.0     ∞  False

    See also
    --------
    get_thetas : Get theta parameters
    get_omegas : Get omega parameters
    """
    sigmas = [
        p for p in model.parameters if p.symbol in model.random_variables.epsilons.free_symbols
    ]
    return Parameters(sigmas)


def set_initial_estimates(model: Model, inits):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_initial_estimates
    >>> model = load_example_model("pheno")
    >>> set_initial_estimates(model, {'THETA(1)': 2})   # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 2, lower=0.0, upper=1000000.0, fix=False)

    See also
    --------
    fix_parameters_to : Fixing and setting parameter initial estimates in the same function
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
    """
    model.parameters = model.parameters.set_initial_estimates(inits)
    return model


def set_upper_bounds(model: Model, bounds):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_upper_bounds
    >>> model = load_example_model("pheno")
    >>> set_upper_bounds(model, {'THETA(1)': 10})   # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=10, fix=False)

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
    model.parameters = Parameters(new)
    return model


def set_lower_bounds(model: Model, bounds):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, set_lower_bounds
    >>> model = load_example_model("pheno")
    >>> set_lower_bounds(model, {'THETA(1)': -10})   # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=-10, upper=1000000.0, fix=False)

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
    model.parameters = Parameters(new)
    return model


def fix_parameters(model: Model, parameter_names):
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
        Reference to the same model object

    Example
    -------
    >>> from pharmpy.modeling import fix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> fix_parameters(model, 'THETA(1)')       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000.0, fix=True)

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
    model.parameters = Parameters(new)
    return model


def unfix_parameters(model: Model, parameter_names):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters, unfix_parameters, load_example_model
    >>> model = load_example_model("pheno")
    >>> fix_parameters(model, ['THETA(1)', 'THETA(2)', 'THETA(3)'])     # doctest: +ELLIPSIS
    <...>
    >>> model.parameters.fix    # doctest: +ELLIPSIS
    {'THETA(1)': True, 'THETA(2)': True, 'THETA(3)': True, ...}
    >>> unfix_parameters(model, 'THETA(1)')       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters.fix        # doctest: +ELLIPSIS
    {'THETA(1)': False, 'THETA(2)': True, 'THETA(3)': True, ...}

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
    model.parameters = Parameters(new)
    return model


def fix_parameters_to(model: Model, inits):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters_to, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> fix_parameters_to(model, {'THETA(1)': 0.5})       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.5, lower=0.0, upper=1000000.0, fix=True)

    See also
    --------
    fix_parameters : Fix parameters
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    unfix_paramaters : Unfixing parameters
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function

    """
    fix_parameters(model, inits.keys())
    set_initial_estimates(model, inits)
    return model


def unfix_parameters_to(model: Model, inits):
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
    >>> fix_parameters(model, ['THETA(1)', 'THETA(2)', 'THETA(3)'])     # doctest: +ELLIPSIS
    <...>
    >>> model.parameters.fix    # doctest: +ELLIPSIS
    {'THETA(1)': True, 'THETA(2)': True, 'THETA(3)': True, ...}
    >>> unfix_parameters_to(model, {'THETA(1)': 0.5})       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters.fix        # doctest: +ELLIPSIS
    {'THETA(1)': False, 'THETA(2)': True, 'THETA(3)': True, ...}
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.5, lower=0.0, upper=1000000.0, fix=False)

    Returns
    -------
    Model
        Reference to the same model object

    See also
    --------
    fix_parameters : Fix parameters
    fix_or_unfix_parameters : Fix or unfix parameters (given boolean)
    unfix_paramaters : Unfixing parameters
    fix_paramaters_to : Fixing parameters and setting a new initial estimate in the same
        function
    """
    unfix_parameters(model, inits.keys())
    set_initial_estimates(model, inits)
    return model


def fix_or_unfix_parameters(model: Model, parameters):
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
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> fix_or_unfix_parameters(model, {'THETA(1)': True})       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000.0, fix=True)

    Returns
    -------
    Model
        Reference to the same model object

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
    model.parameters = Parameters(new)
    return model


def unconstrain_parameters(model: Model, parameter_names):
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
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000.0, fix=False)
    >>> unconstrain_parameters(model, ['THETA(1)'])       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=-∞, upper=∞, fix=False)

    Returns
    -------
    Model
        Reference to the same model object

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
    model.parameters = Parameters(new)
    return model


def add_population_parameter(model: Model, name, init, lower=None, upper=None, fix=False):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import add_population_parameter, load_example_model
    >>> model = load_example_model("pheno")
    >>> add_population_parameter(model, 'POP_KA', 2)       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters
                   value lower      upper    fix
    THETA(1)    0.004693   0.0  1000000.0  False
    THETA(2)    1.009160   0.0  1000000.0  False
    THETA(3)    0.100000 -0.99  1000000.0  False
    OMEGA(1,1)  0.030963   0.0          ∞  False
    OMEGA(2,2)  0.031128   0.0          ∞  False
    SIGMA(1,1)  0.013241   0.0          ∞  False
    POP_KA      2.000000    -∞          ∞  False
    """

    param = Parameter.create(name, init, lower=lower, upper=upper, fix=fix)
    params = model.parameters + param
    model.parameters = params
    return model
