"""Common modeling pipeline elements
:meta private:
"""

from io import StringIO
from pathlib import Path

import sympy

from pharmpy import config
from pharmpy.estimation import EstimationStep
from pharmpy.model_factory import Model
from pharmpy.modeling.help_functions import _as_integer
from pharmpy.statements import Assignment, CompartmentalSystem


def read_model(path):
    """Read model from file

    Parameters
    ----------
    path : str or Path
        Path to model

    Returns
    -------
    Model
        Read model object

    Example
    -------
    >>> from pharmpy.modeling import read_model
    >>> model = read_model("/home/run1.mod")    # doctest: +SKIP

    See also
    --------
    read_model_from_database : Read model from database
    read_model_from_string : Read model from string

    """
    model = Model(path)
    return model


def read_model_from_database(name, database=None):
    """Read model from model database

    Parameters
    ----------
    name : str
        Name of model to use as lookup
    database : Database
        Database to use. Will use default database if not specified.

    Returns
    -------
    Model
        Read model object

    Examples
    --------
    >>> from pharmpy.modeling import read_model_from_database
    >>> model = read_model_from_database("run1")    # doctest: +SKIP

    See also
    --------
    read_model : Read model from file
    read_model_from_string : Read model from string

    """
    if database is None:
        import pharmpy.workflows

        database = pharmpy.workflows.default_model_database()
    model = database.get_model(name)
    return model


def read_model_from_string(code, path=None):
    """Read model from the model code in a string

    Parameters
    ----------
    code : str
        Model code to read
    path : Path or str
        Specified to set the path for the created model

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import read_model_from_string
    >>> s = '''$PROBLEM
    ... $INPUT ID DV TIME
    ... $DATA file.csv
    ... $PRED
    ... Y=THETA(1)+ETA(1)+ERR(1)
    ... $THETA 1
    ... $OMEGA 0.1
    ... $SIGMA 1
    ... $ESTIMATION METHOD=1'''
    >>> read_model_from_string(s)  # doctest:+ELLIPSIS
    <...>

    See also
    --------
    read_model : Read model from file
    read_model_from_database : Read model from database

    """
    model = Model(StringIO(code))
    if path is not None:
        path = Path(path)
        import pharmpy.workflows

        model.database = pharmpy.workflows.default_model_database(path)
    return model


def write_model(model, path='', force=True):
    """Write model code to file

    Parameters
    ----------
    model : Model
        Pharmpy model
    path : str
        Destination path
    force : bool
        Force overwrite, default is True

    Returns
    -------
    Model
        Reference to the same model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, write_model
    >>> model = load_example_model("pheno")
    >>> write_model(model)   # doctest: +SKIP

    """
    model.write(path=path, force=force)
    return model


def convert_model(model, to_format):
    """Convert model to other format

    Parameters
    ----------
    model : Model
        Model to convert
    to_format : str
        Name of format to convert into. Currently supported 'nlmixr' and 'nonmem'

    Returns
    -------
    Model
        New model object with new underlying model format

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, convert_model
    >>> model = load_example_model("pheno")
    >>> converted_model = convert_model(model, "nlmixr")    # doctest: +SKIP

    """
    supported = ['nlmixr', 'nonmem']
    if to_format not in supported:
        raise ValueError(f"Unknown format {to_format}: supported formats are f{supported}")
    # FIXME: Use generic code below
    if to_format == 'nlmixr':
        import pharmpy.plugins.nlmixr.model as nlmixr

        new = nlmixr.convert_model(model)
    else:
        import pharmpy.plugins.nonmem.model as nonmem

        new = nonmem.convert_model(model)
    return new


def generate_model_code(model):
    """Get the model code of the underlying model language

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    str
        Model code

    Examples
    --------
    >>> from pharmpy.modeling import generate_model_code, load_example_model
    >>> model = load_example_model("pheno")
    >>> generate_model_code(model)  # doctest: +SKIP

    """
    return model.model_code


def copy_model(model, name=None):
    """Copies model to a new model object

    Parameters
    ----------
    model : Model
        Pharmpy model
    name : str
        Optional new name of model

    Returns
    -------
    Model
        A copy of the input model

    Example
    -------
    >>> from pharmpy.modeling import copy_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model_copy = copy_model(model, "pheno2")

    """
    new = model.copy()
    if name is not None:
        new.name = name
    return new


def set_name(model, new_name):
    """Set name of model object

    Parameters
    ----------
    model : Model
        Pharmpy model
    new_name : str
        New name of model

    Returns
    -------
    Model
        Reference to the same model object

    Example
    -------
    >>> from pharmpy.modeling import set_name, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.name
    'pheno'
    >>> set_name(model, "run2")  # doctest: +ELLIPSIS
    <...>
    >>> model.name
    'run2'

    """
    model.name = new_name
    return model


def set_initial_estimates(model, inits):
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
    Parameter("THETA(1)", 2, lower=0.0, upper=1000000, fix=False)

    """
    model.parameters.inits = inits
    return model


def fix_parameters(model, parameter_names):
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
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000, fix=False)
    >>> fix_parameters(model, 'THETA(1)')       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000, fix=True)

    See also
    --------
    fix_parameters_to : Fixing and setting parameter initial estimates in the same function
    unfix_paramaters : Unfixing parameters
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function

    """
    if isinstance(parameter_names, str):
        d = {parameter_names: True}
    else:
        d = {name: True for name in parameter_names}
    model.parameters.fix = d
    return model


def unfix_parameters(model, parameter_names):
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
    fix_parameters_to : Fixing and setting parameter initial estimates in the same function

    """
    if isinstance(parameter_names, str):
        d = {parameter_names: False}
    else:
        d = {name: False for name in parameter_names}
    model.parameters.fix = d
    return model


def fix_parameters_to(model, parameter_names, values):
    """Fix parameters to

    Fix all listed parameters to specified value/values

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list or str
        one parameter name or a list of parameter names
    values : list or float
        one value or a list of values (must be equal to number of parameter_names)

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters_to, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.00469307, lower=0.0, upper=1000000, fix=False)
    >>> fix_parameters_to(model, 'THETA(1)', 0.5)       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.5, lower=0.0, upper=1000000, fix=True)

    See also
    --------
    fix_parameters : Fix parameters
    unfix_paramaters : Unfixing parameters
    unfix_paramaters_to : Unfixing parameters and setting a new initial estimate in the same
        function

    """
    if not parameter_names:
        parameter_names = [p.name for p in model.parameters]

    fix_parameters(model, parameter_names)
    d = _create_init_dict(parameter_names, values)
    set_initial_estimates(model, d)

    return model


def unfix_parameters_to(model, parameter_names, values):
    """Unix parameters to

    Unfix all listed parameters to specified value/values

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_names : list or str
        one parameter name or a list of parameter names
    values : list or float
        one value or a list of values (must be equal to number of parameter_names)

    Examples
    --------
    >>> from pharmpy.modeling import fix_parameters, unfix_parameters_to, load_example_model
    >>> model = load_example_model("pheno")
    >>> fix_parameters(model, ['THETA(1)', 'THETA(2)', 'THETA(3)'])     # doctest: +ELLIPSIS
    <...>
    >>> model.parameters.fix    # doctest: +ELLIPSIS
    {'THETA(1)': True, 'THETA(2)': True, 'THETA(3)': True, ...}
    >>> unfix_parameters_to(model, 'THETA(1)', 0.5)       # doctest: +ELLIPSIS
    <...>
    >>> model.parameters.fix        # doctest: +ELLIPSIS
    {'THETA(1)': False, 'THETA(2)': True, 'THETA(3)': True, ...}
    >>> model.parameters['THETA(1)']
    Parameter("THETA(1)", 0.5, lower=0.0, upper=1000000, fix=False)

    Returns
    -------
    Model
        Reference to the same model object
    """
    if not parameter_names:
        parameter_names = [p.name for p in model.parameters]

    unfix_parameters(model, parameter_names)
    d = _create_init_dict(parameter_names, values)
    set_initial_estimates(model, d)

    return model


def _create_init_dict(parameter_names, values):
    if isinstance(parameter_names, str):
        d = {parameter_names: values}
    else:
        if not isinstance(values, list):
            values = [values] * len(parameter_names)
        if len(parameter_names) != len(values):
            raise ValueError(
                'Incorrect number of values, must be equal to number of parameters '
                '(or one if same for all)'
            )
        d = {name: value for name, value in zip(parameter_names, values)}

    return d


def set_estimation_step(model, method, idx=0, **kwargs):
    """Set estimation step

    Sets estimation step for a model. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM, BAYES

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : str
        estimation method to change to
    idx : int
        index of estimation step, default is 0 (first estimation step)
    kwargs
        Arguments to pass to EstimationStep (such as interaction, evaluation)

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> set_estimation_step(model, "IMP", evaluation=True, tool_options=opts)   # doctest: +ELLIPSIS
    <...>
    >>> model.estimation_steps[0]   # doctest: +ELLIPSIS
    EstimationStep("IMP", interaction=True, cov=True, evaluation=True, ..., tool_options=...

    See also
    --------
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step

    """
    try:
        idx = _as_integer(idx)
    except TypeError:
        raise TypeError(f'Index must be integer: {idx}')

    model.estimation_steps[idx].method = method
    for key, value in kwargs.items():
        setattr(model.estimation_steps[idx], key, value)
    return model


def add_estimation_step(model, method, idx=None, **kwargs):
    """Add estimation step

    Adds estimation step for a model in a given index. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : str
        estimation method to change to
    idx : int
        index of estimation step (starting from 0), default is None (adds step at the end)
    kwargs
        Arguments to pass to EstimationStep (such as interaction, evaluation)

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> add_estimation_step(model, "IMP", tool_options=opts)   # doctest: +ELLIPSIS
    <...>
    >>> ests = model.estimation_steps
    >>> len(ests)
    2
    >>> ests[1]   # doctest: +ELLIPSIS
    EstimationStep("IMP", interaction=False, cov=False, ..., tool_options={'NITER': 1000,...

    See also
    --------
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step

    """
    meth = EstimationStep(method, **kwargs)

    if idx is not None:
        try:
            idx = _as_integer(idx)
        except TypeError:
            raise TypeError(f'Index must be integer: {idx}')
        model.estimation_steps.insert(idx, meth)
    else:
        model.estimation_steps.append(meth)

    return model


def remove_estimation_step(model, idx):
    """Remove estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model
    idx : int
        index of estimation step to remove (starting from 0)

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_estimation_step(model, 0)      # doctest: +ELLIPSIS
    <...>
    >>> ests = model.estimation_steps
    >>> len(ests)
    0

    See also
    --------
    add_estimation_step
    set_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step

    """
    try:
        idx = _as_integer(idx)
    except TypeError:
        raise TypeError(f'Index must be integer: {idx}')

    del model.estimation_steps[idx]
    return model


def append_estimation_step_options(model, tool_options, idx):
    """Append estimation step options

    Appends options to an existing estimation step.

    Parameters
    ----------
    model : Model
        Pharmpy model
    tool_options : dict
        any additional tool specific options
    idx : int
        index of estimation step (starting from 0)

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> append_estimation_step_options(model, tool_options=opts, idx=0)   # doctest: +ELLIPSIS
    <...>
    >>> est = model.estimation_steps[0]
    >>> len(est.tool_options)
    2

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    add_covariance_step
    remove_covariance_step

    """
    try:
        idx = _as_integer(idx)
    except TypeError:
        raise TypeError(f'Index must be integer: {idx}')

    model.estimation_steps[idx].append_tool_options(tool_options)
    return model


def add_covariance_step(model):
    """Adds covariance step to the final estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_estimation_step(model, 'FOCE', cov=False)      # doctest: +ELLIPSIS
    <...>
    >>> add_covariance_step(model)      # doctest: +ELLIPSIS
    <...>
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep("FOCE", interaction=True, cov=True, ...)

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    remove_covariance_step

    """
    model.estimation_steps[-1].cov = True
    return model


def remove_covariance_step(model):
    """Removes covariance step to the final estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_covariance_step(model)      # doctest: +ELLIPSIS
    <...>
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep("FOCE", interaction=True, cov=False, ...)

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step

    """
    model.estimation_steps[-1].cov = False
    return model


def load_example_model(name):
    """Load an example model

    Load an example model from models built into Pharmpy

    Parameters
    ----------
    name : str
        Name of the model. Currently available model is "pheno"

    Returns
    -------
    Model
        Loaded model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model
    >>> model = load_example_model("pheno")
    >>> model.statements
             ⎧TIME  for AMT > 0
             ⎨
    BTIME := ⎩ 0     otherwise
    TAD := -BTIME + TIME
    TVCL := THETA(1)⋅WGT
    TVV := THETA(2)⋅WGT
           ⎧TVV⋅(THETA(3) + 1)  for APGR < 5
           ⎨
    TVV := ⎩       TVV           otherwise
                ETA(1)
    CL := TVCL⋅ℯ
              ETA(2)
    V := TVV⋅ℯ
    S₁ := V
    Bolus(AMT)
    ┌───────┐       ┌──────┐
    │CENTRAL│──CL/V→│OUTPUT│
    └───────┘       └──────┘
         A_CENTRAL
         ─────────
    F :=     S₁
    W := F
    Y := EPS(1)⋅W + F
    IPRED := F
    IRES := DV - IPRED
             IRES
             ────
    IWRES :=  W

    """
    available = ('pheno',)
    if name not in available:
        raise ValueError(f'Unknown example model {name}. Available examples: {available}')
    path = Path(__file__).parent.resolve() / 'example_models' / (name + '.mod')
    model = read_model(path)
    return model


def get_model_covariates(model, strings=False):
    """List of covariates used in model

    A covariate in the model is here defined to be a data item
    affecting the model prediction excluding dosing items.

    Parameters
    ----------
    model : Model
        Pharmpy model
    strings : bool
        Return strings instead of symbols? False (default) will give symbols

    Returns
    -------
    list
        Covariate symbols or names

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_model_covariates
    >>> model = load_example_model("pheno")
    >>> get_model_covariates(model)
    [APGR, WGT]
    >>> get_model_covariates(model, strings=True)
    ['APGR', 'WGT']

    """
    symbs = model.statements.dependencies(model.dependent_variable)
    # Remove dose symbol if not used as covariate
    datasymbs = {sympy.Symbol(s) for s in model.dataset.columns}
    cov_dose_symbols = set()
    if isinstance(model.statements.ode_system, CompartmentalSystem):
        dosecmt = model.statements.ode_system.dosing_compartment
        dosesyms = dosecmt.free_symbols
        for s in model.statements:
            if isinstance(s, Assignment):
                cov_dose_symbols |= dosesyms.intersection(s.rhs_symbols)
    covs = list(symbs.intersection(datasymbs) - cov_dose_symbols)
    covs = sorted(covs, key=lambda x: x.name)  # sort to make order deterministic
    if strings:
        covs = [str(x) for x in covs]
    return covs


def print_model_symbols(model):
    """Print all symbols defined in a model

    Symbols will be in one of the categories thetas, etas, omegas, epsilons, sigmas,
    variables and data columns

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, print_model_symbols
    >>> model = load_example_model("pheno")
    >>> print_model_symbols(model)
    Thetas: THETA(1), THETA(2), THETA(3)
    Etas: ETA(1), ETA(2)
    Omegas: OMEGA(1,1), OMEGA(2,2)
    Epsilons: EPS(1)
    Sigmas: SIGMA(1,1)
    Variables: BTIME, TAD, TVCL, TVV, TVV, CL, V, S₁, F, W, Y, IPRED, IRES, IWRES
    Data columns: ID, TIME, AMT, WGT, APGR, DV, FA1, FA2

    """
    etas = [sympy.pretty(sympy.Symbol(name)) for name in model.random_variables.etas.names]
    epsilons = [sympy.pretty(sympy.Symbol(name)) for name in model.random_variables.epsilons.names]
    omegas = [sympy.pretty(sympy.Symbol(n)) for n in model.random_variables.etas.parameter_names]
    sigmas = [
        sympy.pretty(sympy.Symbol(n)) for n in model.random_variables.epsilons.parameter_names
    ]
    thetas = []
    for param in model.parameters:
        if param.name not in model.random_variables.parameter_names:
            thetas.append(sympy.pretty(param.symbol))
    variables = []
    for sta in model.statements:
        if hasattr(sta, 'symbol'):
            variables.append(sympy.pretty(sta.symbol))
    s = f'Thetas: {", ".join(thetas)}\n'
    s += f'Etas: {", ".join(etas)}\n'
    s += f'Omegas: {", ".join(omegas)}\n'
    s += f'Epsilons: {", ".join(epsilons)}\n'
    s += f'Sigmas: {", ".join(sigmas)}\n'
    s += f'Variables: {", ".join(variables)}\n'
    s += f'Data columns: {", ".join(model.dataset.columns)}'
    print(s)


def get_config_path():
    """Returns path to the user config path

    Returns
    -------
    str
        Path to user config

    Example
    -------
    >>> from pharmpy.modeling import get_config_path
    >>> get_config_path()    # doctest: +ELLIPSIS
    '.../pharmpy.conf'
    """
    return str(config.user_config_dir())
