"""Common modeling pipeline elements
:meta private:
"""

import re
import warnings
from io import StringIO
from pathlib import Path
from typing import Union

import pharmpy.config as config
from pharmpy.deps import sympy
from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.model import (
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    JointNormalDistribution,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
    RandomVariables,
)


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
    path = normalize_user_given_path(path)
    model = Model.create_model(path)
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
    model = database.retrieve_model(name)
    return model


def read_model_from_string(code):
    """Read model from the model code in a string

    Parameters
    ----------
    code : str
        Model code to read

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
    model = Model.create_model(StringIO(code))
    return model


def write_model(model: Model, path: Union[str, Path] = '', force: bool = True):
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
    path = normalize_user_given_path(path)
    if not path or path.is_dir():
        try:
            filename = f'{model.name}{model.filename_extension}'
        except AttributeError:
            raise ValueError(
                'Cannot name model file as no path argument was supplied and the '
                'model has no name.'
            )
        path = path / filename
        new_name = None
    else:
        # Set new name given filename, but after we've checked for existence
        new_name = path.stem
    if not force and path.exists():
        raise FileExistsError(f'File {path} already exists.')
    if new_name:
        model.name = new_name
    model.update_source(path=path, force=force)
    if not force and path.exists():
        raise FileExistsError(f'Cannot overwrite model at {path} with "force" not set')
    with open(path, 'w', encoding='latin-1') as fp:
        fp.write(model.model_code)
    return model


def convert_model(model, to_format):
    """Convert model to other format

    Note that the operation is not done inplace.

    Parameters
    ----------
    model : Model
        Model to convert
    to_format : str
        Name of format to convert into. Currently supported 'generic', 'nlmixr' and 'nonmem'

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
    supported = ['generic', 'nlmixr', 'nonmem']
    if to_format not in supported:
        raise ValueError(f"Unknown format {to_format}: supported formats are f{supported}")
    # FIXME: Use code that can discover plugins below
    if to_format == 'generic':
        new = Model.create_model()
        new.parameters = model.parameters
        new.random_variables = model.random_variables
        new.statements = model.statements
        new.dataset = model.dataset
        new.estimation_steps = model.estimation_steps
        new.datainfo = model.datainfo
        new.name = model.name
        new.description = model.description
        new.dependent_variable = model.dependent_variable
        new.observation_transformation = model.observation_transformation
        new.parent_model = model.name
        try:
            new.filename_extension = model.filename_extension
        except AttributeError:
            pass
        try:
            if model.initial_individual_estimates is not None:
                new.initial_individual_estimates = model.initial_individual_estimates.copy()
        except AttributeError:
            pass
        try:
            new.database = model.database
        except AttributeError:
            pass
        return new
    elif to_format == 'nlmixr':
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


def print_model_code(model):
    """Print the model code of the underlying model language

    Parameters
    ----------
    model : Model
        Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import print_model_code, load_example_model
    >>> model = load_example_model("pheno")
    >>> print_model_code(model)
    $PROBLEM PHENOBARB SIMPLE MODEL
    $DATA 'pheno.dta' IGNORE=@
    $INPUT ID TIME AMT WGT APGR DV FA1 FA2
    $SUBROUTINE ADVAN1 TRANS2
    <BLANKLINE>
    $PK
    IF(AMT.GT.0) BTIME=TIME
    TAD=TIME-BTIME
    TVCL=THETA(1)*WGT
    TVV=THETA(2)*WGT
    IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
    CL=TVCL*EXP(ETA(1))
    V=TVV*EXP(ETA(2))
    S1=V
    <BLANKLINE>
    $ERROR
    W=F
    Y=F+W*EPS(1)
    IPRED=F
    IRES=DV-IPRED
    IWRES=IRES/W
    <BLANKLINE>
    $THETA (0,0.00469307) ; PTVCL
    $THETA (0,1.00916) ; PTVV
    $THETA (-.99,.1)
    $OMEGA DIAGONAL(2)
     0.0309626  ;       IVCL
     0.031128  ;        IVV
    <BLANKLINE>
    $SIGMA 0.013241
    $ESTIMATION METHOD=1 INTERACTION
    $COVARIANCE UNCONDITIONAL
    $TABLE ID TIME AMT WGT APGR IPRED PRED TAD CWRES NPDE NOAPPEND
           NOPRINT ONEHEADER FILE=pheno.tab
    """
    print(model.model_code)


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


def bump_model_number(model, path=None):
    """If the model name ends in a number increase it

    If path is set increase the number until no file exists
    with the same name in path.
    If model name does not end in a number do nothing.

    Parameters
    ----------
    model : Model
        Pharmpy model object
    path : Path in which to find next unique number
        Default is to not look for files.

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import bump_model_number, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.name = "run2"
    >>> bump_model_number(model)    # doctest: +ELLIPSIS
    <...>
    >>> model.name
    'run3'
    """
    name = model.name
    m = re.search(r'(.*?)(\d+)$', name)
    if m:
        stem = m.group(1)
        n = int(m.group(2))
        if path is None:
            new_name = f'{stem}{n + 1}'
        else:
            path = normalize_user_given_path(path)
            while True:
                n += 1
                new_name = f'{stem}{n}'
                new_path = (path / new_name).with_suffix(model.filename_extension)
                if not new_path.exists():
                    break
        model.name = new_name
    return model


def load_example_model(name):
    """Load an example model

    Load an example model from models built into Pharmpy

    Parameters
    ----------
    name : str
        Name of the model. Currently available models are "pheno" and "pheno_linear"

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
    BTIME = ⎩ 0     otherwise
    TAD = -BTIME + TIME
    TVCL = THETA(1)⋅WGT
    TVV = THETA(2)⋅WGT
          ⎧TVV⋅(THETA(3) + 1)  for APGR < 5
          ⎨
    TVV = ⎩       TVV           otherwise
               ETA(1)
    CL = TVCL⋅ℯ
             ETA(2)
    V = TVV⋅ℯ
    S₁ = V
    Bolus(AMT)
    ┌───────┐       ┌──────┐
    │CENTRAL│──CL/V→│OUTPUT│
    └───────┘       └──────┘
        A_CENTRAL
        ─────────
    F =     S₁
    W = F
    Y = EPS(1)⋅W + F
    IPRED = F
    IRES = DV - IPRED
            IRES
            ────
    IWRES =  W

    """
    available = ('pheno', 'pheno_linear')
    if name not in available:
        raise ValueError(f'Unknown example model {name}. Available examples: {available}')
    path = Path(__file__).resolve().parent / 'example_models' / (name + '.mod')
    model = read_model(path)
    return model


def get_model_covariates(model, strings=False):
    """List of covariates used in model

    A covariate in the model is here defined to be a data item
    affecting the model prediction excluding dosing items that
    are not used in model code.

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
    datasymbs = {sympy.Symbol(s) for s in model.datainfo.names}

    odes = model.statements.ode_system

    # Consider statements that are dependencies of the ode system and y
    if odes:
        odes = odes.to_compartmental_system()
        dose_comp = odes.dosing_compartment
        cb = CompartmentalSystemBuilder(odes)
        cb.set_dose(dose_comp, None)
        cs = CompartmentalSystem(cb)
        statements = model.statements.before_odes + cs + model.statements.after_odes
        ode_deps = statements.dependencies(cs)
    else:
        ode_deps = set()

    y = model.statements.find_assignment(model.dependent_variable)
    y_deps = model.statements.error.dependencies(y)

    covs = datasymbs.intersection(ode_deps | y_deps)

    # Disallow ID from being a covariate
    covs = covs - {sympy.Symbol(model.datainfo.id_column.name)}

    covs = list(covs)
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
    s += f'Data columns: {", ".join(model.datainfo.names)}'
    print(s)


def get_config_path():
    r"""Returns path to the user config path

    Returns
    -------
    str or None
        Path to user config or None if file does not exist

    Example
    -------
    >>> from pharmpy.modeling import get_config_path
    >>> get_config_path()  # doctest: +SKIP
    """
    if config.user_config_file_enabled():
        config_path = config.user_config_dir()
        if config_path.exists():
            return str(config_path)
        else:
            warnings.warn(f'Cannot find config path {config_path}')
            return None
    else:
        warnings.warn('User config file is disabled')
        return None


def remove_unused_parameters_and_rvs(model):
    """Remove any parameters and rvs that are not used in the model statements

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Model
        Reference to same model object
    """
    symbols = model.statements.free_symbols

    # Find unused rvs needing unjoining
    to_unjoin = []
    for dist in model.random_variables:
        if isinstance(dist, JointNormalDistribution):
            names = dist.names
            for i, name in enumerate(names):
                params = dist.variance[i, :].free_symbols
                symb = sympy.Symbol(name)
                if symb not in symbols and symbols.isdisjoint(params):
                    to_unjoin.append(name)

    rvs = model.random_variables.unjoin(to_unjoin)

    new_dists = []
    for dist in rvs:
        if isinstance(dist, NormalDistribution):
            if not symbols.isdisjoint(dist.free_symbols):
                new_dists.append(dist)
        else:
            new_dists.append(dist)

    new_rvs = RandomVariables(tuple(new_dists), rvs._eta_levels, rvs._epsilon_levels)
    model.random_variables = new_rvs

    new_params = []
    for p in model.parameters:
        symb = p.symbol
        if symb in symbols or symb in new_rvs.free_symbols or (p.fix and p.init == 0):
            new_params.append(p)
    model.parameters = Parameters(new_params)
    return model


def rename_symbols(model, new_names):
    """Rename symbols in the model

    Make sure that no name clash occur.

    Parameters
    ----------
    model : Model
        Pharmpy model object
    new_names : dict
        From old name or symbol to new name or symbol

    Returns
    -------
    Model
        Reference to same model object
    """
    d = {sympy.Symbol(key): sympy.Symbol(val) for key, val in new_names.items()}

    new = []
    for p in model.parameters:
        if p.symbol in d:
            newparam = Parameter(
                name=d[p.symbol].name, init=p.init, lower=p.lower, upper=p.upper, fix=p.fix
            )
        else:
            newparam = p
        new.append(newparam)
    model.parameters = Parameters(new)

    model.statements = model.statements.subs(d)
    return model
    # FIXME: Only handles parameters and statements and no clashes and circular renaming
