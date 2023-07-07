"""Common modeling pipeline elements
:meta private:
"""

import importlib
import re
import warnings
from pathlib import Path
from typing import Dict, Union

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


def read_model(path: Union[str, Path]):
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
    model = Model.parse_model(path)
    return model


def read_model_from_string(code: str):
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
    model = Model.parse_model_from_string(code)
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
        Pharmpy model object

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
        model = model.replace(name=new_name)
    model = model.write_files(path=path, force=force)
    if not force and path.exists():
        raise FileExistsError(f'Cannot overwrite model at {path} with "force" not set')
    with open(path, 'w', encoding='latin-1') as fp:
        fp.write(model.model_code)
    return model


def convert_model(model: Model, to_format: str):
    """Convert model to other format

    Note that the operation is not done inplace.

    Parameters
    ----------
    model : Model
        Model to convert
    to_format : str
        Name of format to convert into. Currently supported 'generic', 'nlmixr', 'nonmem', and 'rxode'

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
    supported = ['generic', 'nlmixr', 'nonmem', 'rxode']
    if to_format not in supported:
        raise ValueError(f"Unknown format {to_format}: supported formats are f{supported}")
    module_name = 'pharmpy.model.external.' + to_format
    module = importlib.import_module(module_name)
    new = module.convert_model(model)
    return new


def get_model_code(model: Model):
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
    >>> from pharmpy.modeling import get_model_code, load_example_model
    >>> model = load_example_model("pheno")
    >>> get_model_code(model)  # doctest: +SKIP

    """
    return model.model_code


def print_model_code(model: Model):
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


def set_name(model: Model, new_name: str):
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
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import set_name, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.name
    'pheno'
    >>> model = set_name(model, "run2")
    >>> model.name
    'run2'

    """
    model = model.replace(name=new_name)
    return model


def bump_model_number(model: Model, path: Union[str, Path] = None):
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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import bump_model_number, load_example_model
    >>> model = load_example_model("pheno")
    >>> model = model.replace(name="run2")
    >>> model = bump_model_number(model)
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
        model = model.replace(name=new_name)
    return model


def load_example_model(name: str):
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
    TVCL = PTVCL⋅WGT
    TVV = PTVV⋅WGT
          ⎧TVV⋅(THETA₃ + 1)  for APGR < 5
          ⎨
    TVV = ⎩       TVV           otherwise
               ETA₁
    CL = TVCL⋅ℯ
             ETA₂
    V = TVV⋅ℯ
    S₁ = V
    Bolus(AMT, admid=1)
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL
        ─────────
    F =     S₁
    W = F
    Y = EPS₁⋅W + F
    IPRED = F
    IRES = DV - IPRED
            IRES
            ────
    IWRES =  W

    """
    available = ('moxo', 'pheno', 'pheno_linear')
    if name not in available:
        raise ValueError(f'Unknown example model {name}. Available examples: {available}')
    path = Path(__file__).resolve().parent / 'example_models' / (name + '.mod')
    model = read_model(path)
    return model


def get_model_covariates(model: Model, strings: bool = False):
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
        dose_comp = odes.dosing_compartment[0]
        cb = CompartmentalSystemBuilder(odes)
        cb.set_dose(dose_comp, None)
        cs = CompartmentalSystem(cb)
        statements = model.statements.before_odes + cs + model.statements.after_odes
        ode_deps = statements.dependencies(cs)
    else:
        ode_deps = set()

    # FIXME: This should be handled for all DVs
    first_dv = list(model.dependent_variables.keys())[0]
    y = model.statements.find_assignment(first_dv)
    y_deps = model.statements.error.dependencies(y)

    covs = datasymbs.intersection(ode_deps | y_deps)

    # Disallow ID from being a covariate
    covs = covs - {sympy.Symbol(model.datainfo.id_column.name)}

    covs = list(covs)
    covs = sorted(covs, key=lambda x: x.name)  # sort to make order deterministic
    if strings:
        covs = [str(x) for x in covs]
    return covs


def print_model_symbols(model: Model):
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
    Thetas: PTVCL, PTVV, THETA₃
    Etas: ETA₁, ETA₂
    Omegas: IVCL, IVV
    Epsilons: EPS₁
    Sigmas: SIGMA₁ ₁
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
        env_path = config.env_config_path()
        if env_path is not None:
            return str(env_path.resolve())
        else:
            config_path = config.user_config_path()
            if config_path.exists():
                return str(config_path.resolve())
            else:
                warnings.warn(f'Cannot find config path {config_path}')
                return None
    else:
        warnings.warn('User config file is disabled')
        return None


def create_config_template():
    r"""Create a basic config file template

    If a configuration file already exists it will not be overwritten

    Example
    -------
    >>> from pharmpy.modeling import create_config_template
    >>> create_config_template()  # doctest: +SKIP
    """
    template = r"""[pharmpy.plugins.nonmem]
;default_nonmem_path="""

    if config.user_config_file_enabled():
        path = config.user_config_path()
        if path is not None:
            if not path.exists():
                path.parent.mkdir(parents=True)
                with open(path, 'w') as fp:
                    print(template, file=fp)
            else:
                warnings.warn('Config file already exists')
    else:
        warnings.warn('User config file is disabled')


def remove_unused_parameters_and_rvs(model: Model):
    """Remove any parameters and rvs that are not used in the model statements

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Model
        Pharmpy model object
    """
    new_rvs, new_params = _get_unused_parameters_and_rvs(
        model.statements, model.parameters, model.random_variables
    )
    model = model.replace(random_variables=new_rvs, parameters=new_params)
    return model.update_source()


def _get_unused_parameters_and_rvs(statements, parameters, random_variables):
    symbols = statements.free_symbols

    # Find unused rvs needing unjoining
    to_unjoin = []
    for dist in random_variables:
        if isinstance(dist, JointNormalDistribution):
            names = dist.names
            for i, name in enumerate(names):
                params = dist.variance[i, :].free_symbols
                symb = sympy.Symbol(name)
                if symb not in symbols and symbols.isdisjoint(params):
                    to_unjoin.append(name)

    rvs = random_variables.unjoin(to_unjoin)

    new_dists = []
    for dist in rvs:
        if isinstance(dist, NormalDistribution):
            if not symbols.isdisjoint(dist.free_symbols):
                new_dists.append(dist)
        else:
            new_dists.append(dist)

    new_rvs = RandomVariables(tuple(new_dists), rvs._eta_levels, rvs._epsilon_levels)

    new_params = []
    for p in parameters:
        symb = p.symbol
        if symb in symbols or symb in new_rvs.free_symbols or (p.fix and p.init == 0):
            new_params.append(p)

    return new_rvs, Parameters.create(new_params)


def rename_symbols(
    model: Model, new_names: Dict[Union[str, sympy.Symbol], Union[str, sympy.Symbol]]
):
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
        Pharmpy model object
    """
    d = {
        (sympy.Symbol(key) if isinstance(key, str) else key): (
            sympy.Symbol(val) if isinstance(val, str) else val
        )
        for key, val in new_names.items()
    }

    new = []
    for p in model.parameters:
        if p.symbol in d:
            newparam = Parameter(
                name=d[p.symbol].name, init=p.init, lower=p.lower, upper=p.upper, fix=p.fix
            )
        else:
            newparam = p
        new.append(newparam)

    model = model.replace(
        parameters=Parameters.create(new),
        statements=model.statements.subs(d),
        random_variables=model.random_variables.subs(d),
    )
    return model.update_source()
    # FIXME: Only handles parameters, statements and random_variables and no clashes and circular renaming
