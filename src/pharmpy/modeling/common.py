"""Common modeling pipeline elements
:meta private:
"""

from __future__ import annotations

import importlib
import itertools
import re
import warnings
from pathlib import Path
from typing import Literal, Mapping, Optional, Union

import pharmpy.config as config
from pharmpy.basic import Expr, TSymbol
from pharmpy.deps import pandas
from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.model import (
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    JointNormalDistribution,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
    RandomVariables,
    get_and_check_dataset,
)
from pharmpy.model.statements import Output


def read_model(path: Union[str, Path], missing_data_token: Optional[str] = None) -> Model:
    """Read model from file

    Parameters
    ----------
    path : str or Path
        Path to model
    missing_data_token : str
        Use this token for missing data. This option will override the token from the config.
        (This option was added in Pharmpy version 1.2.0)

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
    model = Model.parse_model(path, missing_data_token=missing_data_token)
    return model


def read_model_from_string(code: str) -> Model:
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


def write_model(model: Model, path: Union[str, Path] = '', force: bool = True) -> Model:
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
        fp.write(model.code)
    return model


def convert_model(
    model: Model, to_format: Literal['generic', 'nlmixr', 'nonmem', 'rxode']
) -> Model:
    """Convert model to other format

    Note that the operation is not done inplace.

    Parameters
    ----------
    model : Model
        Model to convert
    to_format : {'generic', 'nlmixr', 'nonmem', 'rxode'}
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


def get_model_code(model: Model) -> str:
    """Get the model code of the underlying model language as a string

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
    >>> code = get_model_code(model)

    """
    return model.code


def print_model_code(model: Model) -> None:
    """Print the model code of the underlying model language to the console

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
    $DATA pheno.dta IGNORE=@
    $INPUT ID TIME AMT WGT APGR DV FA1 FA2
    $SUBROUTINE ADVAN1 TRANS2
    $ABBREV REPLACE ETA_CL=ETA(1)
    $ABBREV REPLACE ETA_VC=ETA(2)
    <BLANKLINE>
    $PK
    TVCL = THETA(1)*WGT
    TVV = THETA(2)*WGT
    IF(APGR.LT.5) TVV = TVV*(1 + THETA(3))
    CL = TVCL*EXP(ETA_CL)
    VC = TVV*EXP(ETA_VC)
    V = VC
    S1 = VC
    <BLANKLINE>
    $ERROR
    Y = F + F*EPS(1)
    <BLANKLINE>
    $THETA  (0,0.00469307) ; POP_CL
    $THETA  (0,1.00916) ; POP_VC
    $THETA  (-.99,.1) ; COVAPGR
    <BLANKLINE>
    $OMEGA  0.0309626 ; IIV_CL
    $OMEGA  0.031128 ; IIV_VC
    <BLANKLINE>
    $SIGMA  0.0130865  ; SIGMA
    <BLANKLINE>
    $ESTIMATION METHOD=1 INTERACTION MAXEVALS=99999
    $COVARIANCE UNCONDITIONAL PRINT=E
    $TABLE ID TIME DV CIPREDI PRED RES CWRES NOAPPEND NOPRINT ONEHEADER FILE=pheno.tab
    <BLANKLINE>

    """
    print(model.code)


def set_name(model: Model, new_name: str) -> Model:
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


def set_description(model: Model, new_description: str) -> Model:
    """Set description of model object

    Parameters
    ----------
    model : Model
        Pharmpy model
    new_description : str
        New description of model

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import set_description, load_example_model
    >>> model = load_example_model("pheno")
    >>> model.description
    'PHENOBARB SIMPLE MODEL'
    >>> model = set_description(model, "PHENOBARB run 2")
    >>> model.description
    'PHENOBARB run 2'

    """
    model = model.replace(description=new_description)
    return model


def bump_model_number(model: Model, path: Optional[Union[str, Path]] = None) -> Model:
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


def load_example_model(name: str) -> Model:
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
    TVCL = POP_CL⋅WGT
    TVV = POP_VC⋅WGT
          ⎧TVV⋅(COVAPGR + 1)  for APGR < 5
          ⎨
    TVV = ⎩       TVV          otherwise
               ETA_CL
    CL = TVCL⋅ℯ
              ETA_VC
    VC = TVV⋅ℯ
    V = VC
    S₁ = VC
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL(t)
        ────────────
    F =      S₁
    Y = EPS₁⋅F + F

    """
    available = ('moxo', 'pheno', 'pheno_linear')
    if name not in available:
        raise ValueError(f'Unknown example model {name}. Available examples: {available}')
    path = Path(__file__).resolve().parent.parent / 'internals' / 'example_models' / (name + '.mod')
    model = read_model(path)
    return model


def get_model_covariates(model: Model, strings: bool = False) -> Union[list[str], list[Expr]]:
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
    datasymbs = {Expr.symbol(s) for s in model.datainfo.names}

    odes = model.statements.ode_system

    # Consider statements that are dependencies of the ode system and y
    if odes:
        dose_comp = odes.dosing_compartments[0]
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
    covs = covs - {Expr.symbol(model.datainfo.id_column.name)}

    covs = list(covs)
    covs = list(sorted(covs, key=lambda x: x.name))  # sort to make order deterministic
    if strings:
        covs = [str(x) for x in covs]
    return covs


def print_model_symbols(model: Model) -> None:
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
    Thetas: POP_CL, POP_VC, COVAPGR
    Etas: ETA_CL, ETA_VC
    Omegas: IIV_CL, IIV_VC
    Epsilons: EPS₁
    Sigmas: SIGMA
    Variables: TVCL, TVV, TVV, CL, VC, V, S₁, F, Y
    Data columns: ID, TIME, AMT, WGT, APGR, DV, FA1, FA2

    """
    etas = [Expr.symbol(name).unicode() for name in model.random_variables.etas.names]
    epsilons = [Expr.symbol(name).unicode() for name in model.random_variables.epsilons.names]
    omegas = [Expr.symbol(n).unicode() for n in model.random_variables.etas.parameter_names]
    sigmas = [Expr.symbol(n).unicode() for n in model.random_variables.epsilons.parameter_names]
    thetas = []
    for param in model.parameters:
        if param.name not in model.random_variables.parameter_names:
            thetas.append(param.symbol.unicode())
    variables = []
    for sta in model.statements:
        if hasattr(sta, 'symbol'):
            variables.append(sta.symbol.unicode())
    s = f'Thetas: {", ".join(thetas)}\n'
    s += f'Etas: {", ".join(etas)}\n'
    s += f'Omegas: {", ".join(omegas)}\n'
    s += f'Epsilons: {", ".join(epsilons)}\n'
    s += f'Sigmas: {", ".join(sigmas)}\n'
    s += f'Variables: {", ".join(variables)}\n'
    s += f'Data columns: {", ".join(model.datainfo.names)}'
    print(s)


def get_config_path() -> Optional[str]:
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


def create_config_template() -> None:
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


def remove_unused_parameters_and_rvs(model: Model) -> Model:
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
                symb = Expr.symbol(name)
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


def rename_symbols(model: Model, new_names: Mapping[TSymbol, TSymbol]) -> Model:
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
    d = {Expr(key): Expr(val) for key, val in new_names.items()}

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


def filter_dataset(model: Model, expr: str) -> Model:
    """Filter dataset according to expr and return a model with the filtered dataset.

    Example: "DVID == 1" will filter the dataset so that only the rows with DVID = 1 remain.

    Parameters
    ----------
    model : Model
        Pharmpy model object
    expr : str
        expression for dataset query

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model.dataset
         ID   TIME   AMT  WGT  APGR    DV  FA1  FA2
    0     1    0.0  25.0  1.4   7.0   0.0  1.0  1.0
    1     1    2.0   0.0  1.4   7.0  17.3  0.0  0.0
    2     1   12.5   3.5  1.4   7.0   0.0  1.0  1.0
    3     1   24.5   3.5  1.4   7.0   0.0  1.0  1.0
    4     1   37.0   3.5  1.4   7.0   0.0  1.0  1.0
    ..   ..    ...   ...  ...   ...   ...  ...  ...
    739  59  108.3   3.0  1.1   6.0   0.0  1.0  1.0
    740  59  120.5   3.0  1.1   6.0   0.0  1.0  1.0
    741  59  132.3   3.0  1.1   6.0   0.0  1.0  1.0
    742  59  144.8   3.0  1.1   6.0   0.0  1.0  1.0
    743  59  146.8   0.0  1.1   6.0  40.2  0.0  0.0
    <BLANKLINE>
    [744 rows x 8 columns]
    >>> model = filter_dataset(model, 'WGT < 1.4')
    >>> model.dataset
         ID   TIME   AMT  WGT  APGR    DV  FA1  FA2
    42    4    0.0  18.6  0.9   6.0   0.0  1.0  1.0
    43    4    1.8   0.0  0.9   6.0  20.8  0.0  0.0
    44    4   12.0   2.3  0.9   6.0   0.0  1.0  1.0
    45    4   24.3   2.3  0.9   6.0   0.0  1.0  1.0
    46    4   35.8   2.3  0.9   6.0   0.0  1.0  1.0
    ..   ..    ...   ...  ...   ...   ...  ...  ...
    739  59  108.3   3.0  1.1   6.0   0.0  1.0  1.0
    740  59  120.5   3.0  1.1   6.0   0.0  1.0  1.0
    741  59  132.3   3.0  1.1   6.0   0.0  1.0  1.0
    742  59  144.8   3.0  1.1   6.0   0.0  1.0  1.0
    743  59  146.8   0.0  1.1   6.0  40.2  0.0  0.0
    <BLANKLINE>
    [400 rows x 8 columns]

    """
    original_dataset = get_and_check_dataset(model)
    try:
        new_dataset = original_dataset.query(expr)
        new_model = model.replace(
            dataset=new_dataset,
            description=model.description + ". Filtered dataset.",
            name=model.name + "_filtered",
        )
    except pandas.errors.UndefinedVariableError as e:
        raise ValueError(f'The expression `{expr}` is invalid: {e}')
    return new_model


def get_nested_model(model_1: Model, model_2: Model) -> Optional[Model]:
    """Return nested model from a pair of models

    Function to get a nested model from a pair of models, None
    if neither model is nested. A model is not considered nested if:

    1. They are the same model
    2. They have the same number of parameters
    3. The parameters of the reduced model is not a subset of
       the extended model
    4. The dosing or DV is changed

    Assumptions made:

    1. Parametrization is the same
    2. Parameter names are the same

    Parameters
    ----------
    model_1 : Model
        Pharmpy model object
    model_2 : Model
        Pharmpy model object

    Returns
    -------
    Model | None
        Pharmpy model object or None

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model_1 = load_example_model("pheno")
    >>> model_2 = add_peripheral_compartment(model_1)
    >>> model_2 = set_name(model_2, 'pheno_2')
    >>> nested = get_nested_model(model_1, model_2)
    >>> nested.name
    'pheno'
    """
    if model_1 == model_2:
        return None
    if model_1.dependent_variables != model_2.dependent_variables:
        return None

    models = sorted([model_1, model_2], key=lambda x: len(x.parameters))
    reduced, extended = models[0], models[1]
    params_reduced, params_extended = reduced.parameters, extended.parameters

    if len(params_reduced) == len(params_extended):
        return None
    if not set(params_reduced.symbols).issubset(params_extended.symbols):
        return None

    params_added = set(params_extended.symbols).difference(params_reduced.symbols)
    params_added = params_added.union(extended.random_variables.free_symbols).difference(
        reduced.random_variables.free_symbols
    )

    ode_extended = extended.statements.ode_system
    ode_reduced = reduced.statements.ode_system

    if ode_extended is None or ode_reduced is None:
        return None

    dosing_extended = [comp.doses for comp in ode_extended.dosing_compartments]
    dosing_reduced = [comp.doses for comp in ode_reduced.dosing_compartments]
    if dosing_extended != dosing_reduced:
        return None

    for name in ode_extended.compartment_names:
        comp_extended = ode_extended.find_compartment(name)
        assert comp_extended is not None
        if isinstance(comp_extended, Output):
            continue
        comp_reduced = ode_reduced.find_compartment(name)
        outflows = ode_extended.get_compartment_outflows(comp_extended)

        for comp_out_extended, rate_extended in outflows:
            if isinstance(comp_out_extended, Output):
                comp_out_reduced = Output()
            else:
                assert isinstance(comp_out_extended, Compartment)
                comp_out_reduced = ode_reduced.find_compartment(comp_out_extended.name)

            rate_extended = extended.statements.before_odes.full_expression(rate_extended)
            rate_reduced = ode_reduced.get_flow(comp_reduced, comp_out_reduced)
            rate_reduced = reduced.statements.before_odes.full_expression(rate_reduced)

            if not _is_collapsable(extended, rate_extended, rate_reduced, params_added):
                return None

    y_symb = list(extended.dependent_variables.keys())[0]
    y_extended = extended.statements.after_odes.full_expression(y_symb)
    y_reduced = reduced.statements.after_odes.full_expression(y_symb)

    if not _is_collapsable(extended, y_extended, y_reduced, params_added):
        return None

    return reduced


def _is_collapsable(extended, expr_extended, expr_reduced, params_added):
    if expr_extended == expr_reduced:
        return True
    symbs = params_added.intersection(expr_extended.free_symbols)
    subs_dict = dict()
    for symb in symbs:
        sub_values = []
        if symb in extended.parameters.symbols:
            param = extended.parameters[symb]
            if param.lower <= 0 or param.upper >= 0:
                sub_values.append(0)
            if param.lower <= 1 or param.upper >= 1:
                sub_values.append(1)
            if not sub_values:
                return False
        elif symb in extended.random_variables.symbols:
            # FIXME: check if normal distribution
            sub_values.extend([0, 1])

        subs_dict.update({symb: sub_values})

    permutations = itertools.product(*subs_dict.values())
    permutation_dicts = [dict(zip(subs_dict.keys(), permutation)) for permutation in permutations]

    for sub_dict in permutation_dicts:
        expr_simplified = expr_extended.subs(sub_dict)
        if expr_simplified == expr_reduced:
            return True
    return False
