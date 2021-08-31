"""Common modeling pipeline elements
:meta private:
"""

from io import StringIO
from pathlib import Path

from pharmpy.estimation import EstimationMethod
from pharmpy.model_factory import Model


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
    >>> read_model("/home/run1.mod")    # doctest: +SKIP

    See also
    --------
    read_model_from_string : Read model from string

    """
    model = Model(path)
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
        Read model object

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

    """
    model = Model(StringIO(code))
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
        Name of format to convert into. Currently supported 'nlmixr'

    Returns
    -------
    Model
        New model object with new underlying model format.

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, convert_model
    >>> model = load_example_model("pheno")
    >>> converted_model = convert_model(model, "nlmixr")    # doctest: +SKIP

    """
    if to_format != 'nlmixr':
        raise ValueError(f"Unknown format {to_format}: supported format is 'nlmixr'")
    import pharmpy.plugins.nlmixr.model as nlmixr

    new = nlmixr.convert_model(model)
    return new


def update_source(model):
    """Update source

    Let the code of the underlying source language be updated to reflect
    changes in the model object.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, fix_parameters, update_source
    >>> model = load_example_model("pheno")
    >>> fix_parameters(model, ['THETA(1)'])  # doctest: +ELLIPSIS
    <...>
    >>> update_source(model)  # doctest: +ELLIPSIS
    <...>
    >>> print(str(model).splitlines()[22])
    $THETA (0,0.00469307) FIX ; PTVCL

    """
    model.update_source()
    return model


def copy_model(model):
    """Copies model to a new model object

    Parameters
    ----------
    model : Model
        Pharmpy model

    Results
    -------
    Model : A copy of the input model

    Example
    -------
    >>> from pharmpy.modeling import copy_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> model_copy = copy_model(model)

    """
    return model.copy()


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
    Model : Reference to the same model object

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
    model : Model
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
    Model : Reference to the same model

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
    Model : Reference to the same model object

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
    Model : Reference to the same model object

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
    Model : Reference to the same model

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


def set_estimation_step(model, method, interaction=True, options={}, est_idx=0):
    """Set estimation step

    Sets estimation step for a model. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : str
        estimation method to change to
    interaction : bool
        whether to use interaction or not, default is true
    options : dict
        any additional options. Note that this removes old options
    est_idx : int
        index of estimation step, default is 0 (first estimation step)

    Returns
    -------
    model : Model
    """
    model.estimation_steps[est_idx].method = method
    model.estimation_steps[est_idx].interaction = interaction
    if options:
        model.estimation_steps[est_idx].options = options
    return model


def add_estimation_step(model, method, interaction=True, options=None, idx=None):
    """Add estimation step

    Adds estimation step for a model in a given index. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : str
        estimation method to change to
    interaction : bool
        whether to use interaction or not, default is true
    options : dict
        any additional options. Note that this removes old options
    idx : int
        index of estimation step, default is None (adds step last)

    Returns
    -------
    model : Model
    """
    if options is None:
        options = {}

    meth = EstimationMethod(method, interaction=interaction, options=options)
    if isinstance(idx, int):
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
        index of estimation step to remove

    Returns
    -------
    model : Model
    """
    del model.estimation_steps[idx]
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
    Model : Loaded model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model
    >>> model = load_example_model("pheno")
    >>> print(model.statements)
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
