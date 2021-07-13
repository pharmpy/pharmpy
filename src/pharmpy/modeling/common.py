"""Common modeling pipeline elements
:meta private:
"""

from io import StringIO

from pharmpy.model_factory import Model


def read_model(path):
    """Read model from file"""
    model = Model(path)
    return model


def read_model_from_string(code):
    """Read model directly from the model code in a string"""
    model = Model(StringIO(code))
    return model


def write_model(model, path='', force=False):
    """Write model to file"""
    model.write(path=path, force=force)
    return model


def convert_model(model, to):
    """Convert model to other format

    Parameters
    ----------
    model : Model
        Model to convert
    to : str
        Name of format to convert into. Currently supported 'nlmixr'

    Results
    -------
    Model
        New model object with new underlying model format.
    """
    if to != 'nlmixr':
        raise ValueError(f"Unknown format {to}: supported format is 'nlmixr'")
    import pharmpy.plugins.nlmixr.model as nlmixr

    new = nlmixr.convert_model(model)
    return new


def update_source(model):
    """Update source

    Let the code of the underlying source language be updated to reflect
    changes in the model object.
    """
    model.update_source()
    return model


def copy_model(model):
    """Copies model to a new model object"""
    return model.copy()


def set_name(model, name):
    """Sets name of model object"""
    model.name = name
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
    model : Model
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
    model : Model
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
    values : list or int
        one value or a list of values (must be equal to number of parameter_names)

    Returns
    -------
    model : Model
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
    values : list or int
        one value or a list of values (must be equal to number of parameter_names)

    Returns
    -------
    model : Model
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


def add_estimation_step(model, method, interaction=True, options={}, est_idx=None):
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
    est_idx : int
        index of estimation step, default is None (adds step last)

    Returns
    -------
    model : Model
    """
    model.add_estimation_step(method, interaction, False, options, est_idx)
    return model


def remove_estimation_step(model, est_idx):
    """Remove estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model
    est_idx : int
        index of estimation step to remove

    Returns
    -------
    model : Model
    """
    model.remove_estimation_step(est_idx)
    return model
