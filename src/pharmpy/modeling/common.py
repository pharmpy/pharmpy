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


def update_source(model):
    """Update source

    Let the code of the underlying source language be updated to reflect
    changes in the model object.
    """
    model.update_source()
    return model


def fix_parameters(model, parameter_names):
    """Fix parameters

    Fix all listed parameters

    Parameters
    ----------
    model: Model
    parameter_names: list or str
        one parameter name or a list of parameter names

    Returns
    -------
    model: Model
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
    model: Model
    parameter_names: list or str
        one parameter name or a list of parameter names

    Returns
    -------
    model: Model
    """
    if isinstance(parameter_names, str):
        d = {parameter_names: False}
    else:
        d = {name: False for name in parameter_names}
    model.parameters.fix = d
    return model
