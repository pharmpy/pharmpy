import importlib

import pharmpy.model
import pharmpy.results
import pharmpy.tools.common
import pharmpy.tools.modelfit
from pharmpy.execute import execute_workflow


# TODO: elaborate documentation
def fit(models):
    """Fit models.

    Parameters
    ----------
    models : list
        List of models

    Return
    ------
    Model
        Reference to same model
    """
    if isinstance(models, pharmpy.model.Model):
        models = [models]
        single = True
    else:
        single = False
    tool = pharmpy.tools.modelfit.Modelfit(models)
    tool.run()
    if single:
        return models[0]
    else:
        return models


# TODO: elaborate documentation
def create_results(path, **kwargs):
    """Create results object

    Parameters
    ----------
    path : str, Path
        Path to run directory
    kwargs
        Arguments to pass to tool specific create results function

    Returns
    -------
    Results
        Results object for tool
    """
    res = pharmpy.tools.common.create_results(path, **kwargs)
    return res


# TODO: elaborate documentation
def read_results(path):
    """Read results object

    Parameters
    ----------
    path : str, Path
        Path to results file

    Return
    ------
    Results
        Results object for tool
    """
    res = pharmpy.results.read_results(path)
    return res


def run_tool(name, *args, **kwargs):
    """Run tool workflow

    Parameters
    ----------
    name : str
        Name of tool to run
    args
        Arguments to pass to tool
    kwargs
        Arguments to pass to tool

    Return
    ------
    Results
        Results object for tool
    """
    tool = importlib.import_module(f'pharmpy.tools.{name}')
    wf = tool.create_workflow(*args, **kwargs)
    res = execute_workflow(wf)
    return res
