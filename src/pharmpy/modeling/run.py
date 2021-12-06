import importlib

import pharmpy.model
import pharmpy.results
import pharmpy.tools.common
import pharmpy.tools.modelfit
from pharmpy.workflows import execute_workflow, split_common_options

from .common import read_model_from_database


def fit(models):
    """Fit models.

    Parameters
    ----------
    models : list
        List of models or one single model

    Return
    ------
    Model
        Reference to same model

    Examples
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> fit(model)      # doctest: +SKIP

    See also
    --------
    run_tool

    """
    if isinstance(models, pharmpy.model.Model):
        models = [models]
        single = True
    else:
        single = False
    kept = []
    # Do not fit model if already fit
    for model in models:
        try:
            db_model = read_model_from_database(model.name, database=model.database)
        except KeyError:
            db_model = None
        if db_model and db_model.modelfit_results is not None and db_model == model:
            model.modelfit_results = db_model.modelfit_results
        else:
            kept.append(model)
    if kept:
        run_tool('modelfit', kept)
    if single:
        return models[0]
    else:
        return models


def create_results(path, **kwargs):
    """Create/recalculate results object given path to run directory

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

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> res = create_results("frem_dir1")   # doctest: +SKIP

    See also
    --------
    read_results

    """
    res = pharmpy.tools.common.create_results(path, **kwargs)
    return res


def read_results(path):
    """Read results object from file

    Parameters
    ----------
    path : str, Path
        Path to results file

    Return
    ------
    Results
        Results object for tool

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> res = read_resuts("results.json")     # doctest: +SKIP

    See also
    --------
    create_results

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

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> res = run_tool("resmod", model)   # doctest: +SKIP

    """
    tool = importlib.import_module(f'pharmpy.tools.{name}')
    common_options, tool_options = split_common_options(kwargs)
    wf = tool.create_workflow(*args, **tool_options)
    res = execute_workflow(wf, **common_options)
    return res
