import importlib
import inspect
from datetime import datetime
from pathlib import Path

import pharmpy.model
import pharmpy.results
import pharmpy.tools.modelfit
import pharmpy.tools.psn_helpers
from pharmpy.utils import normalize_user_given_path
from pharmpy.workflows import execute_workflow, split_common_options

from .common import read_model_from_database


def fit(models, tool=None):
    """Fit models.

    Parameters
    ----------
    models : list
        List of models or one single model
    tool : str
        Estimation tool to use. None to use default

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
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
        except (KeyError, AttributeError):
            db_model = None
        if (
            db_model
            and db_model.modelfit_results is not None
            and db_model == model
            and model.has_same_dataset_as(db_model)
        ):
            model.modelfit_results = db_model.modelfit_results
        else:
            kept.append(model)
    if kept:
        run_tool('modelfit', kept, tool=tool)
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
    path = normalize_user_given_path(path)
    res = pharmpy.tools.psn_helpers.create_results(path, **kwargs)
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
    path = normalize_user_given_path(path)
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

    tool_params = inspect.signature(tool.create_workflow).parameters
    tool_metadata = _create_metadata_tool(name, tool_params, tool_options, args)

    wf = tool.create_workflow(*args, **tool_options)

    dispatcher, database = _get_run_setup(common_options, wf.name)
    setup_metadata = _create_metadata_common(common_options, dispatcher, database, wf.name)
    tool_metadata['common_options'] = setup_metadata
    database.store_metadata(tool_metadata)

    res = execute_workflow(wf, dispatcher=dispatcher, database=database)

    tool_metadata['stats']['end_time'] = _now()
    database.store_metadata(tool_metadata)

    return res


def _now():
    return datetime.now().astimezone().isoformat()


def _create_metadata_tool(tool_name, tool_params, tool_options, args):
    # FIXME: add config file dump, Pharmpy version, estimation tool etc.
    tool_metadata = {
        'tool_name': tool_name,
        'stats': {'start_time': _now()},
        'tool_options': dict(),
    }

    for i, p in enumerate(tool_params.values()):
        # Positional args
        if p.default == p.empty:
            try:
                name, value = p.name, args[i]
            except IndexError:
                try:
                    name, value = p.name, tool_options[p.name]
                except KeyError:
                    raise ValueError(f'{tool_name}: \'{p.name}\' was not set')
        # Named args
        else:
            if p.name in tool_options.keys():
                name, value = p.name, tool_options[p.name]
            else:
                name, value = p.name, p.default
        if isinstance(value, pharmpy.Model):
            value = str(value)  # FIXME: better model representation
        tool_metadata['tool_options'][name] = value

    return tool_metadata


def _create_metadata_common(common_options, dispatcher, database, toolname):
    setup_metadata = dict()
    setup_metadata['dispatcher'] = dispatcher.__name__
    # FIXME: naming of workflows/tools should be consistent (db and input name of tool)
    setup_metadata['database'] = {
        'class': type(database).__name__,
        'toolname': toolname,
        'path': str(database.path),
    }
    for key, value in common_options.items():
        if key not in setup_metadata.keys():
            if isinstance(value, Path):
                value = str(value)
            setup_metadata[str(key)] = value

    return setup_metadata


def _get_run_setup(common_options, toolname):
    try:
        dispatcher = common_options['dispatcher']
    except KeyError:
        from pharmpy.workflows import default_dispatcher

        dispatcher = default_dispatcher

    try:
        database = common_options['database']
    except KeyError:
        from pharmpy.workflows import default_tool_database

        if 'path' in common_options.keys():
            path = common_options['path']
        else:
            path = None
        database = default_tool_database(
            toolname=toolname, path=path, exist_ok=common_options.get('resume', False)
        )  # TODO: database -> tool_database

    return dispatcher, database
