import importlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pharmpy
import pharmpy.results
import pharmpy.tools.modelfit
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results
from pharmpy.modeling import check_high_correlations, copy_model, read_model_from_database
from pharmpy.tools.psn_helpers import create_results as psn_create_results
from pharmpy.utils import normalize_user_given_path
from pharmpy.workflows import execute_workflow, split_common_options
from pharmpy.workflows.model_database import LocalModelDirectoryDatabase, ModelDatabase
from pharmpy.workflows.tool_database import ToolDatabase


def fit(
    model_or_models: Union[Model, List[Model]], tool: Optional[str] = None
) -> Union[Model, List[Model]]:
    """Fit models.

    Parameters
    ----------
    model_or_models : Model | list[Model]
        List of models or one single model
    tool : str
        Estimation tool to use. None to use default

    Return
    ------
    Model | list[Model]
        Input model or models with model fit results

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import fit
    >>> model = load_example_model("pheno")      # doctest: +SKIP
    >>> fit(model)      # doctest: +SKIP

    See also
    --------
    run_tool

    """
    single, models = (
        (True, [model_or_models])
        if isinstance(model_or_models, Model)
        else (False, model_or_models)
    )

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

    return models[0] if single else models


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
    >>> from pharmpy.tools import create_results
    >>> res = create_results("frem_dir1")   # doctest: +SKIP

    See also
    --------
    read_results

    """
    path = normalize_user_given_path(path)
    res = psn_create_results(path, **kwargs)
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
    >>> from pharmpy.tools import read_results
    >>> res = read_results("results.json")     # doctest: +SKIP

    See also
    --------
    create_results

    """
    path = normalize_user_given_path(path)
    res = pharmpy.results.read_results(path)
    return res


def run_tool(name, *args, **kwargs) -> Union[Model, List[Model], Tuple[Model], Results]:
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
    >>> from pharmpy.tools import run_tool # doctest: +SKIP
    >>> res = run_tool("ruvsearch", model)   # doctest: +SKIP

    """
    tool = importlib.import_module(f'pharmpy.tools.{name}')
    common_options, tool_options = split_common_options(kwargs)

    tool_params = inspect.signature(tool.create_workflow).parameters
    tool_metadata = _create_metadata_tool(name, tool_params, tool_options, args)

    if validate_input := getattr(tool, 'validate_input', None):
        validate_input(*args, **tool_options)

    wf = tool.create_workflow(*args, **tool_options)

    dispatcher, database = _get_run_setup(common_options, wf.name)
    setup_metadata = _create_metadata_common(common_options, dispatcher, database, wf.name)
    tool_metadata['common_options'] = setup_metadata
    database.store_metadata(tool_metadata)

    if name != 'modelfit':
        _store_input_models(list(args) + list(kwargs.items()), database)

    res = execute_workflow(wf, dispatcher=dispatcher, database=database)
    assert name == 'modelfit' or isinstance(res, Results)

    tool_metadata['stats']['end_time'] = _now()
    database.store_metadata(tool_metadata)

    return res


def _store_input_models(args, database):
    input_models = _get_input_models(args)

    if len(input_models) == 1:
        _create_input_model(input_models[0], database)
    else:
        for i, model in enumerate(input_models, 1):
            _create_input_model(model, database, number=i)


def _get_input_models(args):
    input_models = []
    for arg in args:
        if isinstance(arg, Model):
            input_models.append(arg)
        else:
            arg_as_list = [a for a in arg if isinstance(a, Model)]
            input_models.extend(arg_as_list)
    return input_models


def _create_input_model(model, tool_db, number=None):
    input_name = 'input_model'
    if number is not None:
        input_name += str(number)
    model_copy = copy_model(model, input_name)
    with tool_db.model_database.transaction(model_copy) as txn:
        txn.store_model()
        txn.store_modelfit_results()


def _now():
    return datetime.now().astimezone().isoformat()


def _create_metadata_tool(tool_name, tool_params, tool_options, args):
    # FIXME: add config file dump, estimation tool etc.
    tool_metadata = {
        'pharmpy_version': pharmpy.__version__,
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
        if isinstance(value, Model):
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


def retrieve_models(source, names=None):
    """Retrieve models after a tool run

    Any models created and run by the tool can be
    retrieved.

    Parameters
    ----------
    source : str, Path, Results, ToolDatabase, ModelDatabase
        Source where to find models. Can be a path (as str or Path), a results object, or a
        ToolDatabase/ModelDatabase
    names : list
        List of names of the models to retrieve or None for all

    Return
    ------
    list
        List of retrieved model objects

    Examples
    --------
    >>> from pharmpy.tools import retrieve_models
    >>> tooldir_path = 'path/to/tool/directory'
    >>> models = retrieve_models(tooldir_path, names=['run1'])      # doctest: +SKIP

    See also
    --------
    retrieve_final_model

    """
    if isinstance(source, Path) or isinstance(source, str):
        path = Path(source)
        # FIXME: Should be using metadata to know how to init databases
        db = LocalModelDirectoryDatabase(path / 'models')
    elif isinstance(source, Results):
        if hasattr(source, 'tool_database'):
            db = source.tool_database.model_database
        else:
            raise ValueError(
                f'Results type \'{source.__class__.__name__}\' does not serialize tool database'
            )
    elif isinstance(source, ToolDatabase):
        db = source.model_database
    elif isinstance(source, ModelDatabase):
        db = source
    else:
        raise NotImplementedError(f'Not implemented for type \'{type(source)}\'')
    names_all = db.list_models()
    if names is None:
        names = names_all
    diff = set(names).difference(names_all)
    if diff:
        raise ValueError(f'Models {diff} not in database')
    models = [db.retrieve_model(name) for name in names]
    return models


def retrieve_final_model(res):
    """Retrieve final model from a result object

    Parameters
    ----------
    res : Results
        A results object

    Return
    ------
    Model
        Reference to final model

    Examples
    --------
    >>> from pharmpy.tools import read_results, retrieve_final_model
    >>> res = read_results("results.json")     # doctest: +SKIP
    >>> model = retrieve_final_model(res)      # doctest: +SKIP

    See also
    --------
    retrieve_models

    """
    if res.final_model_name is None:
        raise ValueError('Attribute \'final_model_name\' is None')
    return retrieve_models(res, names=[res.final_model_name])[0]


def print_fit_summary(model):
    """Print a summary of the model fit

    Parameters
    ----------
    model : Model
        Pharmpy model object
    """

    def bool_ok_error(x):
        return "OK" if x else "ERROR"

    def bool_yes_no(x):
        return "YES" if x else "NO"

    def print_header(text, first=False):
        if not first:
            print()
        print(text)
        print("-" * len(text))

    def print_fmt(text, result):
        print(f"{text:33} {result}")

    res = model.modelfit_results

    print_header("Parameter estimation status", first=True)
    print_fmt("Minimization successful", bool_ok_error(res.minimization_successful))
    print_fmt("No rounding errors", bool_ok_error(res.termination_cause != 'rounding_errors'))
    print_fmt("Objective function value", round(res.ofv, 1))

    print_header("Parameter uncertainty status")
    cov_run = model.estimation_steps[-1].cov
    print_fmt("Covariance step run", bool_yes_no(cov_run))

    if cov_run:
        condno = round(np.linalg.cond(res.correlation_matrix), 1)
        print_fmt("Condition number", condno)
        print_fmt("Condition number < 1000", bool_ok_error(condno < 1000))
        cor = model.modelfit_results.correlation_matrix
        hicorr = check_high_correlations(model, cor)
        print_fmt("No correlations arger than 0.9", bool_ok_error(hicorr.empty))

    print_header("Parameter estimates")
    pe = res.parameter_estimates
    if cov_run:
        se = res.standard_errors
        rse = se / pe
        rse.name = 'RSE'
        df = pd.concat([pe, se, rse], axis=1)
    else:
        df = pd.concat([pe], axis=1)
    print(df)


def write_results(results, path, lzma=False, csv=False):
    """Write results object to json (or csv) file

    Note that the csv-file cannot be read into a results object again.

    Parameters
    ----------
    results : Results
        Pharmpy results object
    path : Path
        Path to results file
    lzma : bool
        True for lzma compression. Not applicable to csv file
    csv : bool
        Save as csv file
    """
    if csv:
        results.to_csv(path)
    else:
        results.to_json(path, lzma=lzma)
