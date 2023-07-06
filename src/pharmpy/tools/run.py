from __future__ import annotations

import importlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union, get_type_hints

import pharmpy
import pharmpy.results
import pharmpy.tools.modelfit
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.model import Model, Results
from pharmpy.modeling import calculate_aic, calculate_bic, check_high_correlations, read_model
from pharmpy.modeling.lrt import degrees_of_freedom as lrt_df
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.results import ModelfitResults, mfr
from pharmpy.tools.psn_helpers import create_results as psn_create_results
from pharmpy.workflows import Workflow, execute_workflow, split_common_options
from pharmpy.workflows.model_database import LocalModelDirectoryDatabase, ModelDatabase
from pharmpy.workflows.tool_database import ToolDatabase

from .external import parse_modelfit_results


def fit(
    model_or_models: Union[Model, List[Model]], tool: Optional[str] = None
) -> Union[ModelfitResults, List[ModelfitResults]]:
    """Fit models.

    Parameters
    ----------
    model_or_models : Model | list[Model]
        List of models or one single model
    tool : str
        Estimation tool to use. None to use default

    Return
    ------
    ModelfitResults | list[ModelfitResults]
        ModelfitResults for the model or models

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import fit
    >>> model = load_example_model("pheno")      # doctest: +SKIP
    >>> results = fit(model)      # doctest: +SKIP

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
            # FIXME model.database should be removed
            db_model = retrieve_models(model.database, model.name)[0]
        except (KeyError, AttributeError):
            db_model = None
        if (
            db_model
            and db_model.modelfit_results is not None
            and db_model == model
            and model.has_same_dataset_as(db_model)
        ):
            model = model.replace(modelfit_results=db_model.modelfit_results)
        else:
            kept.append(model)

    if kept:
        models = run_tool('modelfit', kept, tool=tool)

    return mfr(models[0]) if single else list(map(mfr, models))


def create_results(path: Union[str, Path], **kwargs) -> Results:
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


def read_results(path: Union[str, Path]) -> Results:
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


def run_tool(name: str, *args, **kwargs) -> Union[Model, List[Model], Tuple[Model], Results]:
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
    # NOTE The implementation of run_tool is split into those two functions to
    # allow for individual testing and mocking.
    tool = import_tool(name)
    return run_tool_with_name(name, tool, args, kwargs)


def import_tool(name: str):
    return importlib.import_module(f'pharmpy.tools.{name}')


def run_tool_with_name(
    name: str, tool, args: Sequence, kwargs: Mapping[str, Any]
) -> Union[Model, List[Model], Tuple[Model], Results]:
    # FIXME: workaround until ModelfitResults is disentangled with
    #  Model object
    if 'model' in kwargs and 'results' in kwargs:
        model = kwargs['model']
        res = kwargs['results']
        if isinstance(model, Model) and isinstance(res, ModelfitResults):
            model = model.replace(modelfit_results=res)
            kwargs['model'] = model

    common_options, tool_options = split_common_options(kwargs)

    create_workflow = tool.create_workflow

    dispatcher, tool_database = _get_run_setup(common_options, name)

    tool_params = inspect.signature(create_workflow).parameters
    tool_param_types = get_type_hints(create_workflow)

    tool_metadata = _create_metadata(
        database=tool_database,
        dispatcher=dispatcher,
        tool_name=name,
        tool_params=tool_params,
        tool_param_types=tool_param_types,
        args=args,
        tool_options=tool_options,
        common_options=common_options,
    )

    tool_database.store_metadata(tool_metadata)

    if validate_input := getattr(tool, 'validate_input', None):
        validate_input(*args, **tool_options)

    wf: Workflow = create_workflow(*args, **tool_options)
    assert wf.name == name

    res = execute_workflow(wf, dispatcher=dispatcher, database=tool_database)
    assert name == 'modelfit' or isinstance(res, Results)

    tool_metadata = _update_metadata(tool_metadata, res)
    tool_database.store_metadata(tool_metadata)

    return res


def _create_metadata(
    database: ToolDatabase,
    dispatcher,
    tool_name: str,
    tool_params,
    tool_param_types,
    args: Sequence,
    tool_options: Mapping[str, Any],
    common_options: Mapping[str, Any],
):
    tool_metadata = _create_metadata_tool(
        database, tool_name, tool_params, tool_param_types, args, tool_options
    )
    setup_metadata = _create_metadata_common(database, dispatcher, tool_name, common_options)
    tool_metadata['common_options'] = setup_metadata

    return tool_metadata


def _update_metadata(tool_metadata, res):
    # FIXME Make metadata immutable
    tool_metadata['stats']['end_time'] = _now()
    return tool_metadata


def resume_tool(path: str):
    """Resume tool workflow from tool database path

    Parameters
    ----------
    path : str
        The path to the tool database

    Return
    ------
    Results
        Results object for tool

    Examples
    --------
    >>> from pharmpy.modeling import * # doctest: +SKIP
    >>> res = resume_tool("resmod_dir1") # doctest: +SKIP

    """

    dispatcher, tool_database = _get_run_setup_from_metadata(path)

    tool_metadata = tool_database.read_metadata()
    tool_name = tool_metadata['tool_name']

    tool = importlib.import_module(f'pharmpy.tools.{tool_name}')

    create_workflow = tool.create_workflow

    tool_params = inspect.signature(create_workflow).parameters
    tool_param_types = get_type_hints(create_workflow)

    tool_options = _parse_tool_options_from_json_metadata(
        tool_metadata, tool_params, tool_param_types, tool_database
    )

    args, kwargs = _parse_args_kwargs_from_tool_options(tool_params, tool_options)

    if validate_input := getattr(tool, 'validate_input', None):
        validate_input(*args, **kwargs)

    wf: Workflow = create_workflow(*args, **kwargs)
    assert wf.name == tool_name

    res = execute_workflow(wf, dispatcher=dispatcher, database=tool_database)
    assert tool_name == 'modelfit' or isinstance(res, Results)

    tool_metadata = _update_metadata(tool_metadata, res)
    tool_database.store_metadata(tool_metadata)

    return res


def _parse_tool_options_from_json_metadata(
    tool_metadata,
    tool_params,
    tool_param_types,
    tool_database,
):
    tool_options = tool_metadata['tool_options']
    # NOTE Load models to memory
    for model_key in _input_model_param_keys(tool_params, tool_param_types):
        model_metadata = tool_options.get(model_key)
        if model_metadata is None:
            raise ValueError(
                f'Cannot resume run because model argument "{model_key}" cannot be restored.'
            )

        assert model_metadata['__class__'] == 'Model'
        model_name = model_metadata['arg_name']
        db_name = model_metadata['db_name']

        db: ModelDatabase = tool_database.model_database
        try:
            model = db.retrieve_model(db_name)
            model = model.replace(name=model_name)
            res = db.retrieve_modelfit_results(db_name)
            model = model.replace(modelfit_results=res)
        except KeyError:
            raise ValueError(
                f'Cannot resume run because model argument "{model_key}" ({model_name}) cannot be restored.'
            )
        tool_options = tool_options.copy()
        tool_options[model_key] = model

    # NOTE Load results to memory
    for results_key in _results_param_keys(tool_params, tool_param_types):
        results_json = tool_options.get(results_key)
        if results_json is not None:
            tool_options = tool_options.copy()
            tool_options[results_key] = pharmpy.results.read_results(results_json)

    return tool_options


def _parse_args_kwargs_from_tool_options(tool_params, tool_options):
    args = []
    kwargs = {}
    for p in tool_params.values():
        # Positional args
        if p.default == p.empty:
            args.append(tool_options[p.name])
        # Named args
        else:
            if p.name in tool_options.keys():
                kwargs[p.name] = tool_options[p.name]

    return args, kwargs


def _create_metadata_tool(
    database: ToolDatabase,
    tool_name: str,
    tool_params,
    tool_param_types,
    args: Sequence,
    kwargs: Mapping[str, Any],
):
    # FIXME: add config file dump, estimation tool etc.
    tool_metadata = {
        'pharmpy_version': pharmpy.__version__,
        'tool_name': tool_name,
        'stats': {'start_time': _now()},
        'tool_options': {},
    }

    for i, p in enumerate(tool_params.values()):
        try:
            name, value = p.name, args[i]
        except IndexError:
            try:
                name, value = p.name, kwargs[p.name]
            except KeyError:
                if p.default == p.empty:
                    # Positional args
                    raise ValueError(f'{tool_name}: \'{p.name}\' was not set')
                else:
                    # Named args
                    name, value = p.name, p.default

        tool_metadata['tool_options'][name] = value

    if tool_name != 'modelfit':
        db = database.model_database
        for key, arg_name, db_name in _store_input_models(
            db, tool_params, tool_param_types, args, kwargs
        ):
            tool_metadata['tool_options'][key] = {
                '__class__': 'Model',
                'arg_name': arg_name,
                'db_name': db_name,
            }

    return tool_metadata


def _create_metadata_common(
    database: ToolDatabase, dispatcher, toolname: Optional[str], common_options: Mapping[str, Any]
):
    setup_metadata = {}
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


def _store_input_models(
    db: ModelDatabase, params, types, args: Sequence, kwargs: Mapping[str, Any]
):
    for param_key, model in _input_models(params, types, args, kwargs):
        input_model_name = f'input_{param_key}'
        _store_input_model(db, model, input_model_name)
        yield param_key, model.name, input_model_name


def _filter_params(kind, params, types):
    for i, param_key in enumerate(params):
        param = params[param_key]
        param_type = types.get(param_key)
        if param_type in (kind, Optional[kind]):
            # NOTE We do not handle *{param_key}, or **{param_key}
            assert param.kind != param.VAR_POSITIONAL
            assert param.kind != param.VAR_KEYWORD
            yield i, param_key


def _input_model_param_keys(params, types):
    for _, param_key in _filter_params(Model, params, types):
        yield param_key


def _results_param_keys(params, types):
    for _, param_key in _filter_params(ModelfitResults, params, types):
        yield param_key


def _input_models(params, types, args: Sequence, kwargs: Mapping[str, Any]):
    for i, param_key in _filter_params(Model, params, types):
        model = args[i] if i < len(args) else kwargs.get(param_key)
        # NOTE We do not handle missing optional models
        assert model is not None
        yield param_key, model


def _store_input_model(db: ModelDatabase, model: Model, name: str):
    model_copy = model.replace(name=name)
    with db.transaction(model_copy) as txn:
        txn.store_model()
        txn.store_modelfit_results()


def _now():
    return datetime.now().astimezone().isoformat()


def _get_run_setup(common_options, toolname) -> Tuple[Any, ToolDatabase]:
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


def retrieve_models(
    source: Union[str, Path, Results, ToolDatabase, ModelDatabase],
    names: Optional[List[str]] = None,
) -> List[Model]:
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
        try:
            db_tool = getattr(source, 'tool_database')
            db = db_tool.model_database
        except AttributeError:
            raise ValueError(
                f'Results type \'{source.__class__.__name__}\' does not serialize tool database'
            )
    elif isinstance(source, ToolDatabase):
        db = source.model_database
    elif isinstance(source, ModelDatabase):
        db = source
    else:
        raise NotImplementedError(f'Not implemented for type \'{type(source)}\'')
    names_all: List[str] = db.list_models()
    if names is None:
        names = names_all
    diff = set(names).difference(names_all)
    if diff:
        raise ValueError(f'Models {diff} not in database')
    models = [db.retrieve_model(name) for name in names]
    return models


def retrieve_final_model(res: Results) -> Model:
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
    try:
        final_model_name = getattr(res, 'final_model_name')
    except AttributeError:
        raise ValueError('Attribute \'final_model_name\' is missing from results object')

    if final_model_name is None:
        raise ValueError('Attribute \'final_model_name\' is None')

    return retrieve_models(res, names=[final_model_name])[0]


def print_fit_summary(model: Model):
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

    res = mfr(model)

    print_header("Parameter estimation status", first=True)
    print_fmt("Minimization successful", bool_ok_error(res.minimization_successful))
    print_fmt("No rounding errors", bool_ok_error(res.termination_cause != 'rounding_errors'))
    ofv = res.ofv
    assert ofv is not None
    print_fmt("Objective function value", round(ofv, 1))

    print_header("Parameter uncertainty status")
    cov_run = model.estimation_steps[-1].cov
    print_fmt("Covariance step run", bool_yes_no(cov_run))

    if cov_run:
        condno = round(np.linalg.cond(res.correlation_matrix), 1)
        print_fmt("Condition number", condno)
        print_fmt("Condition number < 1000", bool_ok_error(condno < 1000))
        cor = res.correlation_matrix
        assert cor is not None
        hicorr = check_high_correlations(model, cor)
        print_fmt("No correlations arger than 0.9", bool_ok_error(hicorr.empty))

    print_header("Parameter estimates")
    pe = res.parameter_estimates
    if cov_run:
        se = res.standard_errors
        assert se is not None
        rse = se / pe
        rse.name = 'RSE'
        df = pd.concat([pe, se, rse], axis=1)
    else:
        df = pd.concat([pe], axis=1)
    print(df)


def write_results(results: Results, path: Union[str, Path], lzma: bool = False, csv: bool = False):
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
    path = normalize_user_given_path(path)
    if csv:
        results.to_csv(path)
    else:
        results.to_json(path, lzma=lzma)


def summarize_errors(results: Union[ModelfitResults, List[ModelfitResults]]) -> pd.DataFrame:
    """Summarize errors and warnings from one or multiple model runs.

    Summarize the errors and warnings found after running the model/models.

    Parameters
    ----------
    results : list, ModelfitResults
        List of ModelfitResults or single ModelfitResults

    Return
    ------
    pd.DataFrame
        A DataFrame of errors with model name, category (error or warning), and an int as index,
        an empty DataFrame if there were no errors or warnings found.

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import summarize_errors
    >>> model = load_example_model("pheno")
    >>> summarize_errors(model)      # doctest: +SKIP
    """
    # FIXME: have example with errors
    if isinstance(results, ModelfitResults):
        results = [results]

    idcs, rows = [], []

    for res in results:
        if res is not None and len(res.log.log) > 0:
            for i, entry in enumerate(res.log.log):
                idcs.append((res.name, entry.category, i))
                rows.append([entry.time, entry.message])

    index_names = ['model', 'category', 'error_no']
    col_names = ['time', 'message']
    index = pd.MultiIndex.from_tuples(idcs, names=index_names)

    if rows:
        df = pd.DataFrame(rows, columns=col_names, index=index)
    else:
        df = pd.DataFrame(columns=col_names, index=index)

    return df.sort_index()


def rank_models(
    base_model: Model,
    models: List[Model],
    errors_allowed: Optional[List[str]] = None,
    rank_type: str = 'ofv',
    cutoff: Optional[float] = None,
    bic_type: str = 'mixed',
) -> pd.DataFrame:
    """Ranks a list of models

    Ranks a list of models with a given ranking function

    Parameters
    ----------
    base_model : Model
        Base model to compare to
    models : list
        List of models
    errors_allowed : list or None
        List of errors that are allowed for ranking. Currently available is: rounding_errors and
        maxevals_exceeded. Default is None
    rank_type : str
        Name of ranking type. Available options are 'ofv', 'aic', 'bic', 'lrt' (OFV with LRT)
    cutoff : float or None
        Value to use as cutoff. If using LRT, cutoff denotes p-value. Default is None
    bic_type : str
        Type of BIC to calculate. Default is the mixed effects.

    Return
    ------
    pd.DataFrame
        DataFrame of the ranked models

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import rank_models
    >>> model_1 = load_example_model("pheno")
    >>> model_2 = load_example_model("pheno_linear")
    >>> rank_models(model_1, [model_2],
    ...             errors_allowed=['rounding_errors'],
    ...             rank_type='lrt') # doctest: +SKIP
    """
    models_all = [base_model] + models

    rank_values, delta_values = {}, {}
    models_to_rank = []

    ref_value = _get_rankval(base_model, rank_type, bic_type)
    model_dict = {model.name: model for model in models_all}

    # Filter on strictness
    for model in models_all:
        # Exclude OFV etc. if model was not successful
        if not model.modelfit_results or np.isnan(model.modelfit_results.ofv):
            continue
        if not model.modelfit_results.minimization_successful:
            if errors_allowed:
                if model.modelfit_results.termination_cause not in errors_allowed:
                    continue
                if np.isnan(model.modelfit_results.significant_digits):
                    continue
            else:
                continue

        rank_value = _get_rankval(model, rank_type, bic_type)
        if rank_type == 'lrt':
            parent = model_dict[model.parent_model]
            if cutoff is None:
                co = 0.05 if lrt_df(parent, model) >= 0 else 0.01
            elif isinstance(cutoff, tuple):
                co = cutoff[0] if lrt_df(parent, model) >= 0 else cutoff[1]
            else:
                assert isinstance(cutoff, (float, int))
                co = cutoff
            parent_ofv = np.nan if (mfr := parent.modelfit_results) is None else mfr.ofv
            model_ofv = np.nan if (mfr := model.modelfit_results) is None else mfr.ofv
            if not lrt_test(parent, model, parent_ofv, model_ofv, co):
                continue
        elif cutoff is not None:
            if ref_value - rank_value <= cutoff:
                continue

        # Add ranking value and model
        rank_values[model.name] = rank_value
        delta_values[model.name] = ref_value - rank_value
        models_to_rank.append(model)

    # Sort
    def _get_delta(model):
        if np.isnan(ref_value):
            return -rank_values[model.name]
        return delta_values[model.name]

    models_sorted = sorted(models_to_rank, key=_get_delta, reverse=True)

    # Create rank for models, if two have the same value they will have the same rank
    ranking = {}
    rank, count, prev = 0, 0, None
    for model in models_sorted:
        count += 1
        value = _get_delta(model)
        if value != prev:
            rank += count
            prev = value
            count = 0
        ranking[model.name] = rank

    rows = {}
    for model in models_all:
        delta, rank_value, rank = np.nan, np.nan, np.nan
        if model.name in ranking.keys():
            rank = ranking[model.name]
        if model.name in rank_values.keys():
            rank_value = rank_values[model.name]
        if model.name in delta_values.keys():
            delta = delta_values[model.name]

        rows[model.name] = (delta, rank_value, rank)

    if rank_type == 'lrt':
        rank_type_name = 'ofv'
    else:
        rank_type_name = rank_type

    index = pd.Index(rows.keys(), name='model')
    df = pd.DataFrame(
        rows.values(), index=index, columns=[f'd{rank_type_name}', f'{rank_type_name}', 'rank']
    )

    if np.isnan(ref_value):
        return df.sort_values(by=[f'{rank_type_name}'])
    else:
        return df.sort_values(by=[f'd{rank_type_name}'], ascending=False)


def _get_rankval(model, rank_type, bic_type):
    if not model.modelfit_results:
        return np.nan
    if rank_type in ['ofv', 'lrt']:
        return model.modelfit_results.ofv
    elif rank_type == 'aic':
        return calculate_aic(model, model.modelfit_results.ofv)
    elif rank_type == 'bic':
        return calculate_bic(model, model.modelfit_results.ofv, bic_type)
    else:
        raise ValueError('Unknown rank_type: must be ofv, lrt, aic, or bic')


def summarize_modelfit_results(
    results: Union[ModelfitResults, List[ModelfitResults]],
    include_all_estimation_steps: bool = False,
) -> pd.DataFrame:
    """Summarize results of model runs

    Summarize different results after fitting a model, includes runtime, ofv,
    and parameter estimates (with errors). If include_all_estimation_steps is False,
    only the last estimation step will be included (note that in that case, the
    minimization_successful value will be referring to the last estimation step, if
    last step is evaluation it will go backwards until it finds an estimation step
    that wasn't an evaluation).

    Parameters
    ----------
    results : list, ModelfitResults
        List of ModelfitResults or single ModelfitResults
    include_all_estimation_steps : bool
        Whether to include all estimation steps, default is False

    Return
    ------
    pd.DataFrame
        A DataFrame of modelfit results with model name and estmation step as index.

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import summarize_modelfit_results
    >>> model = load_example_model("pheno")
    >>> summarize_modelfit_results(model.modelfit_results) # doctest: +SKIP
                     description  minimization_successful ...        ofv  ... runtime_total  ...
    pheno PHENOBARB SIMPLE MODEL                     True ... 586.276056  ...           4.0  ...
    """
    if isinstance(results, ModelfitResults):
        results = [results]

    if results is None:
        raise ValueError('Option `results` is None')
    if all(res is None for res in results):
        raise ValueError('All input results are empty')

    summaries = []

    for res in results:
        if res is not None:
            summary = _get_model_result_summary(res, include_all_estimation_steps)
            summary.insert(0, 'description', res.description)
            summaries.append(summary)

    df = pd.concat(summaries)

    return df


def _get_model_result_summary(res, include_all_estimation_steps=False):
    if not include_all_estimation_steps:
        summary_dict = _summarize_step(res, -1)
        index = pd.Index([res.name], name='model')
        summary_df = pd.DataFrame(summary_dict, index=index)
    else:
        summary_dicts = []
        tuples = []
        for i in range(len(res.evaluation)):
            summary_dict = _summarize_step(res, i)
            is_evaluation = res.evaluation.iloc[i]
            if is_evaluation:
                run_type = 'evaluation'
            else:
                run_type = 'estimation'
            summary_dict = {'run_type': run_type, **summary_dict}
            summary_dicts.append(summary_dict)
            tuples.append((res.name, i + 1))
        index = pd.MultiIndex.from_tuples(tuples, names=['model', 'step'])
        summary_df = pd.DataFrame(summary_dicts, index=index)

    log_df = res.log.to_dataframe()

    no_of_errors = len(log_df[log_df['category'] == 'ERROR'])
    no_of_warnings = len(log_df[log_df['category'] == 'WARNING'])

    minimization_idx = summary_df.columns.get_loc('minimization_successful')
    summary_df.insert(loc=minimization_idx + 1, column='errors_found', value=no_of_errors)
    summary_df.insert(loc=minimization_idx + 2, column='warnings_found', value=no_of_warnings)

    return summary_df


def _summarize_step(res, i):
    summary_dict = {}

    if i >= 0:
        minsucc = res.minimization_successful_iterations.iloc[i]
    else:
        minsucc = res.minimization_successful

    if minsucc is not None:
        summary_dict['minimization_successful'] = minsucc
    else:
        summary_dict['minimization_successful'] = False

    if i == -1 and res.ofv_iterations is not None:
        i = max(res.ofv_iterations.index.get_level_values(0)) - 1

    summary_dict['ofv'] = _get_ofv(res, i)
    summary_dict['runtime_total'] = res.runtime_total
    summary_dict['estimation_runtime'] = _get_estimation_runtime(res, i)

    pe = _get_parameter_estimates(res, i)
    ses = res.standard_errors
    rses = res.relative_standard_errors

    for param in pe.index:
        summary_dict[f'{param}_estimate'] = pe[param]
        if ses is not None:
            summary_dict[f'{param}_SE'] = ses[param]
        if rses is not None:
            summary_dict[f'{param}_RSE'] = rses[param]

    return summary_dict


def _get_ofv(res, i):
    if res.ofv_iterations is None:
        return res.ofv
    return res.ofv_iterations[i + 1,].iloc[-1]


def _get_parameter_estimates(res, i):
    if res.parameter_estimates_iterations is None:
        return res.parameter_estimates
    return res.parameter_estimates_iterations.loc[i + 1,].iloc[-1]


def _get_estimation_runtime(res, i):
    if res.estimation_runtime_iterations is None:
        return res.runtime_total
    return res.estimation_runtime_iterations.iloc[i]


def read_modelfit_results(path: Union[str, Path]) -> ModelfitResults:
    """Read results from external tool for a model

    Parameters
    ----------
    path : Path or str
        Path to model file

    Return
    ------
    ModelfitResults
        Results object
    """
    path = Path(path)
    model = read_model(path)
    res = parse_modelfit_results(model, path)
    return res


def _get_run_setup_from_metadata(path):
    import pharmpy.workflows as workflows

    tool_database = workflows.default_tool_database(toolname=None, path=path, exist_ok=True)

    tool_metadata = tool_database.read_metadata()
    tool_name = tool_metadata['tool_name']
    common_options = tool_metadata['common_options']

    # TODO be more general
    dispatcher = getattr(workflows, common_options['dispatcher'].split('.')[-1])

    # TODO be more general
    assert common_options['database']['class'] == 'LocalDirectoryToolDatabase'
    assert common_options['database']['toolname'] == tool_name

    return dispatcher, tool_database


def load_example_modelfit_results(name: str):
    """Load the modelfit results of an example model

    Load the modelfit results of an example model built into Pharmpy

    Parameters
    ----------
    name : str
        Name of the model. Currently available models are "pheno" and "pheno_linear"

    Returns
    -------
    ModelfitResults
        Loaded modelfit results object

    Example
    -------
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> results.parameter_estimates
        PTVCL        0.004696
    PTVV         0.984258
    THETA_3      0.158920
    IVCL         0.029351
    IVV          0.027906
    SIGMA_1_1    0.013241
    Name: estimates, dtype: float64

    """
    available = ('moxo', 'pheno', 'pheno_linear')
    if name not in available:
        raise ValueError(f'Unknown example model {name}. Available examples: {available}')
    path = Path(__file__).resolve().parent.parent / 'modeling' / 'example_models' / (name + '.mod')
    res = read_modelfit_results(path)
    return res
