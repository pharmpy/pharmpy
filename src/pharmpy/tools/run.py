from __future__ import annotations

import importlib
import inspect
import re
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, get_type_hints

import pharmpy
import pharmpy.tools.modelfit
import pharmpy.workflows.results
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.model import Model
from pharmpy.modeling import (
    calculate_aic,
    calculate_bic,
    check_high_correlations,
    check_parameters_near_bounds,
    get_omegas,
    get_sigmas,
    get_thetas,
    read_model,
)
from pharmpy.modeling.lrt import degrees_of_freedom as lrt_df
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.tools.psn_helpers import create_results as psn_create_results
from pharmpy.workflows import Results, Workflow, execute_workflow, split_common_options
from pharmpy.workflows.context import Context, LocalDirectoryContext
from pharmpy.workflows.model_database import ModelDatabase
from pharmpy.workflows.model_entry import ModelEntry
from pharmpy.workflows.results import ModelfitResults, mfr

from .external import parse_modelfit_results


def fit(
    model_or_models: Union[Model, list[Model]],
    tool: Optional[str] = None,
    path: Optional[Union[Path, str]] = None,
    context: Optional[Context] = None,
) -> Union[ModelfitResults, list[ModelfitResults]]:
    """Fit models.

    Parameters
    ----------
    model_or_models : Model | list[Model]
        List of models or one single model
    tool : str
        Estimation tool to use. None to use default
    path :  Path | str
        Path to fit directory
    context : Context
        Run in this context

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

    modelfit_results = run_tool('modelfit', models, tool=tool, path=path, context=context)

    return modelfit_results if single else list(modelfit_results)


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
    res = pharmpy.workflows.results.read_results(path)
    return res


def run_tool(name: str, *args, **kwargs) -> Union[Model, list[Model], tuple[Model], Results]:
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
    # NOTE: The implementation of run_tool is split into those two functions to
    # allow for individual testing and mocking.
    tool = import_tool(name)
    return run_tool_with_name(name, tool, args, kwargs)


def import_tool(name: str):
    return importlib.import_module(f'pharmpy.tools.{name}')


def run_tool_with_name(
    name: str, tool, args: Sequence, kwargs: Mapping[str, Any]
) -> Union[Model, list[Model], tuple[Model], Results]:
    dispatching_options, common_options, tool_options = split_common_options(kwargs)

    create_workflow = tool.create_workflow

    dispatcher, ctx = _get_run_setup(dispatching_options, common_options, name)

    tool_params = inspect.signature(create_workflow).parameters
    tool_param_types = get_type_hints(create_workflow)

    tool_metadata = _create_metadata(
        database=ctx,
        dispatcher=dispatcher,
        tool_name=name,
        tool_params=tool_params,
        tool_param_types=tool_param_types,
        args=args,
        tool_options=tool_options,
        common_options=common_options,
    )

    ctx.store_metadata(tool_metadata)

    if validate_input := getattr(tool, 'validate_input', None):
        validate_input(*args, **tool_options)

    wf: Workflow = create_workflow(*args, **tool_options)
    assert wf.name == name

    res = execute_workflow(wf, dispatcher=dispatcher, context=ctx)
    assert name == 'modelfit' or isinstance(res, Results) or name == 'simulation'

    tool_metadata = _update_metadata(tool_metadata, res)
    ctx.store_metadata(tool_metadata)

    return res


def _create_metadata(
    database: Context,
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
    # FIXME: Make metadata immutable
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

    tool_metadata = tool_database.retrieve_metadata()
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
    # NOTE: Load models to memory
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
        except KeyError:
            raise ValueError(
                f'Cannot resume run because model argument "{model_key}" ({model_name}) cannot be restored.'
            )
        tool_options = tool_options.copy()
        tool_options[model_key] = model

    # NOTE: Load results to memory
    for results_key in _results_param_keys(tool_params, tool_param_types):
        results_json = tool_options.get(results_key)
        if results_json is not None:
            tool_options = tool_options.copy()
            tool_options[results_key] = pharmpy.workflows.results.read_results(results_json)

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
    database: Context,
    tool_name: str,
    tool_params,
    tool_param_types,
    args: Sequence,
    kwargs: Mapping[str, Any],
):
    # FIXME: Add config file dump, estimation tool etc.
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
    database: Context, dispatcher, toolname: Optional[str], common_options: Mapping[str, Any]
):
    setup_metadata = {}
    setup_metadata['dispatcher'] = dispatcher.__name__
    # FIXME: Naming of workflows/tools should be consistent (db and input name of tool)
    setup_metadata['context'] = {
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
            # NOTE: We do not handle *{param_key}, or **{param_key}
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
        if model is None:
            continue
        yield param_key, model


def _store_input_model(db: ModelDatabase, model: Model, name: str):
    model_copy = model.replace(name=name)
    with db.transaction(model_copy) as txn:
        txn.store_model()
        txn.store_modelfit_results()


def _now():
    return datetime.now().astimezone().isoformat()


def _get_run_setup(dispatching_options, common_options, toolname) -> tuple[Any, Context]:
    try:
        dispatcher = dispatching_options['dispatcher']
    except KeyError:
        from pharmpy.workflows import default_dispatcher

        dispatcher = default_dispatcher

    ctx = dispatching_options.get('context', None)
    if ctx is None:
        from pharmpy.workflows import default_context

        common_path = dispatching_options.get('path', None)
        if common_path is not None:
            path = Path(dispatching_options['path'])
            ctx = default_context(path.name, path.parent, common_options=common_options)
        else:
            n = 1
            while True:
                name = f"{toolname}{n}"
                if not default_context.exists(name):
                    ctx = default_context(name, common_options=common_options)
                    break
                n += 1

    return dispatcher, ctx


def retrieve_models(
    source: Union[str, Path, Context],
    names: Optional[list[str]] = None,
) -> list[Model]:
    """Retrieve models after a tool run

    Any models created and run by the tool can be
    retrieved.

    Parameters
    ----------
    source : str, Path, Context
        Source where to find models. Can be a path (as str or Path), or a
        Context
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
        context = LocalDirectoryContext(path)
    elif isinstance(source, Context):
        context = source
    else:
        raise NotImplementedError(f'Not implemented for type \'{type(source)}\'')

    names_all = context.list_all_names()
    if names is None:
        names = names_all
    diff = set(names).difference(names_all)
    if diff:
        raise ValueError(f'Models {diff} not in database')
    models = [context.retrieve_model_entry(name).model for name in names]
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
        final_model = getattr(res, 'final_model')
    except AttributeError:
        raise ValueError('Attribute \'final_model\' is missing from results object')

    if final_model is None:
        raise ValueError('Attribute \'final_model\' is None')

    return retrieve_models(res, names=[final_model.name])[0]


def print_fit_summary(model: Model, modelfit_results: ModelfitResults):
    """Print a summary of the model fit

    Parameters
    ----------
    model : Model
        Pharmpy model object
    modelfit_results : ModelfitResults
        Pharmpy ModelfitResults object
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

    res = mfr(modelfit_results)

    print_header("Parameter estimation status", first=True)
    print_fmt("Minimization successful", bool_ok_error(res.minimization_successful))
    print_fmt("No rounding errors", bool_ok_error(res.termination_cause != 'rounding_errors'))
    ofv = res.ofv
    assert ofv is not None
    print_fmt("Objective function value", round(ofv, 1))

    print_header("Parameter uncertainty status")
    cov_run = model.execution_steps[-1].cov
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


def summarize_errors(context: Context) -> pd.DataFrame:
    """Summarize errors and warnings from all runs in a context.

    Summarize the errors and warnings found after running the model/models.

    Parameters
    ----------
    context : Context
        Context in which models were run

    Return
    ------
    pd.DataFrame
        A DataFrame of errors with model name, category (error or warning), and an int as index,
        an empty DataFrame if there were no errors or warnings found.

    """
    names = context.list_all_names()
    mes = [context.retrieve_model_entry(name) for name in names]
    return summarize_errors_from_entries(mes)


def summarize_errors_from_entries(mes: list[ModelEntry]):
    idcs, rows = [], []

    for me in mes:
        name = me.model.name
        res = me.modelfit_results
        if res is not None and len(res.log) > 0:
            for i, entry in enumerate(res.log):
                idcs.append((name, entry.category, i))
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
    base_model_res: ModelfitResults,
    models: list[Model],
    models_res: list[ModelfitResults],
    parent_dict: Optional[Union[dict[str, str], dict[Model, Model]]] = None,
    strictness: Optional[str] = "minimization_successful",
    rank_type: str = 'ofv',
    cutoff: Optional[float] = None,
    bic_type: str = 'mixed',
    **kwargs,
) -> pd.DataFrame:
    """Ranks a list of models

    Ranks a list of models with a given ranking function

    Parameters
    ----------
    base_model : Model
        Base model to compare to
    base_model_res : ModelfitResults
        Results of base model
    models : list
        List of models
    models_res : list
        List of modelfit results
    parent_dict : dict
        Dict where key is child and value is parent. Only relevant for LRT, if None base will be set as parent
    strictness : str or None
        Strictness criteria that are allowed for ranking. Default is "minimization_successful".
    rank_type : str
        Name of ranking type. Available options are 'ofv', 'aic', 'bic', 'lrt' (OFV with LRT)
    cutoff : float or None
        Value to use as cutoff. If using LRT, cutoff denotes p-value. Default is None
    bic_type : str
        Type of BIC to calculate. Default is the mixed effects.
    kwargs
        Arguments to pass to calculate BIC (such as `mult_test_p` and `mult_test_p`)

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
    ...             rank_type='lrt') # doctest: +SKIP
    """
    if len(models) != len(models_res):
        raise ValueError('Different length of `models` and `models_res`')
    if rank_type == 'lrt' and not parent_dict:
        parent_dict = {model.name: base_model.name for model in models}
    if parent_dict and not isinstance(list(parent_dict.keys())[0], str):
        parent_dict = {child.name: parent.name for child, parent in parent_dict.items()}

    models_all = [base_model] + models
    res_all = [base_model_res] + models_res

    rank_values, delta_values = {}, {}
    models_to_rank = []

    ref_value = _get_rankval(base_model, base_model_res, strictness, rank_type, bic_type, **kwargs)
    model_dict = {model.name: (model, res) for model, res in zip(models_all, res_all)}

    # Filter on strictness
    for model, res in zip(models_all, res_all):
        # Exclude OFV etc. if model was not successful
        rank_value = _get_rankval(model, res, strictness, rank_type, bic_type, **kwargs)
        if np.isnan(rank_value):
            continue
        if model.name == base_model.name:
            pass
        elif rank_type == 'lrt':
            parent_model, parent_res = model_dict[parent_dict[model.name]]
            if cutoff is None:
                co = 0.05 if lrt_df(parent_model, model) >= 0 else 0.01
            elif isinstance(cutoff, tuple):
                co = cutoff[0] if lrt_df(parent_model, model) >= 0 else cutoff[1]
            else:
                assert isinstance(cutoff, (float, int))
                co = cutoff
            parent_ofv = np.nan if (mfr := parent_res) is None else mfr.ofv
            model_ofv = np.nan if (mfr := res) is None else mfr.ofv
            if not lrt_test(parent_model, model, parent_ofv, model_ofv, co):
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


class ArrayEvaluator:
    def __init__(self, x):
        self.x = x

    def __lt__(self, value):
        return all(e < value for e in self.x)

    def __eq__(self, value):
        return all(e == value for e in self.x)

    def __le__(self, value):
        return all(e <= value for e in self.x)

    def __ge__(self, value):
        return all(e >= value for e in self.x)

    def __gt__(self, value):
        return all(e > value for e in self.x)


def is_strictness_fulfilled(
    res: ModelfitResults,
    model: Model,
    statement: str,
) -> bool:
    """Takes a ModelfitResults object and a statement as input and returns True/False
    if the evaluation of the statement is True/False.

    Parameters
    ----------
    results : ModelfitResults
        ModelfitResults object
    model : Model
        Model for parameter specific strictness.
    statement : str
        A statement containing the strictness criteria

    Return
    ------
    bool
        A bool indicating whether the strictness criteria are fulfilled or not.

    Examples
    --------
    >>> from pharmpy.tools import *
    >>> from pharmpy.modeling import *
    >>> res = load_example_modelfit_results('pheno')
    >>> model = load_example_model('pheno')
    >>> is_strictness_fulfilled(res, model, "minimization_successful or rounding_errors")
    True
    """
    if res is None or np.isnan(res.ofv):
        return False
    if statement is not None:
        statement = statement.lower()
        allowed_args = [
            'minimization_successful',
            'rounding_errors',
            'sigdigs',
            'maxevals_exceeded',
            'rse',
            'rse_theta',
            'rse_omega',
            'rse_sigma',
            'condition_number',
            'final_zero_gradient',
            'final_zero_gradient_theta',
            'final_zero_gradient_omega',
            'final_zero_gradient_sigma',
            'estimate_near_boundary',
            'estimate_near_boundary_theta',
            'estimate_near_boundary_omega',
            'estimate_near_boundary_sigma',
        ]
        unwanted_args = ['and', 'or', 'not']
        find_all_words = re.findall(r'[^\d\W]+', statement)
        args_in_statement = [w for w in find_all_words if w not in unwanted_args]
        find_all_non_allowed_operators = re.findall(r"[^\w\s\.\<\>\=\!\(\)]", statement)
        if len(find_all_non_allowed_operators) > 0:
            raise ValueError(
                f"Unallowed operators found: {', '.join(find_all_non_allowed_operators)}"
            )

        # Check that only allowed arguments are in the statement
        if not all(map(lambda x: x in allowed_args, args_in_statement)):
            raise ValueError(
                f'Some expressions were not correct. Valid arguments are: {allowed_args}'
            )
        else:
            minimization_successful = res.minimization_successful  # noqa
            rounding_errors = res.termination_cause == "rounding_errors"  # noqa
            maxevals_exceeded = res.termination_cause == "maxevals_exceeded"  # noqa
            sigdigs = ArrayEvaluator([res.significant_digits])  # noqa
            final_zero_gradient = 'final_zero_gradient' in res.warnings  # noqa
            estimate_near_boundary = 'estimate_near_boundary' in res.warnings  # noqa
            if 'condition_number' in args_in_statement:
                if res.covariance_matrix is not None:
                    condition_number = ArrayEvaluator(  # noqa
                        [np.linalg.cond(res.covariance_matrix)]
                    )
                else:
                    raise ValueError("Could not calculate condition_number.")
            if "rse" in args_in_statement:
                if res.relative_standard_errors is not None:
                    rse = ArrayEvaluator(res.relative_standard_errors)  # noqa
                else:
                    raise ValueError("Could not calculate relative standard error.")

            if (
                'rse_theta' in args_in_statement
                or 'rse_omega' in args_in_statement
                or 'rse_sigma' in args_in_statement
            ):
                rse = res.relative_standard_errors
                rse_theta = ArrayEvaluator(rse[rse.index.isin(get_thetas(model).names)])  # noqa
                rse_omega = ArrayEvaluator(rse[rse.index.isin(get_omegas(model).names)])  # noqa
                rse_sigma = ArrayEvaluator(rse[rse.index.isin(get_sigmas(model).names)])  # noqa
            if (
                'final_zero_gradient_theta' in args_in_statement
                or 'final_zero_gradient_omega' in args_in_statement
                or 'final_zero_gradient_sigma' in args_in_statement
            ):
                grd = res.gradients
                final_zero_gradient_theta = (  # noqa
                    grd[grd.index.isin(get_thetas(model).names)] == 0
                ).any() or grd[grd.index.isin(get_thetas(model).names)].isnull().any()
                final_zero_gradient_omega = (  # noqa
                    grd[grd.index.isin(get_omegas(model).names)] == 0
                ).any() or grd[grd.index.isin(get_thetas(model).names)].isnull().any()
                final_zero_gradient_sigma = (  # noqa
                    grd[grd.index.isin(get_sigmas(model).names)] == 0
                ).any() or grd[grd.index.isin(get_thetas(model).names)].isnull().any()
            if (
                'estimate_near_boundary' in args_in_statement
                or 'estimate_near_boundary_theta' in args_in_statement
                or 'estimate_near_boundary_omega' in args_in_statement
                or 'estimate_near_boundary_sigma' in args_in_statement
            ):
                ests = res.parameter_estimates
                estimate_near_boundary = check_parameters_near_bounds(model, ests).any()  # noqa
                estimate_near_boundary_theta = check_parameters_near_bounds(  # noqa
                    model, ests[ests.index.isin(get_thetas(model).names)]
                ).any()
                estimate_near_boundary_omega = check_parameters_near_bounds(  # noqa
                    model, ests[ests.index.isin(get_omegas(model).names)]
                ).any()
                estimate_near_boundary_sigma = check_parameters_near_bounds(  # noqa
                    model, ests[ests.index.isin(get_sigmas(model).names)]
                ).any()

        return eval(statement)
    else:
        return True


def _get_rankval(model, res, strictness, rank_type, bic_type, **kwargs):
    if not is_strictness_fulfilled(res, model, strictness):
        return np.nan
    if rank_type in ['ofv', 'lrt']:
        return res.ofv
    elif rank_type == 'aic':
        return calculate_aic(model, res.ofv)
    elif rank_type == 'bic':
        return calculate_bic(model, res.ofv, bic_type, **kwargs)
    else:
        raise ValueError('Unknown rank_type: must be ofv, lrt, aic, or bic')


def summarize_modelfit_results(
    context: Context,
    include_all_execution_steps: bool = False,
) -> pd.DataFrame:
    """Summarize results of model runs

    Summarize different results after fitting a model, includes runtime, ofv,
    and parameter estimates (with errors). If include_all_execution_steps is False,
    only the last estimation step will be included (note that in that case, the
    minimization_successful value will be referring to the last estimation step, if
    last step is evaluation it will go backwards until it finds an estimation step
    that wasn't an evaluation).

    Parameters
    ----------
    context : Context
        Context in which models were run
    include_all_execution_steps : bool
        Whether to include all estimation steps, default is False

    Return
    ------
    pd.DataFrame
        A DataFrame of modelfit results with model name and estmation step as index.

    """

    names = context.list_all_names()
    mes = [context.retrieve_model_entry(name) for name in names]
    df = summarize_modelfit_results_from_entries(mes, include_all_execution_steps)
    return df


def summarize_modelfit_results_from_entries(
    mes: list[ModelEntry],
    include_all_execution_steps: bool = False,
) -> pd.DataFrame:

    if mes is None:
        raise ValueError('Option `results` is None')
    if all(me is None for me in mes):
        raise ValueError('All input results are empty')

    summaries = []

    for me in mes:
        if me is not None and me.modelfit_results is not None:
            summary = _get_model_result_summary(me, include_all_execution_steps)
            summary.insert(0, 'description', me.model.description)
            summaries.append(summary)

    with warnings.catch_warnings():
        # Needed because of warning in pandas 2.1.1
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )

        df = pd.concat(summaries)

    return df


def _get_model_result_summary(me, include_all_execution_steps=False):
    res = me.modelfit_results
    if not include_all_execution_steps:
        summary_dict = _summarize_step(res, -1)
        index = pd.Index([me.model.name], name='model')
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
            tuples.append((me.model.name, i + 1))
        index = pd.MultiIndex.from_tuples(tuples, names=['model', 'step'])
        summary_df = pd.DataFrame(summary_dicts, index=index)

    no_of_errors = len(res.log.errors)
    no_of_warnings = len(res.log.warnings)

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
    path = normalize_user_given_path(path)
    model = read_model(path)
    res = parse_modelfit_results(model, path)
    return res


def _get_run_setup_from_metadata(path):
    import pharmpy.workflows as workflows

    context = workflows.default_context(name=path, ref=None)

    tool_metadata = context.retrieve_metadata()
    common_options = tool_metadata['common_options']

    # TODO: Be more general
    dispatcher = getattr(workflows, common_options['dispatcher'].split('.')[-1])

    return dispatcher, context


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
    available = ('pheno', 'pheno_linear')
    if name not in available:
        raise ValueError(f'Unknown example model {name}. Available examples: {available}')
    path = Path(__file__).resolve().parent.parent / 'internals' / 'example_models' / (name + '.mod')
    res = read_modelfit_results(path)
    return res
