from __future__ import annotations

import importlib
import inspect
import math
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
from pharmpy.model import Model, RandomVariables
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
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features, parse
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.transits import Transits
from pharmpy.tools.psn_helpers import create_results as psn_create_results
from pharmpy.workflows import (
    DispatchingError,
    Results,
    Workflow,
    execute_subtool,
    execute_workflow,
    split_common_options,
)
from pharmpy.workflows.args import InputValidationError, canonicalize_seed
from pharmpy.workflows.contexts import Context, LocalDirectoryContext
from pharmpy.workflows.dispatchers import Dispatcher
from pharmpy.workflows.model_database import ModelDatabase
from pharmpy.workflows.model_entry import ModelEntry
from pharmpy.workflows.results import ModelfitResults, mfr

from .context import broadcast_log
from .external import parse_modelfit_results


def fit(
    model_or_models: Union[Model, list[Model]],
    esttool: Optional[str] = None,
    name: Optional[str] = None,
    context: Optional[Context] = None,
    ncores: int = 1,
) -> Union[ModelfitResults, list[ModelfitResults]]:
    """Fit models.

    Parameters
    ----------
    model_or_models : Model | list[Model]
        List of models or one single model
    esttool : str
        Estimation tool to use. None to use default
    name : str
        Name of run
    context : Context
        Run in this context
    ncores : int
        Number of cores to use for estimation

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

    if not context:
        dispatcher = 'local_serial'
    else:
        dispatcher = None

    modelfit_results = run_tool(
        'modelfit',
        models,
        esttool=esttool,
        name=name,
        context=context,
        dispatcher=dispatcher,
        ncores=ncores,
    )

    return (
        modelfit_results
        if single or isinstance(modelfit_results, ModelfitResults)
        else list(modelfit_results)
    )


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
    >>> from pharmpy.tools.run import create_results
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


def run_tool(tool_name: str, *args, **kwargs) -> Union[Model, list[Model], tuple[Model], Results]:
    """Run tool workflow

    .. note::
        This is a general function that can run any tool. There is also one function for each
        specific tool. Please refer to the documentation of these for more specific information.

    Parameters
    ----------
    tool_name : str
        Name of tool to run
    args
        Arguments to pass to tool
    kwargs
        Arguments to pass to tool

    Returns
    -------
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
    tool = import_tool(tool_name)
    return run_tool_with_name(tool_name, tool, args, kwargs)


def import_tool(name: str):
    return importlib.import_module(f'pharmpy.tools.{name}')


def run_tool_with_name(
    tool_name: str, tool, args: Sequence, kwargs: Mapping[str, Any]
) -> Union[Model, list[Model], tuple[Model], Results]:
    dispatching_options, common_options, seed, tool_options = split_common_options(kwargs)

    seed = canonicalize_seed(seed)

    if validate_input := getattr(tool, 'validate_input', None):
        try:
            validate_input(*args, **tool_options)
        except Exception as err:
            raise InputValidationError(str(err))

    dispatcher = Dispatcher.select_dispatcher(dispatching_options['dispatcher'])
    ctx = get_context(dispatching_options, tool_name)

    create_workflow = tool.create_workflow

    if ctx.has_started():
        if ctx.has_completed():
            tool_params = inspect.signature(tool.create_workflow).parameters
            tool_param_types = get_type_hints(tool.create_workflow)

            prev_tool_options = _parse_tool_options_from_json_metadata(
                ctx.retrieve_metadata(), tool_params, tool_param_types, ctx
            )

            tool_metadata = _create_metadata_tool(
                ctx, tool_name, create_workflow, args, tool_options
            )
            new_tool_options = _parse_tool_options_from_json_metadata(
                tool_metadata, tool_params, tool_param_types, ctx
            )

            if new_tool_options != prev_tool_options:
                raise DispatchingError(
                    "The arguments to the tool are different from the first time "
                    "it was run. "
                    "Delete the directory or run again using a new name."
                )

            results = ctx.retrieve_results()
            broadcast_log(ctx)
            return results
        else:
            ctx.log_info("Resuming interrupted run")
    else:
        pass

    dispatching_options['context'] = {
        '__class__': type(ctx).__name__,
        'name': str(ctx.name),
        'ref': str(ctx.ref),
    }

    tool_metadata = create_metadata(
        database=ctx,
        tool_name=tool_name,
        tool_func=create_workflow,
        args=args,
        tool_options=tool_options,
        seed=seed,
        common_options=common_options,
        dispatching_options=dispatching_options,
    )

    ctx.store_metadata(tool_metadata)

    if (
        "model" in tool_options
        and "results" in tool_options
        and "esttool" in common_options
        and common_options["esttool"] != "dummy"
    ):

        model_type = str(type(tool_options["model"])).split(".")[-3]
        results = tool_options["results"]
        esttool = common_options["esttool"]
        if results:
            if esttool != model_type:
                if not (esttool is None and model_type == "nonmem"):
                    warnings.warn(
                        f"Not recommended to run tools with different estimation tool ({esttool})"
                        f" than that of the input model ({model_type})"
                    )

    wf: Workflow = create_workflow(*args, **tool_options)
    assert wf.name == tool_name

    res = execute_workflow(wf, dispatcher=dispatcher, context=ctx)
    assert (
        tool_name == 'modelfit'
        or isinstance(res, Results)
        or tool_name == 'simulation'
        or res is None
    )

    tool_metadata = _update_metadata(tool_metadata)
    ctx.store_metadata(tool_metadata)

    ctx.finalize()

    return res


def create_metadata(
    database: Context,
    tool_name: str,
    tool_func,
    args: Sequence,
    tool_options: Mapping[str, Any],
    seed: int,
    common_options: Optional[Mapping[str, Any]] = None,
    dispatching_options: Optional[Mapping[str, Any]] = None,
):
    tool_metadata = _create_metadata_tool(database, tool_name, tool_func, args, tool_options)
    if common_options and dispatching_options:
        setup_metadata = _create_metadata_common(database, tool_name, common_options)
        tool_metadata['common_options'] = setup_metadata
        tool_metadata['dispatching_options'] = dispatching_options
    tool_metadata['seed'] = seed

    return tool_metadata


def _update_metadata(tool_metadata):
    # FIXME: Make metadata immutable
    tool_metadata['stats']['end_time'] = _now()
    return tool_metadata


def run_subtool(tool_name: str, ctx: Context, name=None, **kwargs):
    if not name:
        name = tool_name
    tool = import_tool(tool_name)
    subctx = ctx.create_subcontext(name)

    if subctx.has_completed():
        res = subctx.retrieve_results()
        subctx.log_info("Retrieving results from previously finished subtool")
        return res

    seed = kwargs.get('seed', None)
    seed = canonicalize_seed(seed)
    if 'seed' in kwargs.keys():
        del kwargs['seed']

    create_workflow = tool.create_workflow
    tool_metadata = create_metadata(
        database=ctx,
        tool_name=tool_name,
        tool_func=create_workflow,
        args=tuple(),
        tool_options=kwargs,
        seed=seed,
    )
    subctx.store_metadata(tool_metadata)
    wf: Workflow = create_workflow(**kwargs)
    assert wf.name == tool_name

    res = execute_subtool(wf, context=subctx)
    tool_metadata = _update_metadata(tool_metadata)
    subctx.store_metadata(tool_metadata)

    subctx.finalize()

    return res


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

    dispatcher, ctx = _get_run_setup_from_metadata(path)

    tool_metadata = ctx.retrieve_metadata()
    tool_name = tool_metadata['tool_name']

    tool = importlib.import_module(f'pharmpy.tools.{tool_name}')

    create_workflow = tool.create_workflow

    tool_params = inspect.signature(create_workflow).parameters
    tool_param_types = get_type_hints(create_workflow)

    tool_options = _parse_tool_options_from_json_metadata(
        tool_metadata, tool_params, tool_param_types, ctx
    )

    args, kwargs = _parse_args_kwargs_from_tool_options(tool_params, tool_options)

    if validate_input := getattr(tool, 'validate_input', None):
        validate_input(*args, **kwargs)

    wf: Workflow = create_workflow(*args, **kwargs)
    assert wf.name == tool_name

    res = execute_workflow(wf, dispatcher=dispatcher, database=ctx)
    assert tool_name == 'modelfit' or isinstance(res, Results)

    tool_metadata = _update_metadata(tool_metadata)
    ctx.store_metadata(tool_metadata)

    return res


def _parse_tool_options_from_json_metadata(
    tool_metadata,
    tool_params,
    tool_param_types,
    ctx,
):
    tool_options = tool_metadata['tool_options']
    db: ModelDatabase = ctx.model_database
    # NOTE: Load models to memory
    for model_key in _input_model_param_keys(tool_params, tool_param_types):
        model_metadata = tool_options.get(model_key)
        if model_metadata is None:
            raise ValueError(
                f'Cannot resume run because model argument "{model_key}" cannot be restored.'
            )

        assert model_metadata['__class__'] == 'Model'
        model_hash = model_metadata['key']

        try:
            model = db.retrieve_model(model_hash)
        except KeyError:
            raise ValueError(
                f'Cannot resume run because model argument "{model_key}" ({model_hash}) cannot be restored.'
            )
        tool_options = tool_options.copy()
        tool_options[model_key] = model

    # NOTE: Load results to memory
    for results_key in _results_param_keys(tool_params, tool_param_types):
        results_json = tool_options.get(results_key)
        if results_json is not None:
            tool_options = tool_options.copy()
            tool_options[results_key] = db.retrieve_modelfit_results(results_json["key"])

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
    tool_func,
    args: Sequence,
    kwargs: Mapping[str, Any],
):
    tool_params = inspect.signature(tool_func).parameters
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
        _store_input_models(db, tool_metadata, tool_params, kwargs)

    return tool_metadata


def _store_input_models(db, metadata, tool_params, kwargs):
    # Loop through all kwargs to find Model and ModelfitResults objects
    # If found will attempt to pair them together assuming that ones put closest
    # together given the function signature order belong together
    previous = None
    previous_arg = None
    for arg in tool_params.keys():
        current = kwargs.get(arg, None)
        if isinstance(current, (Model, ModelfitResults)):
            if previous is None:
                previous = current
                previous_arg = arg
            else:
                previous_is_model = isinstance(previous, Model)
                current_is_model = isinstance(current, Model)
                if previous_is_model and not current_is_model:
                    _store_model_and_results(db, metadata, previous_arg, previous, arg, current)
                    previous = None
                elif not previous_is_model and current_is_model:
                    _store_model_and_results(db, metadata, arg, current, previous_arg, previous)
                    previous = None
                elif previous_is_model:
                    _store_model(db, metadata, previous_arg, previous)
                    previous = current
                    previous_arg = arg
                else:
                    previous = current
                    previous_arg = arg
    if previous is not None and isinstance(previous, Model):
        _store_model(db, metadata, previous_arg, previous)


def _store_model_and_results(
    db: ModelDatabase, metadata, model_arg: str, model: Model, results_arg, results: ModelfitResults
):
    me = ModelEntry.create(model=model, modelfit_results=results)
    with db.transaction(me) as txn:
        txn.store_model()
        txn.store_modelfit_results()
        dbkey = str(txn.key)
        metadata['tool_options'][model_arg] = {
            '__class__': 'Model',
            'key': dbkey,
        }
        metadata['tool_options'][results_arg] = {
            '__class__': 'ModelfitResults',
            'key': dbkey,
        }


def _store_model(db: ModelDatabase, metadata, arg: str, model: Model):
    with db.transaction(model) as txn:
        txn.store_model()
        dbkey = str(txn.key)
        metadata['tool_options'][arg] = {
            '__class__': 'Model',
            'key': dbkey,
        }


def _create_metadata_common(
    database: Context, tool_name: Optional[str], common_options: Mapping[str, Any]
):
    setup_metadata = {}
    for key, value in common_options.items():
        if key not in setup_metadata.keys():
            if isinstance(value, Path):
                value = str(value)
            setup_metadata[str(key)] = value

    return setup_metadata


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


def _now():
    return datetime.now().astimezone().isoformat()


def _get_name(options, default_context, tool_name) -> str:
    name = options['name']
    if name is None:
        name = _create_new_context_name(default_context, tool_name)
    return name


def _create_new_context_name(context: type[Context], tool_name: str) -> str:
    n = 1
    while True:
        name = f"{tool_name}{n}"
        if not context.exists(name):
            break
        n += 1
    return name


def get_context(dispatching_options, tool_name) -> Context:
    ctx = dispatching_options['context']
    if ctx is None:
        from pharmpy.workflows import default_context

        name = _get_name(dispatching_options, default_context, tool_name)
        ref = dispatching_options['ref']
        ctx = default_context(name, ref)
    return ctx


def _open_context(source):
    if isinstance(source, Path) or isinstance(source, str):
        path = Path(source)
        context = LocalDirectoryContext(path)
    elif isinstance(source, Context):
        context = source
    else:
        raise NotImplementedError(f'Not implemented for type \'{type(source)}\'')
    return context


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

    """
    context = _open_context(source)
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
    >>> from pharmpy.tools.run import read_results, retrieve_final_model
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
    cov_run = model.execution_steps[-1].parameter_uncertainty_method is not None
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


def write_results(
    results: Results, path: Union[str, Path], compression: bool = False, csv: bool = False
):
    """Write results object to json (or csv) file

    Note that the csv-file cannot be read into a results object again.

    Parameters
    ----------
    results : Results
        Pharmpy results object
    path : Path
        Path to results file
    compression : bool
        True to compress the file. Not applicable to csv file
    csv : bool
        Save as csv file
    """
    path = normalize_user_given_path(path)
    if csv:
        results.to_csv(path)
    else:
        results.to_json(path, lzma=compression)


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
    strictness: str = "minimization_successful",
    rank_type: str = 'ofv',
    cutoff: Optional[float] = None,
    penalties: Optional[list[float]] = None,
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
    strictness : str
        Strictness criteria that are allowed for ranking. Default is "minimization_successful".
    rank_type : str
        Name of ranking type. Available options are 'ofv', 'aic', 'bic', 'lrt' (OFV with LRT)
    cutoff : float or None
        Value to use as cutoff. If using LRT, cutoff denotes p-value. Default is None
    penalties : list
        List of penalties to add to all models (including base model)
    kwargs
        Arguments to pass to calculate BIC (such as `mult_test_p` and `mult_test_p`)

    Return
    ------
    pd.DataFrame
        DataFrame of the ranked models

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools.run import rank_models
    >>> model_1 = load_example_model("pheno")
    >>> model_2 = load_example_model("pheno_linear")
    >>> rank_models(model_1, [model_2],
    ...             rank_type='lrt') # doctest: +SKIP
    """
    if len(models) != len(models_res):
        raise ValueError('Different length of `models` and `models_res`')
    if penalties is not None and len(models) + 1 != len(penalties):
        raise ValueError(
            f'Mismatch in length of `models` and `penalties`: number of `penalties` ({len(penalties)}) '
            f'must be one more than number of `models` ({len(models)})'
        )
    if rank_type == 'lrt' and not parent_dict:
        parent_dict = {model.name: base_model.name for model in models}
    if parent_dict and not isinstance(list(parent_dict.keys())[0], str):
        parent_dict = {child.name: parent.name for child, parent in parent_dict.items()}

    models_all = [base_model] + models
    res_all = [base_model_res] + models_res

    rank_values, delta_values = {}, {}
    models_to_rank = []

    ref_value = get_rankval(base_model, base_model_res, strictness, rank_type, **kwargs)
    if penalties:
        ref_value += penalties[0]
    model_dict = {model.name: (model, res) for model, res in zip(models_all, res_all)}

    # Filter on strictness
    for i, (model, res) in enumerate(zip(models_all, res_all)):
        # Exclude OFV etc. if model was not successful
        rank_value = get_rankval(model, res, strictness, rank_type, **kwargs)
        if np.isnan(rank_value):
            continue
        if penalties:
            rank_value += penalties[i]
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
    model: Model,
    results: ModelfitResults,
    strictness: str,
) -> bool:
    """Takes a ModelfitResults object and a statement as input and returns True/False
    if the evaluation of the statement is True/False.

    Parameters
    ----------
    model : Model
        Model for parameter specific strictness.
    results : ModelfitResults
        ModelfitResults object
    strictness : str
        A strictness expression

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
    >>> is_strictness_fulfilled(model, res, "minimization_successful or rounding_errors")
    True
    """
    if results is None:
        return False
    # FIXME: We should have the assert instead of the is is None
    # assert results is not None, f"results is None for model {model.name}"
    if np.isnan(results.ofv):
        return False
    elif strictness == "":
        return True
    else:
        strictness = strictness.lower()
        allowed_args = (
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
        )
        unwanted_args = ('and', 'or', 'not')
        find_all_words = re.findall(r'[^\d\W]+', strictness)
        args_in_statement = [w for w in find_all_words if w not in unwanted_args]
        find_all_non_allowed_operators = re.findall(r"[^\w\s\.\<\>\=\!\(\)]", strictness)
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
            minimization_successful = results.minimization_successful  # noqa
            rounding_errors = results.termination_cause == "rounding_errors"  # noqa
            maxevals_exceeded = results.termination_cause == "maxevals_exceeded"  # noqa
            sigdigs = ArrayEvaluator([results.significant_digits])  # noqa
            final_zero_gradient = 'final_zero_gradient' in results.warnings  # noqa
            estimate_near_boundary = 'estimate_near_boundary' in results.warnings  # noqa
            if 'condition_number' in args_in_statement:
                if results.covariance_matrix is not None:
                    condition_number = ArrayEvaluator(  # noqa
                        [np.linalg.cond(results.covariance_matrix)]
                    )
                else:
                    raise ValueError("Could not calculate condition_number.")
            if "rse" in args_in_statement:
                if results.relative_standard_errors is not None:
                    rse = ArrayEvaluator(results.relative_standard_errors)  # noqa
                else:
                    raise ValueError("Could not calculate relative standard error.")

            if (
                'rse_theta' in args_in_statement
                or 'rse_omega' in args_in_statement
                or 'rse_sigma' in args_in_statement
            ):
                rse = results.relative_standard_errors
                rse_theta = ArrayEvaluator(rse[rse.index.isin(get_thetas(model).names)])  # noqa
                rse_omega = ArrayEvaluator(rse[rse.index.isin(get_omegas(model).names)])  # noqa
                rse_sigma = ArrayEvaluator(rse[rse.index.isin(get_sigmas(model).names)])  # noqa
            if (
                'final_zero_gradient_theta' in args_in_statement
                or 'final_zero_gradient_omega' in args_in_statement
                or 'final_zero_gradient_sigma' in args_in_statement
            ):
                grd = results.gradients
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
                ests = results.parameter_estimates
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

        return eval(strictness)


def get_rankval(model, res, strictness, rank_type, **kwargs):
    if not is_strictness_fulfilled(model, res, strictness):
        return np.nan
    if rank_type in ['ofv', 'lrt']:
        return res.ofv
    elif rank_type == 'aic':
        return calculate_aic(model, res.ofv)
    elif rank_type == 'bic':
        bic_type = kwargs.get('bic_type')
        return calculate_bic(model, res.ofv, type=bic_type)
    else:
        raise ValueError(f'Unknown rank_type: got `{rank_type}`, must be ofv, lrt, aic, or bic')


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

    no_of_errors = len(res.log.errors) if res.log is not None else 0
    no_of_warnings = len(res.log.warnings) if res.log is not None else 0

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


def read_modelfit_results(path: Union[str, Path], esttool: str = None) -> ModelfitResults:
    """Read results from external tool for a model

    Parameters
    ----------
    path : Path or str
        Path to model file
    esttool : str
        Set if other than the default estimation tool is to be used

    Return
    ------
    ModelfitResults
        Results object
    """
    path = normalize_user_given_path(path)
    model = read_model(path)
    res = parse_modelfit_results(model, path, esttool)
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
    POP_CL     0.004696
    POP_VC     0.984258
    COVAPGR    0.158920
    IIV_CL     0.029351
    IIV_VC     0.027906
    SIGMA      0.013241
    Name: estimates, dtype: float64

    """
    available = ('pheno', 'pheno_linear')
    if name not in available:
        raise ValueError(f'Unknown example model {name}. Available examples: {available}')
    path = Path(__file__).resolve().parent.parent / 'internals' / 'example_models' / (name + '.mod')
    res = read_modelfit_results(path)
    return res


def calculate_mbic_penalty(
    candidate_model: Model,
    search_space: Union[str, list[str], ModelFeatures],
    base_model: Optional[Model] = None,
    E_p: Optional[Union[float, str]] = 1.0,
    E_q: Optional[Union[float, str]] = 1.0,
    keep: Optional[list[str]] = None,
):
    if E_p == 0 or E_q == 0:
        raise ValueError('E-values cannot be 0')
    if isinstance(search_space, str) or isinstance(search_space, ModelFeatures):
        if base_model:
            raise ValueError('Cannot provide both `search_space` and `base_model`')
        if E_p is None:
            raise ValueError(
                'Missing value for `E_p`, must be specified when using MFL in `search_space`'
            )

        cand_features = get_model_features(candidate_model)

        if isinstance(search_space, str):
            search_space_mfl = parse(search_space, mfl_class=True)
        else:
            search_space_mfl = search_space
        cand_mfl = parse(cand_features, mfl_class=True)
        # FIXME: Workaround to skip covariate effects detected in search space
        cand_mfl = ModelFeatures.create(
            absorption=cand_mfl.absorption,
            elimination=cand_mfl.elimination,
            transits=cand_mfl.transits,
            peripherals=cand_mfl.peripherals,
            lagtime=cand_mfl.lagtime,
        )

        p, k_p = get_penalty_parameters_mfl(search_space_mfl, cand_mfl)

        q = 0
        k_q = 0
    if isinstance(search_space, list):
        allowed_options = ['iiv_diag', 'iiv_block', 'iov']
        for search_space_type in search_space:
            if search_space_type not in allowed_options:
                raise ValueError(
                    f'Unknown `search_space`: {search_space_type} (must be one of {allowed_options})'
                )
        if 'iiv_block' in search_space:
            if 'iov' in search_space:
                raise ValueError(
                    'Incorrect `search_space`: `iiv_block` and `iov` cannot be tested in same search space'
                )
            if E_q is None:
                raise ValueError(
                    'Missing value for `E_q`, must be specified when using `iiv_block` in `search_space`'
                )
        if 'iiv_diag' in search_space or 'iov' in search_space:
            if E_p is None:
                raise ValueError(
                    'Missing value for `E_p`, must be specified when using `iiv_diag` or `iov` in `search_space`'
                )
        if not base_model:
            raise ValueError(
                'Missing `base_model`: reference model is needed to determine search space'
            )

        p, k_p, q, k_q = get_penalty_parameters_rvs(base_model, candidate_model, search_space, keep)

    # To avoid domain error
    p = p if k_p != 0 else 1
    q = q if k_q != 0 else 1
    # If either are omitted
    E_p = _prepare_E_value(E_p, p, type='p')
    E_q = _prepare_E_value(E_q, q, type='q')

    return 2 * k_p * math.log(p / E_p) + 2 * k_q * math.log(q / E_q)


def _prepare_E_value(e, p, type='p'):
    if isinstance(e, str):
        e = (float(e.strip('%')) / 100) * p
    elif e is None:
        e = 1
    else:
        e = e
    if e > p:
        raise ValueError(f'`E_{type}` cannot be bigger than `{type}`: E_{type}={e}, {type}={p}')
    return e


def get_penalty_parameters_mfl(search_space_mfl, cand_mfl):
    p, k_p = 0, 0
    for attr_name, attr in vars(cand_mfl).items():
        if not attr:
            continue
        attr_search_space = getattr(search_space_mfl, attr_name)
        if isinstance(attr, tuple):
            assert len(attr) == 1 and len(attr_search_space) == 1
            attr, attr_search_space = attr[0], attr_search_space[0]

        if len(attr_search_space) == 1:
            continue

        if isinstance(attr, Absorption):
            abs_search_space = [mode.name for mode in attr_search_space.modes]
            if 'SEQ-ZO-FO' in abs_search_space:
                p_attr = 1
            else:
                p_attr = 0
            abs_type = attr.modes[0].name
            if abs_type == 'SEQ-ZO-FO':
                k_p_attr = 1
            else:
                k_p_attr = 0
            if 'INST' in abs_search_space:
                p_attr += 1
                if abs_type != 'INST':
                    k_p_attr += 1
        elif isinstance(attr, Transits):

            def _has_depot(attr):
                return 'DEPOT' in [mode.name for mode in attr.depot]

            if _has_depot(attr_search_space.eval):
                p_attr = len([n for n in attr_search_space.counts if n > 0])
                if _has_depot(attr):
                    k_p_attr = 1 if attr.counts[0] > 0 else 0
                else:
                    k_p_attr = 0
            else:
                p_attr = 0
                k_p_attr = 0
        elif isinstance(attr, LagTime):
            p_attr = 1
            k_p_attr = 1 if attr.modes[0].name == 'ON' else 0
        elif isinstance(attr, Elimination):
            attr_names = [mode.name for mode in attr_search_space.modes]
            sort_val = {'FO': 0, 'MM': 1, 'MIX-FO-MM': 2}
            attr_names.sort(key=lambda x: sort_val[x])
            p_attr = len(attr_names) - 1
            elim_type = attr.modes[0].name
            k_p_attr = list(attr_names).index(elim_type)
        elif isinstance(attr, Peripherals):
            # FIXME: This will not work with e.g. `PERIPHERALS([0,2])`
            p_attr = len(attr_search_space) - 1
            k_p_attr = attr.counts[0]
        else:
            raise ValueError(f'MFL attribute of type `{type(attr)}` not supported.')

        p += p_attr
        k_p += k_p_attr

    return p, k_p


def get_penalty_parameters_rvs(base_model, cand_model, search_space, keep=None):
    base_etas = _get_var_params(base_model, search_space)
    cand_etas = _get_var_params(cand_model, search_space)

    p, k_p, q, k_q = 0, 0, 0, 0
    if 'iiv_diag' in search_space or 'iov' in search_space:
        p = len(base_etas.variance_parameters)
        k_p = len(cand_etas.variance_parameters)
        if keep:
            p -= len(keep)
            k_p -= len(keep)
    if 'iiv_block' in search_space:
        q = int(len(base_etas.variance_parameters) * (len(base_etas.variance_parameters) - 1) / 2)
        cov_params = [
            p for p in cand_etas.parameter_names if p not in cand_etas.variance_parameters
        ]
        k_q = len(cov_params)

    return p, k_p, q, k_q


def _get_var_params(model, search_space):
    etas = []
    fixed_params = model.parameters.fixed.names
    if any(s.startswith('iiv') for s in search_space):
        iivs = model.random_variables.iiv
        iivs_non_fixed = [iiv for iiv in iivs if set(iiv.parameter_names).isdisjoint(fixed_params)]
        etas.extend(iivs_non_fixed)
    if 'iov' in search_space:
        iovs = model.random_variables.iov
        iovs_non_fixed = [iov for iov in iovs if set(iov.parameter_names).isdisjoint(fixed_params)]
        etas.extend(iovs_non_fixed)

    return RandomVariables.create(etas)
