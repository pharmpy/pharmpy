import inspect
from typing import TypeVar

from pharmpy.model import Model, Results
from pharmpy.utils import normalize_user_given_path

from .workflows import Workflow

T = TypeVar('T')


def execute_workflow(
    workflow: Workflow[T], dispatcher=None, database=None, path=None, resume=False
) -> T:
    """Execute workflow

    Parameters
    ----------
    workflow : Workflow
        Workflow to execute
    dispatcher : ExecutionDispatcher
        Dispatcher to use
    database : ToolDatabase
        Tool database to use. None for the default Tool database.
    path : Path
        Path to use for database if applicable.
    resume : bool
        Whether to allow resuming previous workflow execution.

    Returns
    -------
    Results
        Results object created by workflow
    """
    # FIXME Return type is not always Results
    if dispatcher is None:
        from pharmpy.workflows import default_dispatcher

        dispatcher = default_dispatcher
    if database is None:
        from pharmpy.workflows import default_tool_database

        database = default_tool_database(
            toolname=workflow.name, path=path, exist_ok=resume
        )  # TODO: database -> tool_database

    # For all input models set new database and read in results
    original_input_models = []
    input_models = []
    for task in workflow.tasks:
        if task.has_input():
            new_inp = []

            for inp in task.task_input:
                if isinstance(inp, Model):
                    original_input_models.append(inp)
                    inp.modelfit_results  # To read in the results
                    new_model = inp.copy()
                    new_model.parent_model = new_model.name
                    new_model.dataset
                    new_inp.append(new_model)
                    input_models.append(new_model)
                else:
                    new_inp.append(inp)
            task.task_input = new_inp

    insert_context(workflow, database)

    res: T = dispatcher.run(workflow)

    if isinstance(res, Results):
        if hasattr(res, 'tool_database'):
            res.tool_database = database  # pyright: ignore [reportGeneralTypeIssues]
        database.store_results(res)
        if hasattr(res, 'rst_path'):
            from pharmpy.modeling.reporting import create_report

            create_report(res, database.path)
    elif isinstance(res, Model):
        original_input_models[0].modelfit_results = res.modelfit_results
    elif isinstance(res, list) or isinstance(res, tuple):
        for model in res:
            for original_model in original_input_models:
                if original_model.name == model.name:
                    original_model.modelfit_results = model.modelfit_results
                    break

    return res


def split_common_options(d):
    """Split the dict into common options and other options

    Parameters
    ----------
    d : dict
        Dictionary of all options

    Returns
    -------
    Tuple of common options and other option dictionaries
    """
    execute_options = ['path', 'resume']
    common_options = dict()
    other_options = dict()
    for key, value in d.items():
        if key in execute_options:
            if key == 'path':
                if value is not None:
                    value = normalize_user_given_path(value)
            common_options[key] = value
        else:
            other_options[key] = value
    return common_options, other_options


def insert_context(workflow, context):
    """Insert tool context (database) for all tasks in a workflow needing it

    having context as first argument of function
    """
    for task in workflow.tasks:
        if tuple(inspect.signature(task.function).parameters)[0] == 'context':
            task.task_input = [context] + list(task.task_input)


def call_workflow(wf: Workflow[T], unique_name, db) -> T:
    """Dynamically call a workflow from another workflow.

    Currently only supports dask distributed

    Parameters
    ----------
    wf : Workflow
        A workflow object
    unique_name : str
        A name of the results node that is unique between parent and dynamically created workflows
    db : ToolDatabase
        ToolDatabase to pass to new workflow

    Returns
    -------
    Any
        Whatever the dynamic workflow returns
    """
    from dask.distributed import get_client, rejoin, secede

    from .optimize import optimize_task_graph_for_dask_distributed

    insert_context(wf, db)

    client = get_client()
    dsk = wf.as_dask_dict()
    dsk[unique_name] = dsk.pop('results')
    dsk_optimized = optimize_task_graph_for_dask_distributed(client, dsk)
    futures = client.get(dsk_optimized, unique_name, sync=False)
    secede()
    res: T = client.gather(futures)  # pyright: ignore [reportGeneralTypeIssues]
    rejoin()
    return res
