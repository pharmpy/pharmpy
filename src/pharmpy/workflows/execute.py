from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from pharmpy.model import Model, Results

from .context import insert_context
from .workflow import Workflow

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
        new_inp = []

        for inp in task.task_input:
            if isinstance(inp, Model):
                original_input_models.append(inp)
                inp.modelfit_results  # To read in the results
                new_model = inp.replace(parent_model=inp.name)
                new_inp.append(new_model)
                input_models.append(new_model)
            else:
                new_inp.append(inp)

        task.task_input = tuple(new_inp)

    insert_context(workflow, database)

    res: T = dispatcher.run(workflow)

    if isinstance(res, Results):
        if hasattr(res, 'tool_database'):
            res = replace(res, tool_database=database)
        database.store_results(res)
        if hasattr(res, 'rst_path'):
            from pharmpy.tools.reporting import create_report

            create_report(res, database.path)
    elif isinstance(res, Model):
        # original_input_models[0].modelfit_results = res.modelfit_results
        pass
    elif isinstance(res, list) or isinstance(res, tuple):
        # index = {model.name: model for model in res}
        # for original_model in original_input_models:
        #    if (model := index.get(original_model.name, None)) is not None:
        #        original_model.modelfit_results = model.modelfit_results
        pass

    return res
