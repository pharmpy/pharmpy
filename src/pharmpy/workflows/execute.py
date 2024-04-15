from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

from pharmpy.model import Model

from .results import ModelfitResults, Results
from .workflow import Workflow, WorkflowBuilder, insert_context

T = TypeVar('T')


def execute_workflow(
    workflow: Workflow[T], dispatcher=None, context=None, path=None, resume=False
) -> T:
    """Execute workflow

    Parameters
    ----------
    workflow : Workflow
        Workflow to execute
    dispatcher : ExecutionDispatcher
        Dispatcher to use
    context : Context
        Context to use. None for the default context.
    path : Path
        Path to use for context if applicable.
    resume : bool
        Whether to allow resuming previous workflow execution.

    Returns
    -------
    Results
        Results object created by workflow
    """
    # FIXME: Return type is not always Results
    if dispatcher is None:
        from pharmpy.workflows import default_dispatcher

        dispatcher = default_dispatcher
    if context is None:
        from pharmpy.workflows import default_context

        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        context = default_context(workflow.name, ref=path)

    # For all input models set new database and read in results
    original_input_models = []
    input_models = []
    wb = WorkflowBuilder(workflow)
    for task in workflow.tasks:
        new_inp = []

        for inp in task.task_input:
            if isinstance(inp, Model):
                original_input_models.append(inp)
                new_model = inp.replace(parent_model=inp.name)
                new_inp.append(new_model)
                input_models.append(new_model)
            else:
                new_inp.append(inp)

        new_task = task.replace(task_input=tuple(new_inp))
        wb.replace_task(task, new_task)

    insert_context(wb, context)
    workflow = Workflow(wb)

    res: T = dispatcher.run(workflow)

    if isinstance(res, Results) and not isinstance(res, ModelfitResults):
        context.store_results(res)
        if hasattr(res, 'rst_path'):
            from pharmpy.tools.reporting import create_report

            if os.name == 'nt':
                # Workaround for issue with dask versions >= 2023.7.0 on Windows
                import asyncio

                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            create_report(res, context.path)

    return res
