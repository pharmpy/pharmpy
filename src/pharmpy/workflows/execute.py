from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

from .dispatchers import Dispatcher
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
    dispatcher : Dispatcher
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
        dispatcher = Dispatcher.select_dispatcher(dispatcher)
    if context is None:
        from pharmpy.workflows import default_context

        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        context = default_context(workflow.name, ref=path)

    wb = WorkflowBuilder(workflow)
    insert_context(wb, context)
    workflow = Workflow(wb)

    res: T = dispatcher.run(workflow, context)
    if isinstance(res, Results) and not isinstance(res, ModelfitResults):
        handle_results(res, context)

    return res


def execute_subtool(workflow: Workflow[T], context):
    assert context is not None

    res_name = context.context_path.replace('/', '_')
    res = context.dispatcher.call_workflow(workflow, unique_name=f'{res_name}_results', ctx=context)
    if isinstance(res, Results) and not isinstance(res, ModelfitResults):
        handle_results(res, context)

    return res


def handle_results(res, context):
    context.store_results(res)
    from pharmpy.tools.reporting import report_available

    if report_available(res):
        from pharmpy.tools.reporting import create_report

        if os.name == 'nt':
            # Workaround for issue with dask versions >= 2023.7.0 on Windows
            import asyncio

            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        create_report(res, context.path)
