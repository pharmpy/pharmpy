from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, TypeVar

from .dispatchers import Dispatcher
from .results import ModelfitResults, Results
from .workflow import Workflow, WorkflowBuilder, insert_context

T = TypeVar('T')


def execute_workflow(
    workflow: Workflow[T], dispatcher=None, context=None, path=None, resume=False
) -> Optional[T]:
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
        from pharmpy.workflows import Context

        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        assert workflow.name is not None
        context = Context.select_context(None, workflow.name, ref=str(path))

    wb = WorkflowBuilder(workflow)
    insert_context(wb, context)
    workflow = Workflow(wb)

    res = dispatcher.run(workflow, context)
    if isinstance(res, Results):
        handle_results(res, context)

    return res  # pyright: ignore [reportReturnType]


def execute_subtool(workflow: Workflow[T], context):
    assert context is not None

    res_name = context.context_path.replace('/', '_')
    res = context.dispatcher.call_workflow(
        workflow, unique_name=f'{res_name}_results', context=context
    )
    if isinstance(res, Results) and not isinstance(res, ModelfitResults):
        handle_results(res, context)

    return res


def handle_results(res: Results, context) -> None:
    context.store_results(res)
    from pharmpy.tools.reporting import report_available

    if report_available(res):
        from pharmpy.tools.reporting import create_report

        if os.name == 'nt':
            # Workaround for issue with dask versions >= 2023.7.0 on Windows
            import asyncio

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                )
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        create_report(res, context.path)
