from typing import Literal, Optional, Union

import pharmpy.tools.covsearch.tool as scm_tool
from pharmpy.model import Model
from pharmpy.tools.covsearch.forward import fast_forward
from pharmpy.tools.covsearch.util import init_search_state, store_input_model
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.workflows import Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

NAME_WF = 'covsearch'


def samba_workflow(
    search_space: Union[str, ModelFeatures],
    max_steps: int = -1,
    alpha: float = 0.05,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    max_eval: bool = False,
    statsmodels: bool = True,
    algorithm: Literal['samba-saem', 'samba-foce'] = 'samba-saem',
    nsamples: int = 10,
    weighted_linreg: bool = False,
    lin_filter: int = 2,
):
    """
    Workflow builder for SAMBA covariate search algorithm.
    """

    wb = WorkflowBuilder(name=NAME_WF)

    # Initiate model and search state
    store_task = Task("store_input_model", store_input_model, model, results, max_eval)
    wb.add_task(store_task)

    init_task = Task("init", init_search_state, search_space, algorithm, nsamples)
    wb.add_task(init_task, predecessors=store_task)

    # SAMBA search task
    samba_search_task = Task(
        "samba_search",
        fast_forward,
        alpha,
        max_steps,
        algorithm,
        nsamples,
        weighted_linreg,
        statsmodels,
        lin_filter,
    )
    wb.add_task(samba_search_task, predecessors=init_task)
    search_output = wb.output_tasks

    # Results task
    results_task = Task(
        "results",
        samba_task_results,
        alpha,
    )
    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


def samba_task_results(context, p_forward, state):
    # set p_backward and strictness to None
    return scm_tool.task_results(context, p_forward, p_backward=None, strictness=None, state=state)
