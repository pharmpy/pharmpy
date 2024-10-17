from typing import Literal, Optional, Union

import pharmpy.tools.covsearch.tool as scm_tool
from pharmpy.model import Model
from pharmpy.tools.covsearch.forward import fast_forward
from pharmpy.tools.covsearch.util import init_search_state, store_input_model
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.workflows import ModelfitResults, Task, Workflow, WorkflowBuilder


def fast_scm_workflow(
    search_space: Union[str, ModelFeatures],
    max_steps: int = -1,
    p_forward: float = 0.05,
    p_backward: float = 0.01,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    max_eval: bool = False,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    naming_index_offset: Optional[int] = 0,
    statsmodels: bool = True,
    algorithm: Literal["scm-fastforward", "scm-fastforward-then-backward"] = "scm-fastforward",
    nsamples: int = 0,
    weighted_linreg: bool = True,
    lin_filter: int = 0,
):
    wb = WorkflowBuilder(name="covsearch")
    store_task = Task("store_input_model", store_input_model, model, results, max_eval)
    wb.add_task(store_task)

    init_task = Task("init", init_search_state, search_space, algorithm, nsamples)
    wb.add_task(init_task, predecessors=store_task)

    fast_forward_task = Task(
        "fast_forward",
        fast_forward,
        p_forward,
        max_steps,
        algorithm,
        nsamples,
        weighted_linreg,
        statsmodels,
        lin_filter,
    )
    wb.add_task(fast_forward_task, predecessors=init_task)
    search_output = wb.output_tasks

    if algorithm == 'scm-fastforward-then-backward':
        backward_search_task = Task(
            'backward-search',
            scm_tool.task_greedy_backward_search,
            p_backward,
            max_steps,
            naming_index_offset,
            strictness,
        )
        wb.add_task(backward_search_task, predecessors=search_output)
        search_output = wb.output_tasks

    results_task = Task("results", scm_tool.task_results, p_forward, p_backward, strictness)
    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)
