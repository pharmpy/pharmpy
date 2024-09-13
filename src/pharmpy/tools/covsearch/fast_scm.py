import pharmpy.tools.covsearch.tool as scm_tool
from pharmpy.tools.covsearch.fast_forward import fast_forward
from pharmpy.tools.covsearch.util import init_search_state, store_input_model
from pharmpy.workflows import Task, Workflow, WorkflowBuilder


def fast_scm_workflow(
    model,
    results,
    max_eval,
    search_space,
    strictness,
    naming_index_offset,
    algorithm="scm-fastforward",
    nsamples=0,
    p_forward=0.05,
    max_steps=-1,
    p_backward=0.01,
):
    wb = WorkflowBuilder(name="covsearch")
    store_task = Task("store_input_model", store_input_model, model, results, max_eval)
    wb.add_task(store_task)

    init_task = Task("init", init_search_state, search_space, algorithm, nsamples)
    wb.add_task(init_task, predecessors=store_task)

    fast_forward_task = Task("fast_forward", fast_forward, p_forward, max_steps, nsamples)
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
