from typing import Optional

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import is_simulation_model, resample_data
from pharmpy.tools.bootstrap.results import calculate_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults


def create_workflow(model: Model, results: Optional[ModelfitResults] = None, resamples: int = 1):
    """Run bootstrap tool

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results for model
    resamples : int
        Number of bootstrap resamples

    Returns
    -------
    BootstrapResults
        Bootstrap tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_bootstrap, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_bootstrap(model, res, resamples=500) # doctest: +SKIP
    """

    wb = WorkflowBuilder(name='bootstrap')

    start_task = Task('start', start, model, results)

    for i in range(resamples):
        task_resample = Task('resample', resample_model, f'bs_{i + 1}')
        wb.add_task(task_resample, predecessors=start_task)
        task_execute = Task('run_model', run_model)
        wb.add_task(task_execute, predecessors=task_resample)

    task_result = Task('results', post_process_results, model, results)
    wb.add_task(task_result, predecessors=wb.output_tasks)

    return Workflow(wb)


def start(context, input_model, results):
    context.log_info("Starting tool bootstrap")
    input_me = ModelEntry.create(input_model, modelfit_results=results)
    context.store_input_model_entry(input_me)
    return input_model


def resample_model(name, input_model):
    resample = resample_data(
        input_model, input_model.datainfo.id_column.name, resamples=1, replace=True, name=name
    )
    model, groups = next(resample)
    model_entry = ModelEntry.create(model=model, parent=input_model)
    return (model_entry, groups)


def run_model(context, pair):
    me = pair[0]
    groups = pair[1]
    wf_fit = create_fit_workflow(me)
    res_me = context.call_workflow(wf_fit, f"fit-{me.model.name}")
    return (res_me, groups)


def post_process_results(context, original_model, original_model_res, *pairs):
    model_entries = [pair[0] for pair in pairs]
    groups = [pair[1] for pair in pairs]
    models = [model_entry.model for model_entry in model_entries]
    modelfit_results = [model_entry.modelfit_results for model_entry in model_entries]
    res = calculate_results(
        models,
        results=modelfit_results,
        original_results=original_model_res,
        included_individuals=groups,
        dofv_results=None,
    )
    context.log_info("Finishing tool bootstrap")
    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(model, results, resamples):
    if is_simulation_model(model):
        raise ValueError('Input model is a simulation model. Bootstrap needs an estimation model')
    if resamples < 1:
        raise ValueError('The number of samples must at least one')
