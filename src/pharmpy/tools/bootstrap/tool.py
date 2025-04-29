from typing import Optional

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import (
    is_simulation_model,
    resample_data,
    set_dataset,
    set_evaluation_step,
    set_initial_estimates,
    set_name,
)
from pharmpy.tools.bootstrap.results import calculate_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults


def create_workflow(
    model: Model, results: Optional[ModelfitResults] = None, resamples: int = 1, dofv: bool = False
):
    """Run bootstrap tool

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results for model
    resamples : int
        Number of bootstrap resamples
    dofv : bool
        Will evaluate bootstrap models with original dataset if set

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
        task_resample = Task('resample', resample_model, f'bs_{i + 1}', results)
        wb.add_task(task_resample, predecessors=start_task)
        task_execute = Task('run_model', run_model)
        wb.add_task(task_execute, predecessors=task_resample)
        if dofv:
            task_dofv = Task('run_dofv', run_dofv, model)
            wb.add_task(task_dofv, predecessors=task_execute)

    task_result = Task('results', post_process_results, model, results)
    wb.add_task(task_result, predecessors=wb.output_tasks)

    return Workflow(wb)


def start(context, input_model, results):
    context.log_info("Starting tool bootstrap")
    input_me = ModelEntry.create(input_model, modelfit_results=results)
    context.store_input_model_entry(input_me)
    return input_model


def resample_model(name, input_results, input_model):
    resample = resample_data(
        input_model, input_model.datainfo.id_column.name, resamples=1, replace=True, name=name
    )
    model, groups = next(resample)
    if input_results is not None:
        model = set_initial_estimates(model, input_results.parameter_estimates)
    model_entry = ModelEntry.create(model=model, parent=input_model)
    return (model_entry, groups)


def run_model(context, pair):
    me = pair[0]
    groups = pair[1]
    wf_fit = create_fit_workflow(me)
    res_me = context.call_workflow(wf_fit, f"fit-{me.model.name}")
    return (res_me, groups, None)


def run_dofv(context, input_model, tpl):
    me = tpl[0]
    groups = tpl[1]
    model = me.model
    dofv_model = set_initial_estimates(model, me.modelfit_results.parameter_estimates)
    dofv_model = set_evaluation_step(dofv_model)
    dofv_model = set_name(dofv_model, f"dofv_{model.name[3:]}")
    dofv_model = set_dataset(dofv_model, input_model.dataset, datatype='nonmem')
    dofv_me = ModelEntry.create(model=dofv_model)
    wf = create_fit_workflow(dofv_me)
    res_me = context.call_workflow(wf, dofv_model.name)
    return (me, groups, res_me)


def post_process_results(context, original_model, original_model_res, *tpls):
    model_entries = [tpl[0] for tpl in tpls]
    groups = [tpl[1] for tpl in tpls]
    dofv_mes = [tpl[2] for tpl in tpls]
    dofv_results = None if dofv_mes[0] is None else [me.modelfit_results for me in dofv_mes]
    models = [model_entry.model for model_entry in model_entries]
    modelfit_results = [model_entry.modelfit_results for model_entry in model_entries]
    res = calculate_results(
        models,
        results=modelfit_results,
        original_results=original_model_res,
        included_individuals=groups,
        dofv_results=dofv_results,
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
