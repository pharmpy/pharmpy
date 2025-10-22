from typing import Optional

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import (
    set_initial_estimates,
    set_simulation,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import run_subtool
from pharmpy.tools.vpc.results import calculate_results
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

from .frem import prepare_evaluation_model, prepare_frem_model


def create_workflow(
    model: Model,
    results: Optional[ModelfitResults] = None,
    samples: int = 20,
    stratify: Optional[str] = None,
    frem: bool = False,
):
    """Run VPC

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results for model
    samples : int
        Number of samples
    stratify : str
        Column to stratify on
    frem : bool
        Should we run the special vpc procedure a FREM model?

    Returns
    -------
    VPCResults
        VPC results object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_vpc, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> res = load_example_modelfit_results("pheno")
    >>> run_vpc(model, res, samples=200) # doctest: +SKIP
    """

    wb = WorkflowBuilder(name='vpc')

    start_task = Task('start', start, model, results)
    wb.add_task(start_task)

    if frem:
        frem_prep_eval_task = Task('prepare_frem_evaluation', prepare_evaluation_model)
        wb.add_task(frem_prep_eval_task, predecessors=[start_task])
        eval_wfl = create_fit_workflow(n=1)
        wb.insert_workflow(eval_wfl, predecessors=[frem_prep_eval_task])
        frem_prep_task = Task('prepare_frem_model', prepare_frem_model)
        wb.add_task(frem_prep_task, predecessors=wb.output_tasks)

    simulation_task = Task('simulation', simulation, samples)
    wb.add_task(simulation_task, predecessors=wb.output_tasks)

    task_result = Task('results', post_process_results, stratify)
    wb.add_task(task_result, predecessors=wb.output_tasks)

    return Workflow(wb)


def start(context, input_model, results):
    context.log_info("Starting tool vpc")
    input_me = ModelEntry.create(input_model, modelfit_results=results)
    context.store_input_model_entry(input_me)
    return input_me


def simulation(context, samples, input_me):
    # NOTE: The seed is set to be in range for NONMEM
    rng = context.create_rng(1)
    sim_model = set_simulation(input_me.model, n=samples, seed=context.spawn_seed(rng, n=31))
    if input_me.modelfit_results is not None:
        sim_model = set_initial_estimates(sim_model, input_me.modelfit_results.parameter_estimates)
    sim_res = run_subtool('simulation', context, name='simulation', model=sim_model)
    simulation_data = sim_res.table
    return input_me.model, simulation_data


def post_process_results(context, stratify, piped):
    input_model, simulation_data = piped
    res = calculate_results(input_model, simulation_data, stratify=stratify)
    context.log_info("Finishing tool vpc")
    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(model, results, samples):
    if samples < 1:
        raise ValueError('The number of samples must at least one')
