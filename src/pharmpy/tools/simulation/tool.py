from __future__ import annotations

from typing import Optional

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model, SimulationStep
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder


def create_workflow(
    model: Optional[Model] = None,
):
    """Run the simulation tool.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    SimulationResult
        SimulationResults object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import run_simulation, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> model = set_simulation(model, n=10)
    >>> run_simulations(model)   # doctest: +SKIP
    """

    wb = WorkflowBuilder(name="simulation")
    start_task = Task('run_simulations', run_simulations, model)
    wb.add_task(start_task)
    return Workflow(wb)


def run_simulations(context, model):
    context.log_info("Starting tool simulation")
    modelentry = ModelEntry.create(model=model, simulation_results=None)
    wf = create_fit_workflow(modelentry)
    wb = WorkflowBuilder(wf)
    task_results = Task('results', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    modelentry_simulation = context.call_workflow(Workflow(wb), 'results_remaining')[0]
    return modelentry_simulation.simulation_results


def bundle_results(context, *args):
    context.log_info("Finishing tool simulation")
    return args


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    model,
):
    steps = model.execution_steps
    if len(steps) == 0 or not isinstance(steps[-1], SimulationStep):
        raise ValueError("The final execution step must be a simulation")
