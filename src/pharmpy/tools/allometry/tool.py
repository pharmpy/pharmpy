from functools import partial

from pharmpy.modeling import (
    add_allometry,
    summarize_errors,
    summarize_individuals,
    summarize_individuals_count_table,
    summarize_modelfit_results,
)
from pharmpy.results import Results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(
    model=None,
    allometric_variable='WT',
    reference_value=70,
    parameters=None,
    initials=None,
    lower_bounds=None,
    upper_bounds=None,
    fixed=True,
):
    """Run allometry tool. For more details, see :ref:`allometry`.

    Parameters
    ----------
    model : Model
        Pharmpy model
    allomectric_variable : str
        Name of the variable to use for allometric scaling (default is WT)
    reference_value : float
        Reference value for the allometric variable (default is 70)
    parameters : list
        Parameters to apply scaling to (default is all CL, Q and V parameters)
    initials : list
        Initial estimates for the exponents. (default is to use 0.75 for CL and Qs and 1 for Vs)
    lower_bounds : list
        Lower bounds for the exponents. (default is 0 for all parameters)
    upper_bounds : list
        Upper bounds for the exponents. (default is 2 for all parameters)
    fixed : bool
        Should the exponents be fixed or not. (default True)

    Returns
    -------
    AllometryResults
        Allometry tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_allometry(model=model, allometric_variable='WGT')      # doctest: +SKIP

    """

    wf = Workflow()
    wf.name = "allometry"
    if model is not None:
        start_task = Task('start_allometry', start, model)
    else:
        start_task = Task('start_allometry', start)
    _add_allometry = partial(
        _add_allometry_on_model,
        allometric_variable=allometric_variable,
        reference_value=reference_value,
        parameters=parameters,
        initials=initials,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        fixed=fixed,
    )
    task_add_allometry = Task('add allometry', _add_allometry)
    wf.add_task(task_add_allometry, predecessors=start_task)
    fit_wf = create_fit_workflow(n=1)
    wf.insert_workflow(fit_wf, predecessors=task_add_allometry)
    results_task = Task('results', results)
    wf.add_task(results_task, predecessors=[start_task] + fit_wf.output_tasks)
    return wf


def start(model):
    return model


def _add_allometry_on_model(
    input_model,
    allometric_variable,
    reference_value,
    parameters,
    initials,
    lower_bounds,
    upper_bounds,
    fixed,
):
    model = input_model.copy()
    add_allometry(
        model,
        allometric_variable=allometric_variable,
        reference_value=reference_value,
        parameters=parameters,
        initials=initials,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        fixed=fixed,
    )

    model.name = "scaled_model"
    model.description = "Allometry model"
    return model


def results(start_model, model):
    summods = summarize_modelfit_results([start_model, model])
    suminds = summarize_individuals([start_model, model])
    sumcount = summarize_individuals_count_table(df=suminds)
    sumerrs = summarize_errors([start_model, model])
    res = AllometryResults(
        summary_models=summods,
        summary_individuals=suminds,
        summary_individuals_count=sumcount,
        summary_errors=sumerrs,
        best_model=model,
    )
    return res


class AllometryResults(Results):
    def __init__(
        self,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_errors=None,
        best_model=None,
    ):
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors
        self.best_model = best_model
