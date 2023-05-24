from typing import Optional

from pharmpy.model import Model
from pharmpy.modeling import resample_data
from pharmpy.results import ModelfitResults
from pharmpy.tools.bootstrap.results import calculate_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


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

    wf = Workflow()
    wf.name = 'bootstrap'

    for i in range(resamples):
        task_resample = Task('resample', resample_model, model, f'bs_{i + 1}')
        wf.add_task(task_resample)

    wf_fit = create_fit_workflow(n=resamples)
    wf.insert_workflow(wf_fit)

    task_result = Task('results', post_process_results, model)
    wf.add_task(task_result, predecessors=wf.output_tasks)

    return wf


def resample_model(model, name):
    resample = resample_data(model, model.datainfo.id_column.name, resamples=1, name=name)
    model, _ = next(resample)
    return model


def post_process_results(original_model, *models):
    res = calculate_results(
        models, original_model=original_model, included_individuals=None, dofv_results=None
    )
    return res
