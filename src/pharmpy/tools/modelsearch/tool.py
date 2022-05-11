import pharmpy.results
import pharmpy.tools.modelsearch.algorithms as algorithms
from pharmpy.modeling import summarize_individuals, summarize_modelfit_results
from pharmpy.tools.common import summarize_tool
from pharmpy.workflows import Task, Workflow


def create_workflow(
    search_space,
    algorithm,
    iiv_strategy=0,
    rankfunc='bic',
    cutoff=None,
    model=None,
):
    """Run Modelsearch tool. For more details, see https://pharmpy.github.io/latest/modelsearch.html.

    Parameters
    ----------
    search_space : str
        Search space to test
    algorithm : str
        Algorithm to use (e.g. exhaustive)
    iiv_strategy : int
        If/how IIV should be added to candidate models. Default is 0 (no IIVs added)
    rankfunc : str
        Which ranking function should be used (OFV, AIC, BIC). Default is BIC
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    model : Model
        Pharmpy model

    Returns
    -------
    ModelSearchResults
        Modelsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_modelsearch('ABSORPTION(ZO);PERIPHERALS(1)', 'exhaustive', model=model) # doctest: +SKIP

    """
    check_input(model)
    algorithm_func = getattr(algorithms, algorithm)

    wf = Workflow()
    wf.name = 'modelsearch'

    if model:
        start_task = Task('start_modelsearch', start, model)
    else:
        start_task = Task('start_modelsearch', start)

    wf.add_task(start_task)

    wf_search, candidate_model_tasks = algorithm_func(search_space, iiv_strategy)
    wf.insert_workflow(wf_search, predecessors=wf.output_tasks)

    task_result = Task(
        'results',
        post_process_results,
        rankfunc,
        cutoff,
    )

    wf.add_task(task_result, predecessors=[start_task] + candidate_model_tasks)

    return wf


def check_input(model):
    if model is None:
        return
    try:
        cmt = model.datainfo.typeix['compartment']
    except IndexError:
        pass
    else:
        raise ValueError(
            f"Found compartment column {cmt.name} in dataset. "
            f"This is currently not supported by modelsearch. "
            f"Please remove or drop this column and try again"
        )


def start(model):
    check_input(model)
    return model


def post_process_results(rankfunc, cutoff, *models):
    res_models = []
    start_model = None
    for model in models:
        if not model.name.startswith('modelsearch_candidate'):
            start_model = model
        else:
            res_models.append(model)

    if not start_model:
        raise ValueError('Error in workflow: No starting model')

    summary_tool = summarize_tool(res_models, start_model, rankfunc, cutoff)
    summary_models = summarize_modelfit_results([start_model] + res_models)
    summary_individuals = summarize_individuals([start_model] + res_models)

    best_model_name = summary_tool['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = start_model

    res = ModelSearchResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals=summary_individuals,
        best_model=best_model,
        start_model=start_model,
        models=res_models,
    )

    return res


class ModelSearchResults(pharmpy.results.Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        best_model=None,
        start_model=None,
        models=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.best_model = best_model
        self.start_model = start_model
        self.models = models
