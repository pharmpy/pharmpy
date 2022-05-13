import pharmpy.results
import pharmpy.tools.modelsearch.algorithms as algorithms
from pharmpy.modeling import summarize_modelfit_results
from pharmpy.tools.common import summarize_tool, summarize_tool_individuals
from pharmpy.workflows import Task, Workflow


def create_workflow(
    search_space,
    algorithm,
    iiv_strategy=0,
    rankfunc='bic',
    cutoff=None,
    model=None,
):
    """Run Modelsearch tool. For more details, see :ref:`modelsearch`.

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
    input_model = None
    for model in models:
        if not model.name.startswith('modelsearch_candidate'):
            input_model = model
        else:
            res_models.append(model)

    if not input_model:
        raise ValueError('Error in workflow: No input model')

    summary_tool = summarize_tool(res_models, input_model, rankfunc, cutoff)
    summary_models = summarize_modelfit_results([input_model] + res_models).sort_values(
        by=[rankfunc]
    )
    summary_individuals, summary_individuals_count = summarize_tool_individuals(
        [input_model] + res_models, summary_tool['description'], summary_tool[f'd{rankfunc}']
    )

    summary_models.sort_values(by=[f'{rankfunc}'])
    summary_individuals_count.sort_values(by=[f'd{rankfunc}'])

    best_model_name = summary_tool['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = input_model

    res = ModelSearchResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals=summary_individuals,
        summary_individuals_count=summary_individuals_count,
        best_model=best_model,
        input_model=input_model,
        models=res_models,
    )

    return res


class ModelSearchResults(pharmpy.results.Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        best_model=None,
        input_model=None,
        models=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.best_model = best_model
        self.input_model = input_model
        self.models = models
