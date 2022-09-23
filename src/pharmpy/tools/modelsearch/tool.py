import pharmpy.tools.modelsearch.algorithms as algorithms
from pharmpy.model import Model, Results
from pharmpy.modeling.results import RANK_TYPES
from pharmpy.tools.common import create_results
from pharmpy.utils import same_signature_as
from pharmpy.workflows import Task, Workflow

from ..mfl.parse import parse


def create_workflow(
    search_space,
    algorithm,
    iiv_strategy='absorption_delay',
    rank_type='bic',
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
    iiv_strategy : str
        If/how IIV should be added to candidate models. Possible strategies are 'no_add',
        'add_diagonal', 'fullblock', or 'absorption_delay'. Default is 'absorption_delay'
    rank_type : str
        Which ranking type should be used (OFV, AIC, BIC). Default is BIC
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
    >>> from pharmpy.tools import run_modelsearch # doctest: +SKIP
    >>> run_modelsearch('ABSORPTION(ZO);PERIPHERALS(1)', 'exhaustive', model=model) # doctest: +SKIP

    """

    wf = Workflow()
    wf.name = 'modelsearch'

    if model:
        start_task = Task('start_modelsearch', start, model)
    else:
        start_task = Task('start_modelsearch', start)

    wf.add_task(start_task)

    algorithm_func = getattr(algorithms, algorithm)
    wf_search, candidate_model_tasks = algorithm_func(search_space, iiv_strategy)
    wf.insert_workflow(wf_search, predecessors=wf.output_tasks)

    task_result = Task(
        'results',
        post_process,
        rank_type,
        cutoff,
    )

    wf.add_task(task_result, predecessors=[start_task] + candidate_model_tasks)

    return wf


def start(model):
    validate_model(model)
    return model


def post_process(rank_type, cutoff, *models):
    res_models = []
    input_model = None
    for model in models:
        if not model.name.startswith('modelsearch_candidate'):
            input_model = model
        else:
            res_models.append(model)

    if not input_model:
        raise ValueError('Error in workflow: No input model')

    res = create_results(
        ModelSearchResults, input_model, input_model, res_models, rank_type, cutoff
    )

    return res


@same_signature_as(create_workflow)
def validate_input(
    search_space,
    algorithm,
    iiv_strategy,
    rank_type,
    cutoff,
    model,
):

    if not hasattr(algorithms, algorithm):
        raise ValueError(
            f'Invalid algorithm: got "{algorithm}" of type {type(algorithm)},'
            f' must be one of {sorted(dir(algorithms))}.'
        )

    if rank_type not in RANK_TYPES:
        raise ValueError(
            f'Invalid rank_type: got "{rank_type}" of type {type(rank_type)},'
            f' must be one of {sorted(RANK_TYPES)}.'
        )

    if iiv_strategy not in algorithms.IIV_STRATEGIES:
        raise ValueError(
            f'Invalid IIV strategy: got "{iiv_strategy}" of type {type(iiv_strategy)},'
            f' must be one of {sorted(algorithms.IIV_STRATEGIES)}.'
        )

    if not isinstance(cutoff, (type(None), int, float)):
        raise TypeError(
            f'Invalid cutoff: got "{cutoff}" of type {type(cutoff)},'
            f' must be None/NULL or an int/float.'
        )

    try:
        parse(search_space)
    except:  # noqa E722
        raise ValueError(f'Invalid search_space, could not be parsed: "{search_space}"')

    validate_model(model)


def validate_model(model):
    if model is None:
        return

    if not isinstance(model, Model):
        raise TypeError(f'Invalid model: got "{model}" of type {type(model)}, must be a {Model}.')
    try:
        cmt = model.datainfo.typeix['compartment']
    except IndexError:
        pass
    else:
        raise ValueError(
            f"Found compartment column {cmt.names} in dataset. "
            f"This is currently not supported by modelsearch. "
            f"Please remove or drop this column and try again"
        )


class ModelSearchResults(Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_errors=None,
        final_model_name=None,
        models=None,
        tool_database=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors
        self.final_model_name = final_model_name
        self.models = models
        self.tool_database = tool_database
