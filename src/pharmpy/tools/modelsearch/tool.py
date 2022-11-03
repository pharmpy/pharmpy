from typing import Optional, Union

import pharmpy.tools.modelsearch.algorithms as algorithms
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model, Results
from pharmpy.modeling.results import RANK_TYPES
from pharmpy.results import ModelfitResults
from pharmpy.tools import summarize_modelfit_results
from pharmpy.tools.common import create_results
from pharmpy.workflows import Task, Workflow

from ..mfl.parse import parse


def create_workflow(
    search_space: str,
    algorithm: str,
    iiv_strategy: str = 'absorption_delay',
    rank_type: str = 'bic',
    cutoff: Optional[Union[float, int]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
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
    results : ModelfitResults
        Results for model
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
    >>> res = model.modelfit_results
    >>> run_modelsearch('ABSORPTION(ZO);PERIPHERALS(1)', 'exhaustive', results=res, model=model) # doctest: +SKIP

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
    return model


def post_process(rank_type, cutoff, *models):
    res_models = []
    input_model = None
    for model in models:
        if not model.name.startswith('modelsearch_run'):
            input_model = model
        else:
            res_models.append(model)

    if not input_model:
        raise ValueError('Error in workflow: No input model')

    res = create_results(
        ModelSearchResults, input_model, input_model, res_models, rank_type, cutoff
    )

    summary_input = summarize_modelfit_results(input_model.modelfit_results)
    summary_candidates = summarize_modelfit_results(
        [model.modelfit_results for model in res_models]
    )

    res.summary_models = pd.concat([summary_input, summary_candidates], keys=[0, 1], names=['step'])

    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    search_space,
    algorithm,
    iiv_strategy,
    rank_type,
    model,
):

    if not hasattr(algorithms, algorithm):
        raise ValueError(
            f'Invalid `algorithm`: got `{algorithm}`, must be one of {sorted(dir(algorithms))}.'
        )

    if rank_type not in RANK_TYPES:
        raise ValueError(
            f'Invalid `rank_type`: got `{rank_type}`, must be one of {sorted(RANK_TYPES)}.'
        )

    if iiv_strategy not in algorithms.IIV_STRATEGIES:
        raise ValueError(
            f'Invalid `iiv_strategy`: got `{iiv_strategy}`,'
            f' must be one of {sorted(algorithms.IIV_STRATEGIES)}.'
        )

    try:
        parse(search_space)
    except:  # noqa E722
        raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')

    if model is not None:
        try:
            cmt = model.datainfo.typeix['compartment']
        except IndexError:
            pass
        else:
            raise ValueError(
                f"Invalid `model`: found compartment column {cmt.names} in dataset. "
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
