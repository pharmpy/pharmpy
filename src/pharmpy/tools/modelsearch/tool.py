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
    check_input(model)
    algorithm_func = getattr(algorithms, algorithm)

    wf = Workflow()
    wf.name = 'modelsearch'

    if model:
        start_task = Task('start_modelsearch', start, model)
    else:
        start_task = Task('start_modelsearch', start)

    wf.add_task(start_task)

    wf_search, candidate_model_tasks, model_features = algorithm_func(search_space, iiv_strategy)
    wf.insert_workflow(wf_search, predecessors=wf.output_tasks)

    task_result = Task(
        'results',
        post_process_results,
        algorithm,
        rankfunc,
        cutoff,
        model_features,
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


def post_process_results(algorithm, rankfunc, cutoff, model_features, *models):
    res_models = []
    start_model = None
    for model in models:
        if not model.name.startswith('modelsearch_candidate'):
            start_model = model
        else:
            res_models.append(model)

    if not start_model:
        raise ValueError('Error in workflow: No starting model')

    if algorithm == 'reduced_stepwise':
        model_features = _update_model_features(start_model, res_models, model_features)

    summary_tool = summarize_tool(res_models, start_model, rankfunc, cutoff, model_features)
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


def _update_model_features(start_model, res_models, model_features_original):
    model_features_new = dict()
    for model in res_models:
        if model.name == start_model.name:
            feat_all = None
        else:
            feat = model_features_original[model.name]
            if (
                model.parent_model in model_features_new.keys()
                and model.parent_model != start_model.name
            ):
                feat_parent = model_features_new[model.parent_model]
                feat_all = feat_parent + (feat,)
            else:
                feat_all = (feat,)
        model_features_new[model.name] = feat_all

    return model_features_new


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
