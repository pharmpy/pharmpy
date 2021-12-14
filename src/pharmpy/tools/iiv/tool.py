import pharmpy.results
import pharmpy.tools.iiv.algorithms as algorithms
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(algorithm, rankfunc='ofv', cutoff=None, model=None):
    algorithm_func = getattr(algorithms, algorithm)

    wf = Workflow()
    wf.name = "iiv"

    if model is not None:
        start_task = Task('start_iiv', start, model)
    else:
        start_task = Task('start_iiv', start)

    wf.add_task(start_task)

    if model and not model.modelfit_results:
        wf_fit = create_fit_workflow(n=1)
        wf.insert_workflow(wf_fit, predecessors=start_task)
        start_model_task = wf_fit.output_tasks
    else:
        start_model_task = [start_task]

    wf_method, model_features = algorithm_func(model)
    wf.insert_workflow(wf_method)

    task_result = Task(
        'results',
        post_process_results,
        rankfunc,
        cutoff,
        model_features,
    )

    wf.add_task(task_result, predecessors=start_model_task + wf.output_tasks)

    return wf


def start(model):
    return model


def post_process_results(rankfunc, cutoff, model_features, *models):
    start_model, res_models = models

    if isinstance(res_models, tuple):
        res_models = list(res_models)
    else:
        res_models = [res_models]

    df = pharmpy.tools.common.create_summary(
        res_models, start_model, rankfunc, cutoff, model_features
    )

    best_model_name = df['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = start_model

    res = IIVResults(summary=df, best_model=best_model, start_model=start_model, models=res_models)

    return res


def _get_iiv_block(rvs):
    return [eta.name for eta in rvs.iiv if len(eta.joint_names) > 0]


class IIVResults(pharmpy.results.Results):
    def __init__(self, summary=None, best_model=None, start_model=None, models=None):
        self.summary = summary
        self.best_model = best_model
        self.start_model = start_model
        self.models = models
