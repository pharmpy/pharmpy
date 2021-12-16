import pharmpy.results
from pharmpy.modeling import (
    add_estimation_step,
    copy_model,
    remove_estimation_step,
    summarize_modelfit_results,
    update_inits,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(methods=None, model=None):
    wf = Workflow()
    wf.name = "estmethod"

    if model is not None:
        start_task = Task('start_estmethod', start, model)
    else:
        start_task = Task('start_estmethod', start)

    task_base_model = Task('create_base_model', _create_base_model)
    wf.add_task(task_base_model, predecessors=start_task)

    wf_fit = create_fit_workflow(n=1)
    wf.insert_workflow(wf_fit, predecessors=task_base_model)

    task_base_model_fit = wf.output_tasks

    if not methods:
        methods = ['foce', 'foce_fast', 'imp', 'laplace']
    if isinstance(methods, str):
        methods = [methods]
    if 'foce' not in methods:
        methods.insert(0, 'foce')

    for method in methods:
        if method != 'foce':
            task_no_update = Task(f'create_{method}_no_update', _create_est_model, method, False)
            wf.add_task(task_no_update, predecessors=task_base_model_fit)

        task_update = Task(f'create_{method}_update', _create_est_model, method, True)
        wf.add_task(task_update, predecessors=task_base_model_fit)

    no_of_models = 2 * len(methods) - 1

    wf_fit = create_fit_workflow(n=no_of_models)
    wf.insert_workflow(wf_fit, predecessors=wf.output_tasks)

    task_post_process = Task('post_process', post_process)
    wf.add_task(
        task_post_process, predecessors=[start_task] + task_base_model_fit + wf.output_tasks
    )

    return wf


def post_process(*models):
    res_models = []
    for model in models:
        if not model.name.startswith('estmethod_'):
            start_model = model
        else:
            res_models.append(model)

    df = summarize_modelfit_results(res_models, include_all_estimation_steps=True)

    res = EstMethodResults(summary=df, start_model=start_model, models=res_models)

    return res


def start(model):
    return model


def _create_base_model(model):
    base_model = copy_model(model, 'estmethod_foce_no_update')
    _clear_estimation_steps(base_model)
    est_settings = _create_est_settings('foce')
    eval_settings = _create_eval_settings()
    add_estimation_step(base_model, **est_settings)
    add_estimation_step(base_model, **eval_settings)
    return base_model


def _create_eval_settings(laplace=False):
    # FIXME: handle tool options
    evaluation_step = {
        'method': 'IMP',
        'interaction': True,
        'laplace': laplace,
        'evaluation': True,
        'maximum_evaluations': 9999,
    }
    return evaluation_step


def _create_est_settings(method):
    # FIXME: handle tool options
    settings = dict()
    interaction = True
    laplace = False
    maximum_evaluations = 9999

    if method == 'foce_fast':
        settings['method'] = 'foce'
    elif method == 'laplace':
        settings['method'] = 'foce'
        laplace = True
    else:
        settings['method'] = method

    settings['interaction'] = interaction
    settings['laplace'] = laplace
    settings['maximum_evaluations'] = maximum_evaluations

    return settings


def _create_est_model(method, update, model):
    if update:
        model_name = f'estmethod_{method}_update'
    else:
        model_name = f'estmethod_{method}_no_update'
    est_model = copy_model(model, model_name)
    _clear_estimation_steps(est_model)
    if update:
        update_inits(est_model)
    est_settings = _create_est_settings(method)
    if method == 'laplace':
        laplace = True
    else:
        laplace = False
    eval_settings = _create_eval_settings(laplace)
    add_estimation_step(est_model, **est_settings)
    add_estimation_step(est_model, **eval_settings)
    return est_model


def _clear_estimation_steps(model):
    while len(model.estimation_steps) > 0:
        remove_estimation_step(model, 0)


class EstMethodResults(pharmpy.results.Results):
    def __init__(self, summary=None, best_model=None, start_model=None, models=None):
        self.summary = summary
        self.best_model = best_model
        self.start_model = start_model
        self.models = models

    def to_json(self, path=None, lzma=False):
        s = pharmpy.results.Results.to_json(self.summary, path, lzma)
        if s:
            return s
