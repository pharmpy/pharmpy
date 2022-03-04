from pathlib import Path

import pandas as pd

import pharmpy.results
from pharmpy.modeling import (
    add_estimation_step,
    copy_model,
    remove_estimation_step,
    set_ode_solver,
    summarize_modelfit_results,
    update_inits,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(methods=None, solvers=None, model=None):
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
        methods = ['foce', 'fo', 'imp', 'impmap', 'its', 'saem', 'laplace', 'bayes']
    elif isinstance(methods, str):
        methods = [methods]
    elif isinstance(methods, list):
        methods = [method.lower() for method in methods]

    if solvers == 'all':
        solvers = [None, 'cvodes', 'dgear', 'dverk', 'ida', 'lsoda', 'lsodi']
    elif isinstance(solvers, str) or not solvers:
        solvers = [solvers]
    elif isinstance(solvers, list):
        solvers = [solver.lower() for solver in solvers]
    if None not in solvers:
        solvers.insert(0, None)

    for method in methods:
        for solver in solvers:
            if solver:
                task_name = f'create_{method.upper()}_{solver.upper()}'
            else:
                task_name = f'create_{method.upper()}'
            if method != 'foce' or solver is not None:
                task_no_update = Task(
                    f'{task_name}_raw_inits', _create_est_model, method, solver, False
                )
                wf.add_task(task_no_update, predecessors=task_base_model_fit)

            task_update = Task(f'{task_name}_update_inits', _create_est_model, method, solver, True)
            wf.add_task(task_update, predecessors=task_base_model_fit)

    wf_fit = create_fit_workflow(n=len(wf.output_tasks))
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

    summary = summarize_modelfit_results(res_models)
    settings = summarize_estimation_steps(res_models)

    res = EstMethodResults(
        summary=summary, settings=settings, start_model=start_model, models=res_models
    )

    return res


def start(model):
    return model


def _create_base_model(model):
    base_model = copy_model(model, 'estmethod_FOCE_raw_inits')
    _clear_estimation_steps(base_model)
    est_settings = _create_est_settings('foce')
    eval_settings = _create_eval_settings()
    add_estimation_step(base_model, **est_settings)
    add_estimation_step(base_model, **eval_settings)
    return base_model


def _create_eval_settings(laplace=False):
    eval_settings = {
        'method': 'imp',
        'interaction': True,
        'laplace': laplace,
        'evaluation': True,
        'maximum_evaluations': 9999,
        'isample': 10000,
        'niter': 10,
        'keep_every_nth_iter': 10,
    }
    return eval_settings


def _create_est_settings(method):
    est_settings = dict()
    interaction = True
    laplace = False
    maximum_evaluations = 9999
    auto = True
    keep_every_nth_iter = 10

    if method == 'laplace':
        est_settings['method'] = 'foce'
        laplace = True
    else:
        est_settings['method'] = method

    est_settings['interaction'] = interaction
    est_settings['laplace'] = laplace
    est_settings['maximum_evaluations'] = maximum_evaluations
    est_settings['auto'] = auto
    est_settings['keep_every_nth_iter'] = keep_every_nth_iter

    return est_settings


def _create_est_model(method, solver, update, model):
    if solver:
        model_name = f'estmethod_{method.upper()}_{solver.upper()}'
    else:
        model_name = f'estmethod_{method.upper()}'
    if update:
        model_name += '_update_inits'
    else:
        model_name += '_raw_inits'
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
    if solver:
        set_ode_solver(est_model, solver)
    return est_model


def _clear_estimation_steps(model):
    while len(model.estimation_steps) > 0:
        remove_estimation_step(model, 0)


class EstMethodResults(pharmpy.results.Results):
    rst_path = Path(__file__).parent / 'report.rst'

    def __init__(self, summary=None, settings=None, best_model=None, start_model=None, models=None):
        self.summary = summary
        self.settings = settings
        self.best_model = best_model
        self.start_model = start_model
        self.models = models

    def sorted_by_ofv(self):
        df = self.summary[['ofv', 'runtime_total', 'estimation_runtime']].sort_values(by=['ofv'])
        return df


def summarize_estimation_steps(models):
    dfs = dict()
    for model in models:
        df = model.estimation_steps.to_dataframe()
        df.index = range(1, len(df) + 1)
        dfs[model.name] = df.drop(columns=['tool_options'])

    return pd.concat(dfs.values(), keys=dfs.keys())
