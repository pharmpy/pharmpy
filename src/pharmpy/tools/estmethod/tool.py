from itertools import product
from pathlib import Path

import pandas as pd

import pharmpy.results
from pharmpy.modeling import (
    add_estimation_step,
    copy_model,
    remove_estimation_step,
    set_ode_solver,
    summarize_modelfit_results,
)
from pharmpy.tools.common import summarize_tool, update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(methods=None, solvers=None, update=False, model=None):
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

    methods, solvers = _format_input_options(methods, solvers)

    candidate_no = 1
    for method, solver in product(methods, solvers):
        if method != 'foce' or solver is not None:
            wf_estmethod_original = _create_estmethod_task(
                candidate_no, method=method, solver=solver, update=False
            )
            wf.insert_workflow(wf_estmethod_original, predecessors=task_base_model_fit)
            candidate_no += 1
        if update:
            wf_estmethod_update = _create_estmethod_task(
                candidate_no, method=method, solver=solver, update=True
            )
            wf.insert_workflow(wf_estmethod_update, predecessors=task_base_model_fit)
            candidate_no += 1

    wf_fit = create_fit_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit, predecessors=wf.output_tasks)

    task_post_process = Task('post_process', post_process, model)
    wf.add_task(task_post_process, predecessors=task_base_model_fit + wf.output_tasks)

    return wf


def _format_input_options(methods, solvers):
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

    return methods, solvers


def _create_estmethod_task(candidate_no, method, solver, update):
    model_name = f'estmethod_candidate{candidate_no}'
    model_description = _create_description(method, solver, update)

    wf = Workflow()
    task_copy = Task('copy_model', _copy_model, model_name, model_description)
    wf.add_task(task_copy)
    if update:
        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)
        task_prev = task_update_inits
    else:
        task_prev = task_copy
    task_create_est_model = Task('create_est_model', _create_est_model, method, solver)
    wf.add_task(task_create_est_model, predecessors=task_prev)
    return wf


def _create_description(method, solver, update=False):
    model_description = f'{method.upper()}'
    if solver:
        model_description += f'+{solver.upper()}'
    if update:
        model_description += ' (update)'
    return model_description


def post_process(input_model, *models):
    res_models = []
    base_model = None
    for model in models:
        if model.name == 'base_model':
            base_model = model
        else:
            res_models.append(model)

    # FIXME: support other rankfuncs, allow None as cutoff
    rankfunc = 'ofv'
    summary_tool = summarize_tool(
        res_models,
        base_model,
        rankfunc,
        -1000,
    )
    summary_models = summarize_modelfit_results([base_model] + res_models).sort_values(
        by=[rankfunc]
    )
    summary_settings = summarize_estimation_steps([base_model] + res_models)

    res = EstMethodResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_settings=summary_settings,
        input_model=input_model,
        models=[base_model] + res_models,
    )

    return res


def start(model):
    return model


def _create_base_model(model):
    base_model = copy_model(model, 'base_model')
    base_model.description = 'FOCE'
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


def _copy_model(name, description, model):
    model_copy = copy_model(model, name)
    model_copy.description = description
    return model_copy


def _create_est_model(method, solver, model):
    _clear_estimation_steps(model)
    est_settings = _create_est_settings(method)
    if method == 'laplace':
        laplace = True
    else:
        laplace = False
    eval_settings = _create_eval_settings(laplace)
    add_estimation_step(model, **est_settings)
    add_estimation_step(model, **eval_settings)
    if solver:
        set_ode_solver(model, solver)
    return model


def _clear_estimation_steps(model):
    while len(model.estimation_steps) > 0:
        remove_estimation_step(model, 0)


class EstMethodResults(pharmpy.results.Results):
    rst_path = Path(__file__).parent / 'report.rst'

    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_settings=None,
        best_model=None,
        input_model=None,
        models=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_settings = summary_settings
        self.best_model = best_model
        self.input_model = input_model
        self.models = models


def summarize_estimation_steps(models):
    dfs = dict()
    for model in models:
        df = model.estimation_steps.to_dataframe()
        df.index = range(1, len(df) + 1)
        dfs[model.name] = df.drop(columns=['tool_options'])

    return pd.concat(dfs.values(), keys=dfs.keys())
