import itertools

from pharmpy.modeling import add_estimation_step, copy_model, remove_estimation_step, set_ode_solver
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def exhaustive(methods, solvers):
    wf = Workflow()

    task_base_model = Task('create_base_model', _create_base_model)
    wf.add_task(task_base_model)
    wf_fit = create_fit_workflow(n=1)
    wf.insert_workflow(wf_fit, predecessors=task_base_model)
    task_base_model_fit = wf.output_tasks

    candidate_no = 1
    for method, solver in itertools.product(methods, solvers):
        # This is equivalent to the base model
        if not (method == 'FOCE' and solver is None):
            # Create model with original estimates
            wf_estmethod_original = _create_candidate_model_wf(
                candidate_no, method, solver, update=False
            )
            wf.insert_workflow(wf_estmethod_original, predecessors=task_base_model_fit)
            candidate_no += 1

        # Create model with updated estimates from FOCE
        wf_estmethod_update = _create_candidate_model_wf(candidate_no, method, solver, update=True)
        wf.insert_workflow(wf_estmethod_update, predecessors=task_base_model_fit)
        candidate_no += 1

    return wf, task_base_model_fit


def reduced(methods, solvers):
    wf = Workflow()

    task_start = Task('start', start)
    wf.add_task(task_start)

    candidate_no = 1
    for method, solver in itertools.product(methods, solvers):
        wf_estmethod = _create_candidate_model_wf(candidate_no, method, solver, update=False)
        wf.insert_workflow(wf_estmethod, predecessors=task_start)
        candidate_no += 1

    return wf, None


def start(model):
    return model


def _create_candidate_model_wf(candidate_no, method, solver, update):
    wf = Workflow()

    model_name = f'estmethod_run{candidate_no}'
    task_copy = Task('copy_model', _copy_model, model_name)
    wf.add_task(task_copy)

    if update:
        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)
        task_prev = task_update_inits
    else:
        task_prev = task_copy
    task_create_est_model = Task('create_est_model', _create_est_model, method, solver, update)
    wf.add_task(task_create_est_model, predecessors=task_prev)
    return wf


def _copy_model(name, model):
    return copy_model(model, name)


def _create_base_model(model):
    est_settings = _create_est_settings('FOCE')
    eval_settings = _create_eval_settings()

    base_model = copy_model(model, 'base_model')
    est_method, eval_method = est_settings['method'], eval_settings['method']
    base_model.description = _create_description(est_method, eval_method, solver=None, update=False)

    while len(base_model.estimation_steps) > 0:
        remove_estimation_step(base_model, 0)

    add_estimation_step(base_model, **est_settings)
    add_estimation_step(base_model, **eval_settings)
    return base_model


def _create_est_model(method, solver, update, model):
    est_settings = _create_est_settings(method)
    laplace = True if method == 'LAPLACE' else False
    eval_settings = _create_eval_settings(laplace)

    eval_method = eval_settings['method']
    model.description = _create_description(method, eval_method, solver=None, update=update)

    while len(model.estimation_steps) > 0:
        remove_estimation_step(model, 0)

    add_estimation_step(model, **est_settings)
    add_estimation_step(model, **eval_settings)
    if solver:
        model = set_ode_solver(model, solver)
    return model


def _create_est_settings(method):
    est_settings = {
        'method': method,
        'interaction': True,
        'laplace': False,
        'maximum_evaluations': 9999,
        'auto': True,
        'keep_every_nth_iter': 10,
    }

    if method == 'LAPLACE':
        est_settings['method'] = 'FOCE'
        est_settings['laplace'] = True

    return est_settings


def _create_eval_settings(laplace=False):
    eval_settings = {
        'method': 'IMP',
        'interaction': True,
        'evaluation': True,
        'laplace': False,
        'maximum_evaluations': 9999,
        'isample': 10000,
        'niter': 10,
        'keep_every_nth_iter': 10,
    }

    if laplace:
        eval_settings['laplace'] = True

    return eval_settings


def _create_description(est_method, eval_method, solver, update=False):
    model_description = f'{est_method},{eval_method}'
    if solver:
        model_description += f';{solver}'
    if update:
        model_description += ' (update)'
    return model_description
