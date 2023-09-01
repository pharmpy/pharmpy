import itertools

from pharmpy.modeling import add_estimation_step, remove_estimation_step, set_ode_solver
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, WorkflowBuilder


def exhaustive(est_methods, solvers, uncert_methods):
    wb = WorkflowBuilder()

    task_start = Task('start', start)
    wb.add_task(task_start)

    candidate_no = 1
    for est_method, solver, uncert_method in itertools.product(
        est_methods, solvers, uncert_methods
    ):
        wf_estmethod = _create_candidate_model_wf(
            candidate_no, est_method, solver, uncert_method, update=False
        )
        wb.insert_workflow(wf_estmethod, predecessors=task_start)
        candidate_no += 1

    return Workflow(wb), None


def exhaustive_with_update(est_methods, solvers, uncert_methods):
    wb = WorkflowBuilder()

    task_base_model = Task('create_base_model', _create_base_model)
    wb.add_task(task_base_model)
    wf_fit = create_fit_workflow(n=1)
    wb.insert_workflow(wf_fit, predecessors=task_base_model)
    task_base_model_fit = wb.output_tasks

    candidate_no = 1
    for est_method, solver, uncert_method in itertools.product(
        est_methods, solvers, uncert_methods
    ):
        # This is equivalent to the base model
        if not (est_method == 'FOCE' and solver is None):
            # Create model with original estimates
            wf_estmethod_original = _create_candidate_model_wf(
                candidate_no, est_method, solver, uncert_method, update=False
            )
            wb.insert_workflow(wf_estmethod_original, predecessors=task_base_model_fit)
            candidate_no += 1

        # Create model with updated estimates from FOCE
        wf_estmethod_update = _create_candidate_model_wf(
            candidate_no, est_method, solver, uncert_method, update=True
        )
        wb.insert_workflow(wf_estmethod_update, predecessors=task_base_model_fit)
        candidate_no += 1

    return Workflow(wb), task_base_model_fit


def exhaustive_only_eval(est_methods, solvers, uncert_methods):
    wb = WorkflowBuilder()

    task_start = Task('start', start)
    wb.add_task(task_start)

    candidate_no = 1
    for est_method, solver, uncert_method in itertools.product(
        est_methods, solvers, uncert_methods
    ):
        wf_estmethod = _create_candidate_model_wf(
            candidate_no, est_method, solver, uncert_method, update=False, is_eval_candidate=True
        )
        wb.insert_workflow(wf_estmethod, predecessors=task_start)
        candidate_no += 1

    return Workflow(wb), None


def start(model):
    return model


def _create_candidate_model_wf(
    candidate_no, est_method, solver, uncert_method, update, is_eval_candidate=False
):
    wb = WorkflowBuilder()

    model_name = f'estmethod_run{candidate_no}'
    task_copy = Task('copy_model', _copy_model, model_name)
    wb.add_task(task_copy)

    if update:
        task_update_inits = Task('update_inits', update_initial_estimates)
        wb.add_task(task_update_inits, predecessors=task_copy)
        task_prev = task_update_inits
    else:
        task_prev = task_copy
    task_create_candidate = Task(
        'create_candidate',
        _create_candidate_model,
        est_method,
        solver,
        uncert_method,
        update,
        is_eval_candidate,
    )
    wb.add_task(task_create_candidate, predecessors=task_prev)
    return Workflow(wb)


def _copy_model(name, model):
    return model.replace(name=name)


def _create_base_model(model):
    est_settings = _create_est_settings('FOCE')
    eval_settings = _create_eval_settings()

    base_model = model.replace(name='base_model')
    est_method, eval_method = est_settings['est_method'], eval_settings['est_method']

    if eval_method is not None:
        uncert_method = eval_settings['uncert_method']
    else:
        uncert_method = est_settings['uncert_method']

    base_model = base_model.replace(
        description=_create_description(
            [est_method, eval_method], uncert_method=uncert_method, solver=None, update=False
        )
    )

    while len(base_model.estimation_steps) > 0:
        base_model = remove_estimation_step(base_model, 0)

    base_model = add_estimation_step(base_model, **est_settings)
    base_model = add_estimation_step(base_model, **eval_settings)
    return base_model


def _create_candidate_model(est_method, solver, uncert_method, update, is_eval_candidate, model):
    est_settings = _create_est_settings(est_method, is_eval_candidate)
    laplace = True if est_method == 'LAPLACE' else False
    eval_settings = _create_eval_settings(laplace, uncert_method)

    eval_method = eval_settings['est_method']
    model = model.replace(
        description=_create_description(
            [est_method, eval_method], solver=solver, uncert_method=uncert_method, update=update
        )
    )

    while len(model.estimation_steps) > 0:
        model = remove_estimation_step(model, 0)

    model = add_estimation_step(model, **est_settings)
    model = add_estimation_step(model, **eval_settings)
    if solver:
        model = set_ode_solver(model, solver)
    return model


def _create_est_settings(est_method, is_eval_candidate=False):
    est_settings = {
        'est_method': est_method,
        'interaction': True,
        'laplace': False,
        'auto': True,
        'keep_every_nth_iter': 10,
        'uncert_method': None,
    }

    if est_method == 'LAPLACE':
        est_settings['est_method'] = 'FOCE'
        est_settings['laplace'] = True

    if is_eval_candidate:
        est_settings['evaluation'] = True
    else:
        est_settings['maximum_evaluations'] = 9999

    return est_settings


def _create_eval_settings(laplace=False, uncert_method=None):
    eval_settings = {
        'est_method': 'IMP',
        'interaction': True,
        'evaluation': True,
        'laplace': False,
        'maximum_evaluations': 9999,
        'isample': 10000,
        'niter': 10,
        'keep_every_nth_iter': 10,
        'uncert_method': uncert_method,
    }

    if laplace:
        eval_settings['laplace'] = True

    return eval_settings


def _create_description(est_methods, solver, uncert_method, update=False):
    model_description = ','.join(est_methods)
    if solver:
        model_description += f';{solver}'
    if uncert_method:
        model_description += f';{uncert_method}'
    if update:
        model_description += ' (update)'
    return model_description
