import re

from pharmpy.modeling import add_iiv, copy_model, create_joint_distribution, update_inits
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow

from .mfl import ModelFeatures


def exhaustive(mfl, add_etas, etas_as_fullblock):
    features = ModelFeatures(mfl)
    wf_search = Workflow()

    model_tasks = []
    model_features = dict()

    combinations = list(features.all_combinations())
    funcs = features.all_funcs()

    for i, combo in enumerate(combinations, 1):
        model_name = f'modelsearch_candidate{i}'

        task_copy = Task('copy', copy, model_name)
        wf_search.add_task(task_copy)

        task_previous = task_copy
        for feat in combo:
            func = funcs[feat]
            task_function = Task(feat, func)
            wf_search.add_task(task_function, predecessors=task_previous)
            if add_etas:
                task_add_etas = Task('add_etas', add_etas_to_func, feat, etas_as_fullblock)
                wf_search.add_task(task_add_etas, predecessors=task_function)
                task_previous = task_add_etas
            else:
                task_previous = task_function

        wf_fit = create_fit_workflow(n=1)
        wf_search.insert_workflow(wf_fit, predecessors=task_previous)

        model_features[model_name] = tuple(combo)

        model_tasks += wf_fit.output_tasks

    return wf_search, model_tasks, model_features


def exhaustive_stepwise(mfl, add_etas, etas_as_fullblock):
    features = ModelFeatures(mfl)
    wf_search = Workflow()

    model_tasks = []
    model_features = dict()

    candidate_count = 1

    while True:
        no_of_trans = 0

        if wf_search.output_tasks:
            actions = {
                task: _find_possible_trans(wf_search, task, features)
                for task in wf_search.output_tasks
            }
        else:
            actions = {'': (dict(), features.all_funcs())}

        for task, trans in actions.items():
            trans_previous, trans_possible = trans

            for feat, func in trans_possible.items():
                model_name = f'modelsearch_candidate{candidate_count}'

                task_copy = Task('copy', copy, model_name)
                if task:
                    wf_search.add_task(task_copy, predecessors=[task])
                else:
                    wf_search.add_task(task_copy)

                task_update_inits = Task('update_inits', update_initial_estimates)
                wf_search.add_task(task_update_inits, predecessors=task_copy)

                task_function = Task(feat, func)
                wf_search.add_task(task_function, predecessors=task_update_inits)

                if add_etas:
                    task_add_etas = Task('add_etas', add_etas_to_func, feat, etas_as_fullblock)
                    wf_search.add_task(task_add_etas, predecessors=task_function)
                    task_transformed = task_add_etas
                else:
                    task_transformed = task_function

                wf_fit = create_fit_workflow(n=1)
                wf_search.insert_workflow(wf_fit, predecessors=task_transformed)

                model_tasks += wf_fit.output_tasks
                model_features[model_name] = tuple(list(trans_previous.keys()) + [feat])

                candidate_count += 1
                no_of_trans += 1
        if no_of_trans == 0:
            break

    return wf_search, model_tasks, model_features


def _find_possible_trans(wf, task, features):
    funcs = features.all_funcs()
    trans_previous = {
        task.name: task.function
        for task in wf.get_upstream_tasks(task)
        if task.function in funcs.values()
    }

    trans_possible = {
        feat: func
        for feat, func in funcs.items()
        if _is_allowed(feat, func, trans_previous, features)
    }

    return trans_previous, trans_possible


def _is_allowed(feat_current, func_current, trans_previous, features):
    if func_current in trans_previous.values():
        return False
    if any(func in features.get_funcs_same_type(feat_current) for func in trans_previous.values()):
        return False
    not_supported_combo = {
        'ABSORPTION(ZO)': 'TRANSITS',
        'ABSORPTION(SEQ-ZO-FO)': 'TRANSITS',
        'LAGTIME': 'TRANSITS',
    }
    for key, value in not_supported_combo.items():
        if any(
            (feat_current.startswith(key) and feat.startswith(value))
            or (feat_current.startswith(value) and feat.startswith(key))
            for feat in trans_previous.keys()
        ):
            return False
    return True


def copy(name, model):
    model_copy = copy_model(model, name)
    return model_copy


def update_initial_estimates(model):
    # FIXME: this should use dynamic workflows and not dispatch the next task
    try:
        update_inits(model)
    except ValueError:
        pass
    return model


def add_etas_to_func(feat, etas_as_fullblock, model):
    eta_dict = {
        'ABSORPTION(ZO)': ['MAT'],
        'ABSORPTION(SEQ-ZO-FO)': ['MAT', 'MDT'],
        'LAGTIME()': ['MDT'],
    }
    parameters = []
    try:
        parameters = eta_dict[feat]
    except KeyError:
        if feat.startswith('TRANSITS'):
            parameters = ['MDT']
        elif feat.startswith('PERIPHERALS'):
            no_of_peripherals = re.search(r'PERIPHERALS\((\d+)\)', feat).group(1)
            parameters = [f'VP{i}' for i in range(1, int(no_of_peripherals) + 1)] + [
                f'QP{i}' for i in range(1, int(no_of_peripherals) + 1)
            ]

    for param in parameters:
        try:
            add_iiv(model, param, 'exp')
        except ValueError as e:
            if not str(e).startswith('Cannot insert parameter with already existing name'):
                raise
    if etas_as_fullblock:
        create_joint_distribution(model)

    return model
