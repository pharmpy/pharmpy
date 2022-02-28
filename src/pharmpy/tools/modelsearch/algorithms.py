import re

from pharmpy.modeling import add_iiv, copy_model, create_joint_distribution, update_inits
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow

from .mfl import ModelFeatures


def exhaustive(mfl, add_iivs, iiv_as_fullblock, add_mdt_iiv):
    features = ModelFeatures(mfl)
    wf_search = Workflow()

    model_tasks = []
    model_features = dict()

    combinations = list(features.all_combinations())
    funcs = features.all_funcs()

    for i, combo in enumerate(combinations, 1):
        model_name = f'modelsearch_candidate{i}'

        task_copy = Task('copy', _copy, model_name)
        wf_search.add_task(task_copy)

        task_previous = task_copy
        for feat in combo:
            func = funcs[feat]
            task_function = Task(feat, func)
            wf_search.add_task(task_function, predecessors=task_previous)
            if add_iivs:
                task_add_iiv = Task(
                    'add_iivs', _add_iiv_to_func, feat, iiv_as_fullblock, add_mdt_iiv
                )
                wf_search.add_task(task_add_iiv, predecessors=task_function)
                task_previous = task_add_iiv
            else:
                task_previous = task_function

        wf_fit = create_fit_workflow(n=1)
        wf_search.insert_workflow(wf_fit, predecessors=task_previous)

        model_features[model_name] = tuple(combo)

        model_tasks += wf_fit.output_tasks

    return wf_search, model_tasks, model_features


def exhaustive_stepwise(mfl, add_iivs, iiv_as_fullblock, add_mdt_iiv):
    features = ModelFeatures(mfl)
    wf_search = Workflow()

    model_tasks = []
    model_features = dict()

    candidate_count = 1

    while True:
        no_of_trans = 0

        actions = _get_possible_actions(wf_search, features)

        for task, trans in actions.items():
            trans_previous, trans_possible = trans

            for feat, func in trans_possible.items():
                model_name = f'modelsearch_candidate{candidate_count}'

                task_copy = Task('copy', _copy, model_name)
                if task:
                    wf_search.add_task(task_copy, predecessors=[task])
                else:
                    wf_search.add_task(task_copy)

                wf_stepwise_step, task_transformed = _create_stepwise_workflow(
                    feat, func, add_iivs, iiv_as_fullblock, add_mdt_iiv
                )

                wf_search.insert_workflow(wf_stepwise_step, predecessors=task_copy)
                model_tasks += wf_stepwise_step.output_tasks

                features_previous = _get_previous_features(wf_search, task_transformed, features)
                model_features[model_name] = tuple(features_previous + [feat])

                candidate_count += 1
                no_of_trans += 1
        if no_of_trans == 0:
            break

    return wf_search, model_tasks, model_features


def _get_possible_actions(wf, features):
    if wf.output_tasks:
        actions = {task: _find_possible_trans(wf, task, features) for task in wf.output_tasks}
    else:
        actions = {'': _find_possible_trans(wf, None, features)}
    return actions


def _create_stepwise_workflow(feat, func, add_iivs, iiv_as_fullblock, add_mdt_iiv):
    wf_stepwise_step = Workflow()

    task_update_inits = Task('update_inits', _update_initial_estimates)
    wf_stepwise_step.add_task(task_update_inits)

    task_function = Task(feat, func)
    wf_stepwise_step.add_task(task_function, predecessors=task_update_inits)

    if add_iivs or add_mdt_iiv:
        task_add_iiv = Task('add_iivs', _add_iiv_to_func, feat, iiv_as_fullblock, add_mdt_iiv)
        wf_stepwise_step.add_task(task_add_iiv, predecessors=task_function)
        task_transformed = task_add_iiv
    else:
        task_transformed = task_function

    wf_fit = create_fit_workflow(n=1)
    wf_stepwise_step.insert_workflow(wf_fit, predecessors=task_transformed)

    return wf_stepwise_step, task_transformed


def _get_previous_features(wf, task, features):
    tasks_upstream = wf.get_upstream_tasks(task)
    tasks_upstream.reverse()
    features_previous = [
        task.name for task in tasks_upstream if task.name in features.all_funcs().keys()
    ]
    return features_previous


def _find_possible_trans(wf, task, features):
    funcs = features.all_funcs()
    if task:
        trans_previous = {
            task.name: task.function
            for task in wf.get_upstream_tasks(task)
            if task.function in funcs.values()
        }
    else:
        trans_previous = dict()

    trans_possible = {
        feat: func
        for feat, func in funcs.items()
        if _is_allowed(feat, func, trans_previous, features)
    }

    return trans_previous, trans_possible


def _is_allowed(feat_current, func_current, trans_previous, features):
    if trans_previous and func_current in trans_previous.values():
        return False
    if feat_current.startswith('PERIPHERALS'):
        return _is_allowed_peripheral(func_current, trans_previous, features)
    if not trans_previous:
        return True
    if any(func in features.get_funcs_same_type(feat_current) for func in trans_previous.values()):
        return False
    not_supported_combo = [
        ('ABSORPTION(ZO)', 'TRANSITS'),
        ('ABSORPTION(SEQ-ZO-FO)', 'TRANSITS'),
        ('ABSORPTION(SEQ-ZO-FO)', 'LAGTIME'),
        ('LAGTIME', 'TRANSITS'),
    ]
    for key, value in not_supported_combo:
        if any(
            (feat_current.startswith(key) and feat.startswith(value))
            or (feat_current.startswith(value) and feat.startswith(key))
            for feat in trans_previous.keys()
        ):
            return False
    return True


def _is_allowed_peripheral(func_current, trans_previous, features):
    n_all = list(features.peripherals.args)
    n = func_current.keywords['n']
    if trans_previous:
        n_prev = [
            func.keywords['n']
            for feat, func in trans_previous.items()
            if feat.startswith('PERIPHERALS')
        ]
    else:
        n_prev = []
    if not n_prev:
        if n == min(n_all):
            return True
        else:
            return False
    n_index = n_all.index(n)
    if n_index > 0 and n_all[n_index - 1] < n:
        return True
    return False


def _copy(name, model):
    model_copy = copy_model(model, name)
    return model_copy


def _update_initial_estimates(model):
    # FIXME: this should use dynamic workflows and not dispatch the next task
    try:
        update_inits(model)
    except ValueError:
        pass
    return model


def _add_iiv_to_func(feat, iiv_as_fullblock, add_mdt_iiv, model):
    eta_dict = {
        'ABSORPTION(ZO)': ['MAT'],
        'ABSORPTION(SEQ-ZO-FO)': ['MAT', 'MDT'],
        'ELIMINATION(ZO)': ['CLMM', 'KM'],
        'ELIMINATION(MM)': ['CLMM', 'KM'],
        'ELIMINATION(MIX-FO-MM)': ['CLMM', 'KM'],
        'LAGTIME()': ['MDT'],
    }
    parameters = []
    if add_mdt_iiv and (feat == 'LAGTIME()' or feat.startswith('TRANSITS')):
        parameters = ['MDT']
    else:
        if feat in eta_dict.keys():
            parameters = eta_dict[feat]
        elif feat.startswith('TRANSITS'):
            parameters = ['MDT']
        elif feat.startswith('PERIPHERALS'):
            no_of_peripherals = re.search(r'PERIPHERALS\((\d+)\)', feat).group(1)
            parameters = [f'VP{i}' for i in range(1, int(no_of_peripherals) + 1)] + [
                f'QP{i}' for i in range(1, int(no_of_peripherals) + 1)
            ]

    for param in parameters:
        assignment = model.statements.find_assignment(param)
        etas = {eta.name for eta in assignment.rhs_symbols}
        if etas.intersection(model.random_variables.etas.names):
            continue
        try:
            add_iiv(model, param, 'exp')
        except ValueError as e:
            if not str(e).startswith('Cannot insert parameter with already existing name'):
                raise
    if iiv_as_fullblock:
        create_joint_distribution(model)

    return model
