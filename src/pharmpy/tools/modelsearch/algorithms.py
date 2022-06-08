from pharmpy.modeling import (
    add_iiv,
    add_pk_iiv,
    copy_model,
    create_joint_distribution,
    set_upper_bounds,
)
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow

from .mfl import ModelFeatures


def exhaustive(search_space, iiv_strategy):
    # TODO: rewrite using _create_model_workflow
    features = ModelFeatures(search_space)
    wf_search = Workflow()

    model_tasks = []

    combinations = list(features.all_combinations())
    funcs = features.all_funcs()

    for i, combo in enumerate(combinations, 1):
        model_name = f'modelsearch_candidate{i}'

        features = ';'.join(combo)
        task_copy = Task('copy', _copy, model_name, features)
        wf_search.add_task(task_copy)

        task_previous = task_copy
        for feat in combo:
            func = funcs[feat]
            task_function = Task(feat, func)
            wf_search.add_task(task_function, predecessors=task_previous)
            if iiv_strategy != 'no_add':
                task_add_iiv = Task('add_iivs', _add_iiv_to_func, iiv_strategy)
                wf_search.add_task(task_add_iiv, predecessors=task_function)
                task_previous = task_add_iiv
            else:
                task_previous = task_function

        wf_fit = create_fit_workflow(n=1)
        wf_search.insert_workflow(wf_fit, predecessors=task_previous)

        model_tasks += wf_fit.output_tasks

    return wf_search, model_tasks


def exhaustive_stepwise(search_space, iiv_strategy):
    mfl_features = ModelFeatures(search_space)
    mfl_funcs = mfl_features.all_funcs()

    wf_search = Workflow()
    model_tasks = []

    while True:
        no_of_trans = 0
        actions = _get_possible_actions(wf_search, mfl_features)
        for task_parent, feat_new in actions.items():
            for feat in feat_new:
                model_no = len(model_tasks) + 1
                model_name = f'modelsearch_candidate{model_no}'

                wf_create_model, task_function = _create_model_workflow(
                    model_name, feat, mfl_funcs[feat], iiv_strategy
                )

                if task_parent:
                    wf_search.insert_workflow(wf_create_model, predecessors=[task_parent])
                else:
                    wf_search += wf_create_model

                model_tasks += wf_create_model.output_tasks

                no_of_trans += 1
        if no_of_trans == 0:
            break

    return wf_search, model_tasks


def reduced_stepwise(mfl, iiv_strategy):
    mfl_features = ModelFeatures(mfl)
    mfl_funcs = mfl_features.all_funcs()

    wf_search = Workflow()
    model_tasks = []

    while True:
        no_of_trans = 0
        actions = _get_possible_actions(wf_search, mfl_features)
        if all(len(feat_new) > 0 for feat_new in actions.values()):
            groups = _find_same_model_groups(wf_search, mfl_funcs)
            if len(groups) > 1:
                for group in groups:
                    task_best_model = Task('choose_best_model', _get_best_model)
                    wf_search.add_task(task_best_model, predecessors=group)
                # Overwrite actions with new collector nodes
                actions = _get_possible_actions(wf_search, mfl_features)

        for task_parent, feat_new in actions.items():
            for feat in feat_new:
                model_no = len(model_tasks) + 1
                model_name = f'modelsearch_candidate{model_no}'

                wf_create_model, task_transformed = _create_model_workflow(
                    model_name, feat, mfl_funcs[feat], iiv_strategy
                )

                if task_parent:
                    wf_search.insert_workflow(wf_create_model, predecessors=[task_parent])
                else:
                    wf_search += wf_create_model

                model_tasks += wf_create_model.output_tasks

                no_of_trans += 1
        if no_of_trans == 0:
            break

    return wf_search, model_tasks


def _find_same_model_groups(wf, mfl_funcs):
    tasks = wf.output_tasks
    tasks_removed = []
    all_groups = []
    for task_start in tasks:
        if id(task_start) in tasks_removed:
            continue
        else:
            tasks_removed += [id(task_start)]
        group = [task_start]
        features_previous = set(_get_previous_features(wf, task_start, mfl_funcs))
        for task in tasks:
            if (
                set(_get_previous_features(wf, task, mfl_funcs)) == features_previous
                and id(task) not in tasks_removed
            ):
                tasks_removed += [id(task)]
                group += [task]
        if len(group) > 1:
            all_groups += [group]
    return all_groups


def _get_best_model(*models):
    models_with_res = [model for model in models if model.modelfit_results]
    if models_with_res:
        return min(models, key=lambda x: x.modelfit_results.ofv)
    # FIXME: should be None, maybe dynamic workflows are needed
    return models[0]


def _get_possible_actions(wf, mfl_features):
    actions = dict()
    if wf.output_tasks:
        tasks = wf.output_tasks
    else:
        tasks = ['']
    for task in tasks:
        mfl_funcs = mfl_features.all_funcs()
        if task:
            feat_previous = _get_previous_features(wf, task, mfl_funcs)
        else:
            feat_previous = dict()

        trans_possible = [
            feat
            for feat, func in mfl_funcs.items()
            if _is_allowed(feat, func, feat_previous, mfl_features)
        ]

        actions[task] = trans_possible
    return actions


def _get_previous_features(wf, task, mfl_funcs):
    tasks_upstream = wf.get_upstream_tasks(task)
    tasks_upstream.reverse()
    features_previous = [task.name for task in tasks_upstream if task.name in mfl_funcs.keys()]
    return features_previous


def _create_model_workflow(model_name, feat, func, iiv_strategy):
    wf_stepwise_step = Workflow()

    task_copy = Task('copy', _copy, model_name, feat)
    wf_stepwise_step.add_task(task_copy)

    task_update_inits = Task('update_inits', update_initial_estimates)
    wf_stepwise_step.add_task(task_update_inits, predecessors=task_copy)

    task_function = Task(feat, _apply_transformation, feat, func)
    wf_stepwise_step.add_task(task_function, predecessors=task_update_inits)

    if iiv_strategy != 'no_add':
        task_add_iiv = Task('add_iivs', _add_iiv_to_func, iiv_strategy)
        wf_stepwise_step.add_task(task_add_iiv, predecessors=task_function)
        task_to_fit = task_add_iiv
    else:
        task_to_fit = task_function

    wf_fit = create_fit_workflow(n=1)
    wf_stepwise_step.insert_workflow(wf_fit, predecessors=task_to_fit)

    return wf_stepwise_step, task_function


def _apply_transformation(feat, func, model):
    old_params = set(model.parameters)
    func(model)
    if feat.startswith('PERIPHERALS'):
        new_params = set(model.parameters)
        diff = new_params - old_params
        peripheral_params = {
            param.name: 999999
            for param in diff
            if param.name.startswith('POP_Q') or param.name.startswith('POP_V')
        }
        set_upper_bounds(model, peripheral_params)
    return model


def _is_allowed(feat_current, func_current, feat_previous, mfl_features):
    mfl_funcs = mfl_features.all_funcs()
    func_type = mfl_features.get_funcs_same_type(feat_current)
    # Check if current function is in previous transformations
    if feat_current in feat_previous:
        return False
    # Check if peripheral transformation is allowed
    if feat_current.startswith('PERIPHERALS'):
        peripheral_previous = [
            mfl_funcs[feat] for feat in feat_previous if feat.startswith('PERIPHERALS')
        ]
        return _is_allowed_peripheral(func_current, peripheral_previous, mfl_features)
    # Check if any functions of the same type has been used
    if any(mfl_funcs[feat] in func_type for feat in feat_previous):
        return False
    # No transformations have been made
    if not feat_previous:
        return True
    # Combinations to skip
    not_supported_combo = [
        ('ABSORPTION(ZO)', 'TRANSITS'),
        ('ABSORPTION(SEQ-ZO-FO)', 'TRANSITS'),
        ('ABSORPTION(SEQ-ZO-FO)', 'LAGTIME'),
        ('LAGTIME', 'TRANSITS'),
    ]
    for feat_1, feat_2 in not_supported_combo:
        if any(
            (feat_current.startswith(feat_1) and feat.startswith(feat_2))
            or (feat_current.startswith(feat_2) and feat.startswith(feat_1))
            for feat in feat_previous
        ):
            return False
    return True


def _is_allowed_peripheral(func_current, peripheral_previous, mfl_features):
    n_all = list(mfl_features.peripherals.args)
    n = func_current.keywords['n']
    if peripheral_previous:
        n_prev = [func.keywords['n'] for func in peripheral_previous]
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


def _copy(name, features, model):
    model_copy = copy_model(model, name)
    if not model.description:
        model_copy.description = features
    else:
        model_copy.description = f'{model.description};{features}'
    return model_copy


def _add_iiv_to_func(iiv_strategy, model):
    sset, rvs = model.statements, model.random_variables
    if iiv_strategy == 'diagonal' or iiv_strategy == 'fullblock':
        try:
            add_pk_iiv(model, initial_estimate=0.01)
        except ValueError as e:
            if str(e) == 'New parameter inits are not valid':
                raise ValueError(
                    f'{model.name}: {e} ' f'(add_pk_iiv, ' f'parent: {model.parent_model})'
                )
        if iiv_strategy == 'fullblock':
            try:
                create_joint_distribution(model)
            except ValueError as e:
                if str(e) == 'New parameter inits are not valid':
                    raise ValueError(
                        f'{model.name}: {e} '
                        f'(create_joint_distribution, '
                        f'parent: {model.parent_model})'
                    )
    elif iiv_strategy == 'absorption_delay':
        mdt = sset.find_assignment('MDT')
        if mdt and not mdt.expression.free_symbols.intersection(rvs.free_symbols):
            add_iiv(model, 'MDT', 'exp', initial_estimate=0.01)
    else:
        raise ValueError(
            f'Invalid IIV strategy (must be "no_add", "diagonal", "fullblock", or '
            f'"absorption_delay"): {iiv_strategy}'
        )

    return model
