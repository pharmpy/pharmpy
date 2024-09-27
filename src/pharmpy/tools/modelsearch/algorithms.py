from typing import Any

from pharmpy.modeling import (
    add_allometry,
    add_iiv,
    add_pk_iiv,
    create_joint_distribution,
    set_upper_bounds,
)
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder

from ..mfl.helpers import all_combinations, get_funcs_same_type, key_to_str

ALGORITHMS = frozenset(('exhaustive', 'exhaustive_stepwise', 'reduced_stepwise'))
IIV_STRATEGIES = frozenset(('no_add', 'add_diagonal', 'fullblock', 'absorption_delay'))


def exhaustive(mfl_funcs, iiv_strategy: str, allometry=None):
    # TODO: rewrite using _create_model_workflow
    wb_search = WorkflowBuilder()

    model_tasks = []

    combinations = list(all_combinations(mfl_funcs))

    for i, combo in enumerate(combinations, 1):
        model_name = f'modelsearch_run{i}'

        # NOTE: The different functions need to be extracted first otherwise an error is raised
        funcs = set(mfl_funcs[feat] for feat in combo)

        task_create_candidate = Task(
            'create_candidate',
            create_candidate_exhaustive,
            model_name,
            combo,
            funcs,
            iiv_strategy,
            allometry,
        )
        wb_search.add_task(task_create_candidate)

        wf_fit = create_fit_workflow(n=1)
        wb_search.insert_workflow(wf_fit, predecessors=task_create_candidate)

        model_tasks += wf_fit.output_tasks

    return Workflow(wb_search), model_tasks


def exhaustive_stepwise(
    mfl_funcs, iiv_strategy: str, wb_search=None, tool_name="modelsearch", allometry=None
):
    if not wb_search:
        wb_search = WorkflowBuilder()
    model_tasks = []

    while True:
        no_of_trans = 0
        actions = _get_possible_actions(wb_search, mfl_funcs)
        for task_parent, feat_new in actions.items():
            for feat in feat_new:
                model_no = len(model_tasks) + 1
                model_name = f'{tool_name}_run{model_no}'

                task_create_candidate = Task(
                    key_to_str(feat),
                    create_candidate_stepwise,
                    model_name,
                    feat,
                    mfl_funcs[feat],
                    iiv_strategy,
                    allometry,
                )

                if task_parent:
                    wb_search.add_task(task_create_candidate, predecessors=[task_parent])
                else:
                    wb_search.add_task(task_create_candidate)

                wf_fit = create_fit_workflow(n=1)
                wb_search.insert_workflow(wf_fit, predecessors=task_create_candidate)

                model_tasks += wf_fit.output_tasks

                no_of_trans += 1
        if no_of_trans == 0:
            break

    return Workflow(wb_search), model_tasks


def reduced_stepwise(mfl_funcs, iiv_strategy: str, allometry=None):
    wb_search = WorkflowBuilder()
    model_tasks = []

    while True:
        no_of_trans = 0
        actions = _get_possible_actions(wb_search, mfl_funcs)
        groups = _find_same_model_groups(wb_search, mfl_funcs)
        if len(groups) > 1:
            for group in groups:
                # Only add collector nodes to tasks with possible actions (i.e. not to leaf nodes)
                if all(len(actions[task]) > 0 for task in group):
                    task_best_model = Task('choose_best_model', get_best_model)
                    wb_search.add_task(task_best_model, predecessors=group)
            # Overwrite actions with new collector nodes
            actions = _get_possible_actions(wb_search, mfl_funcs)

        for task_parent, feat_new in actions.items():
            for feat in feat_new:
                model_no = len(model_tasks) + 1
                model_name = f'modelsearch_run{model_no}'

                task_create_candidate = Task(
                    key_to_str(feat),
                    create_candidate_stepwise,
                    model_name,
                    feat,
                    mfl_funcs[feat],
                    iiv_strategy,
                    allometry,
                )

                if task_parent:
                    wb_search.add_task(task_create_candidate, predecessors=[task_parent])
                else:
                    wb_search.add_task(task_create_candidate)

                wf_fit = create_fit_workflow(n=1)
                wb_search.insert_workflow(wf_fit, predecessors=task_create_candidate)

                model_tasks += wf_fit.output_tasks

                no_of_trans += 1
        if no_of_trans == 0:
            break

    return Workflow(wb_search), model_tasks


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


def get_best_model(*model_entries):
    models_with_res = [model_entry for model_entry in model_entries if model_entry.modelfit_results]
    if models_with_res:
        return min(models_with_res, key=lambda x: x.modelfit_results.ofv)
    # FIXME: Should be None, maybe dynamic workflows are needed
    return model_entries[0]


def _get_possible_actions(wf, mfl_funcs):
    actions = {}
    if wf.output_tasks:
        tasks = wf.output_tasks
    else:
        tasks = ['']
    for task in tasks:
        if task:
            feat_previous = _get_previous_features(wf, task, mfl_funcs)
        else:
            feat_previous = []
        trans_possible = [
            feat
            for feat, func in mfl_funcs.items()
            if _is_allowed(feat, func, feat_previous, mfl_funcs)
        ]

        actions[task] = trans_possible
    return actions


def _get_previous_features(wf, task, mfl_funcs):
    tasks_upstream = wf.get_upstream_tasks(task)
    tasks_upstream.reverse()
    tasks_dict = {key_to_str(key): key for key in mfl_funcs.keys()}
    features_previous = [
        tasks_dict[task.name] for task in tasks_upstream if task.name in tasks_dict
    ]
    return features_previous


def create_candidate_exhaustive(model_name, combo, funcs, iiv_strategy, allometry, model_entry):
    input_model, input_res = model_entry.model, model_entry.modelfit_results
    model = _update_name_and_description(model_name, combo, model_entry)
    model = update_initial_estimates(model, input_res)
    for feat, func in zip(combo, funcs):
        model = _apply_transformation(feat, func, model)
        if iiv_strategy != 'no_add':
            model = _add_iiv_to_func(iiv_strategy, model, model_entry)
    model = _add_allometry(model, allometry)
    return ModelEntry.create(model, modelfit_results=None, parent=input_model)


def create_candidate_stepwise(model_name, feat, func, iiv_strategy, allometry, model_entry):
    input_model, input_res = model_entry.model, model_entry.modelfit_results
    model = _update_name_and_description(model_name, (feat,), model_entry)
    model = update_initial_estimates(model, input_res)
    model = _apply_transformation(feat, func, model)
    if iiv_strategy != 'no_add':
        model = _add_iiv_to_func(iiv_strategy, model, model_entry)
    model = _add_allometry(model, allometry)
    return ModelEntry.create(model, modelfit_results=None, parent=input_model)


def _add_allometry(model, allometry):
    if allometry is None:
        return model
    model = add_allometry(model, allometry.covariate, allometry.reference)
    return model


def _apply_transformation(feat, func, model):
    old_params = set(model.parameters)
    model = func(model)
    if feat[0] == 'PERIPHERALS':
        new_params = set(model.parameters)
        diff = new_params - old_params
        peripheral_params = {
            param.name: 999999
            for param in diff
            if param.name.startswith('POP_Q') or param.name.startswith('POP_V')
        }
        model = set_upper_bounds(model, peripheral_params)
    return model


def _is_allowed(feat_current, func_current, feat_previous, mfl_funcs):
    func_type = get_funcs_same_type(mfl_funcs, feat_current)
    # Check if current function is in previous transformations
    if feat_current in feat_previous:
        return False
    # Check if peripheral transformation is allowed
    if feat_current[0] == 'PERIPHERALS':
        peripheral_previous = [
            mfl_funcs[feat] for feat in feat_previous if feat[0] == 'PERIPHERALS'
        ]
        allowed_p = _is_allowed_peripheral(func_current, peripheral_previous, mfl_funcs)
        return allowed_p
    # Equivalent to changing the absorption rate model to instantaneous absorption
    if feat_current == ('TRANSITS', 0, 'NODEPOT'):
        return False
    # Check if any functions of the same type has been used
    if any(mfl_funcs[feat] in func_type for feat in feat_previous):
        return False
    # No transformations have been made
    if not feat_previous:
        return True
    # Combinations to skip
    not_supported_combo = [
        (
            ('ABSORPTION', 'FO'),
            (
                'TRANSITS',
                1,
                'NODEPOT',
            ),
        ),
        (('ABSORPTION', 'ZO'), ('TRANSITS',)),
        (('ABSORPTION', 'SEQ-ZO-FO'), ('TRANSITS',)),
        (('ABSORPTION', 'SEQ-ZO-FO'), ('LAGTIME', 'ON')),
        (('ABSORPTION', 'INST'), ('LAGTIME', 'ON')),
        (('ABSORPTION', 'INST'), ('TRANSITS',)),
        (('LAGTIME', 'ON'), ('TRANSITS',)),
    ]
    for feat_1, feat_2 in not_supported_combo:
        if any(
            (feat_current[: len(feat_1)] == feat_1 and feat[: len(feat_2)] == feat_2)
            or (feat_current[: len(feat_2)] == feat_2 and feat[: len(feat_1)] == feat_1)
            for feat in feat_previous
        ):
            return False
    return True


def _is_allowed_peripheral(func_current, peripheral_previous, mfl_funcs):
    n_all: list[Any] = list(args[0] for (kind, *args) in mfl_funcs if kind == 'PERIPHERALS')
    n = func_current.keywords['n']
    if peripheral_previous:
        n_prev = [func.keywords['n'] for func in peripheral_previous]
    else:
        n_prev = []
    if not n_prev:
        return n == min(n_all)
    n_index = n_all.index(n)
    return n_index > 0 and n_all[n_index - 1] < n


def _update_name_and_description(name, features, me):
    model = me.model
    features_str = ';'.join(map(key_to_str, features))
    if not model.description or (me.parent is not None and me.parent.name == model.name):
        description = features_str
    else:
        description = f'{model.description};{features_str}'
    return model.replace(name=name, description=description)


def _add_iiv_to_func(iiv_strategy, model, input_model_entry):
    sset, rvs = model.statements, model.random_variables
    if iiv_strategy == 'add_diagonal' or iiv_strategy == 'fullblock':
        try:
            model = add_pk_iiv(model, initial_estimate=0.01)
        except ValueError as e:
            if str(e) == 'New parameter inits are not valid':
                raise ValueError(
                    f'{model.name}: {e} (add_pk_iiv, parent: {input_model_entry.model.name}'
                )
        if iiv_strategy == 'fullblock':
            try:
                model = create_joint_distribution(
                    model,
                    individual_estimates=input_model_entry.modelfit_results.individual_estimates,
                )
            except ValueError as e:
                if str(e) == 'New parameter inits are not valid':
                    raise ValueError(
                        f'{model.name}: {e} '
                        f'(create_joint_distribution, '
                        f'parent: {input_model_entry.parent.name})'
                    )
    else:
        assert iiv_strategy == 'absorption_delay'
        mdt = sset.find_assignment('MDT')
        if mdt and not mdt.expression.free_symbols.intersection(rvs.free_symbols):
            model = add_iiv(model, 'MDT', 'exp', initial_estimate=0.01)

    return model
