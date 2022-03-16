from itertools import combinations

import pharmpy.tools.modelfit as modelfit
from pharmpy.modeling import copy_model, remove_iiv
from pharmpy.modeling.block_rvs import create_joint_distribution, split_joint_distribution
from pharmpy.workflows import Task, Workflow


def brute_force_no_of_etas(iivs):
    wf = Workflow()
    model_features = dict()

    eta_names = [eta.name for eta in iivs]
    eta_combos = _get_combinations(eta_names, include_single=True)

    for i, combo in enumerate(eta_combos, 1):
        model_name = f'iiv_no_of_etas_candidate{i}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_remove_eta = Task('remove_eta', remove_eta, combo)
        wf.add_task(task_remove_eta, predecessors=task_copy)

        model_features[model_name] = combo

    wf_fit = modelfit.create_workflow(n=i)
    wf.insert_workflow(wf_fit)
    return wf, model_features


def remove_eta(etas, model):
    remove_iiv(model, etas)
    return model


def brute_force_block_structure(iivs):
    wf = Workflow()
    model_features = dict()

    eta_combos_single_blocks, eta_combos_multi_blocks = _get_possible_iiv_blocks(iivs)
    eta_combos_all = [None] + eta_combos_single_blocks + eta_combos_multi_blocks

    model_no = 1
    for combo in eta_combos_all:
        # Do not run model with same block structure as start model
        if combo:
            if _is_current_block_structure(iivs, combo):
                continue
        else:
            if all(len(rv.joint_names) == 0 for rv in iivs):
                continue

        model_name = f'iiv_block_structure_candidate{model_no}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        if combo is None:
            task_joint_dist = Task('split_joint_dist', split_joint_dist)
            wf.add_task(task_joint_dist, predecessors=task_copy)
            model_features[model_name] = [[eta.name] for eta in iivs]
        else:
            task_joint_dist = Task('create_joint_dist', create_joint_dist, combo)
            wf.add_task(task_joint_dist, predecessors=task_copy)
            model_features[model_name] = combo
        model_no += 1
    wf_fit = modelfit.create_workflow(n=len(model_features))
    wf.insert_workflow(wf_fit)
    return wf, model_features


def _is_current_block_structure(iivs, list_of_etas):
    if not isinstance(list_of_etas[0], list):
        list_of_etas = [list_of_etas]

    for eta_block in list_of_etas:
        eta_name = eta_block[0]
        if iivs[eta_name].joint_names != eta_block:
            return False
    return True


def _get_combinations(names, include_single=False):
    combos = []
    if include_single:
        start = 1
    else:
        start = 2
    for i in range(start, len(names) + 1):
        combos += [list(combo) for combo in combinations(names, i)]
    return combos


def _get_possible_iiv_blocks(iivs):
    eta_names = iivs.names
    eta_combos_single_blocks = _get_combinations(eta_names)
    if len(eta_names) < 4:
        return eta_combos_single_blocks, []

    no_of_blocks_max = int(len(eta_names) / 2)
    eta_combos_multi_blocks = []

    for i in range(2, no_of_blocks_max + 1):
        etas = []
        for combo in combinations(eta_combos_single_blocks, i):
            combo = list(combo)
            no_of_etas_in_combo = len(flatten(combo))
            no_of_etas_unique = len(set(flatten(combo)))
            if no_of_etas_in_combo == len(eta_names) and no_of_etas_unique == len(eta_names):
                etas.append(combo)
        eta_combos_multi_blocks += etas

    return eta_combos_single_blocks, eta_combos_multi_blocks


def flatten(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]


def create_joint_dist(list_of_etas, model):
    if isinstance(list_of_etas[0], list):
        for eta_block in list_of_etas:
            create_joint_distribution(model, eta_block)
    else:
        create_joint_distribution(model, list_of_etas)
    return model


def split_joint_dist(model):
    split_joint_distribution(model)
    return model


def copy(name, model):
    model_copy = copy_model(model, name)
    return model_copy
