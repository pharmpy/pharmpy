from itertools import combinations

import pharmpy.tools.modelfit as modelfit
from pharmpy.modeling import copy_model, fix_parameters_to
from pharmpy.modeling.block_rvs import create_joint_distribution
from pharmpy.workflows import Task, Workflow


def brute_force_no_of_etas(model):
    wf = Workflow()
    model_features = dict()

    param_names = [eta.parameter_names[0] for eta in model.random_variables.etas]
    param_combos = _get_combinations(param_names, include_single=True)

    for i, combo in enumerate(param_combos, 1):
        model_name = f'iiv_no_of_etas_candidate{i}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_set_eta_to_zero = Task('set_eta_to_zero', set_eta_to_zero, combo)
        wf.add_task(task_set_eta_to_zero, predecessors=task_copy)

        model_features[model_name] = combo

    wf_fit = modelfit.create_workflow(n=i)
    wf.insert_workflow(wf_fit)
    return wf, model_features


def set_eta_to_zero(params, model):
    fix_parameters_to(model, params, 0)
    return model


def brute_force_block_structure(model):
    wf = Workflow()
    model_features = dict()

    eta_combos_single_blocks, eta_combos_multi_blocks = _get_possible_iiv_blocks(model)
    eta_combos_all = eta_combos_single_blocks + eta_combos_multi_blocks

    for i, combo in enumerate(eta_combos_all, 1):
        model_name = f'iiv_block_structure_candidate{i}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_joint_dist = Task('create_joint_dist', create_joint_dist, combo)
        wf.add_task(task_joint_dist, predecessors=task_copy)
        model_features[model_name] = combo

    wf_fit = modelfit.create_workflow(n=len(eta_combos_all))
    wf.insert_workflow(wf_fit)
    return wf, model_features


def _get_combinations(names, include_single=False):
    combos = []
    if include_single:
        start = 1
    else:
        start = 2
    for i in range(start, len(names) + 1):
        combos += [list(combo) for combo in combinations(names, i)]
    return combos


def _get_possible_iiv_blocks(model):
    eta_names = model.random_variables.etas.names
    eta_combos_single_blocks = _get_combinations(eta_names)
    if len(eta_names) < 4:
        return eta_combos_single_blocks, []

    no_of_blocks_max = int(len(eta_names) / 2)
    eta_combos_multi_blocks = []

    for i in range(2, no_of_blocks_max + 1):
        etas = []
        for combo in combinations(eta_combos_single_blocks, i):
            combo = list(combo)
            if len(flatten(combo)) == len(eta_names) and len(set(flatten(combo))) == len(eta_names):
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


def copy(name, model):
    model_copy = copy_model(model, name)
    return model_copy
