from itertools import combinations

from pharmpy.modeling import copy_model
from pharmpy.modeling.block_rvs import create_joint_distribution
from pharmpy.tools.modelfit import create_multiple_fit_workflow
from pharmpy.workflows import Task, Workflow


def brute_force(model):
    wf = Workflow()
    eta_combos_single_blocks, eta_combos_multi_blocks = _get_iiv_combinations(model)
    eta_combos_all = eta_combos_single_blocks + eta_combos_multi_blocks
    for i, combo in enumerate(eta_combos_all, 1):
        model_name = f'iiv_candidate{i}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_joint_dist = Task('create_joint_dist', create_joint_dist, combo)
        wf.add_task(task_joint_dist, predecessors=task_copy)

    wf_fit = create_multiple_fit_workflow(n=len(eta_combos_all))
    wf.insert_workflow(wf_fit)
    return wf


def _get_iiv_combinations(model):
    eta_names = model.random_variables.etas.names
    no_of_etas = len(eta_names)
    eta_combos_single_blocks = []
    for i in range(2, no_of_etas + 1):
        eta_combos_single_blocks += [list(combo) for combo in combinations(eta_names, i)]
    if no_of_etas < 4:
        return eta_combos_single_blocks, []

    no_of_blocks_max = int(no_of_etas / 2)
    eta_combos_multi_blocks = []

    for i in range(2, no_of_blocks_max + 1):
        etas = []
        for combo in combinations(eta_combos_single_blocks, i):
            combo = list(combo)
            if len(flatten(combo)) == no_of_etas and len(set(flatten(combo))) == no_of_etas:
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
