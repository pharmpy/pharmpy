from itertools import combinations

from pharmpy.modeling import copy_model
from pharmpy.modeling.block_rvs import create_joint_distribution
from pharmpy.tools.modelfit import create_multiple_fit_workflow
from pharmpy.workflows import Task, Workflow


def brute_force(model):
    wf = Workflow()
    eta_combos = _get_iiv_combinations(model)

    for i, combo in enumerate(eta_combos, 1):
        model_name = f'iiv_candidate{i}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_joint_dist = Task('create_joint_dist', create_joint_dist, combo)
        wf.add_task(task_joint_dist, predecessors=task_copy)

    wf_fit = create_multiple_fit_workflow(n=len(eta_combos))
    wf.insert_workflow(wf_fit)
    return wf


def _create_joint_distribution(rvs, model):
    return create_joint_distribution(model, rvs)


def _get_iiv_combinations(model):
    eta_names = model.random_variables.etas.names
    no_of_etas = len(eta_names)
    eta_combos = []
    for i in range(2, no_of_etas + 1):
        eta_combos += [list(combo) for combo in combinations(eta_names, i)]
    return eta_combos


def create_joint_dist(list_of_etas, model):
    return create_joint_distribution(model, list_of_etas)


def copy(name, model):
    model_copy = copy_model(model, name)
    return model_copy
