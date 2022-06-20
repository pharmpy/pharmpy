from itertools import combinations
from typing import Dict

import pharmpy.tools.modelfit as modelfit
from pharmpy.model import Model
from pharmpy.modeling import copy_model, remove_iiv
from pharmpy.modeling.block_rvs import create_joint_distribution, split_joint_distribution
from pharmpy.modeling.expressions import get_rv_parameter
from pharmpy.tools.common import update_initial_estimates
from pharmpy.workflows import Task, Workflow


def brute_force_no_of_etas(base_model):
    wf = Workflow()

    base_model.description = _create_description(base_model)

    iivs = base_model.random_variables.iiv
    eta_combos = _get_eta_combinations(iivs)

    for i, combo in enumerate(eta_combos, 1):
        model_name = f'iivsearch_no_of_etas_candidate{i}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)

        task_remove_eta = Task('remove_eta', remove_eta, combo)
        wf.add_task(task_remove_eta, predecessors=task_update_inits)

        task_update_description = Task('update_description', update_description)
        wf.add_task(task_update_description, predecessors=task_remove_eta)

    wf_fit = modelfit.create_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit)
    return wf


def brute_force_block_structure(base_model):
    wf = Workflow()

    base_model.description = _create_description(base_model)

    iivs = base_model.random_variables.iiv
    eta_combos = _get_eta_combinations(iivs, as_blocks=True)
    model_no = 1

    for combo in eta_combos:
        if _is_current_block_structure(iivs, combo):
            continue

        model_name = f'iivsearch_block_structure_candidate{model_no}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)

        task_joint_dist = Task('create_eta_blocks', create_eta_blocks, combo)
        wf.add_task(task_joint_dist, predecessors=task_update_inits)

        task_update_description = Task('update_description', update_description)
        wf.add_task(task_update_description, predecessors=task_joint_dist)

        model_no += 1

    wf_fit = modelfit.create_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit)
    return wf


def _get_eta_combinations(etas, as_blocks=False):
    # All possible combinations of etas
    eta_combos = []
    for i in range(1, len(etas.names) + 1):
        eta_combos += [list(combo) for combo in combinations(etas.names, i)]
    if not as_blocks:
        return eta_combos

    # All possible combinations of blocks
    block_combos = []
    for i in range(1, len(etas.names) + 1):
        for combo in combinations(eta_combos, i):
            combo = list(combo)
            etas_in_combo = _flatten(combo)
            etas_unique = set(etas_in_combo)
            if len(etas_in_combo) == len(etas.names) and len(etas_unique) == len(etas.names):
                block_combos.append(combo)
    return block_combos


def _flatten(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]


def _is_current_block_structure(etas, combos):
    for rvs, dist in etas.distributions():
        names = [rv.name for rv in rvs]
        if names not in combos:
            return False
    return True


def _iiv_param_dict(model: Model) -> Dict[str, str]:
    iiv = model.random_variables.iiv
    return {
        eta.name: get_rv_parameter(model, eta)
        for eta in iiv
        if iiv.get_variance(eta).subs(
            {parameter.symbol: parameter.init for parameter in model.parameters if parameter.fix}
        )
        != 0
    }


def _create_description(model: Model) -> str:
    param_dict = _iiv_param_dict(model)

    if len(param_dict) == 0:
        return '[]'

    blocks = []
    for rvs, _ in model.random_variables.iiv.distributions():
        rvs_names = [rv.name for rv in rvs]
        param_names = [param_dict[name] for name in rvs_names]
        blocks.append(f'[{",".join(param_names)}]')

    return '+'.join(blocks)


def remove_eta(etas, model):
    remove_iiv(model, etas)
    return model


def create_eta_blocks(combos, model):
    for combo in combos:
        if len(combo) == 1:
            split_joint_distribution(model, combo)
        else:
            create_joint_distribution(model, combo)
    return model


def copy(name, model):
    model_copy = copy_model(model, name)
    return model_copy


def update_description(model):
    description = _create_description(model)
    model.description = description
    return model
