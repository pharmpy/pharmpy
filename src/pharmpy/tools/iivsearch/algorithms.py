from itertools import combinations

import pharmpy.tools.modelfit as modelfit
from pharmpy.modeling import copy_model, remove_iiv
from pharmpy.modeling.block_rvs import create_joint_distribution, split_joint_distribution
from pharmpy.statements import ODESystem
from pharmpy.tools.common import update_initial_estimates
from pharmpy.workflows import Task, Workflow


def brute_force_no_of_etas(base_model):
    wf = Workflow()

    iivs = base_model.random_variables.iiv
    eta_combos = _get_eta_combinations(iivs)
    param_dict = _get_param_names(base_model)

    base_model.description = _create_feature_str(param_dict, iivs.names)

    for i, combo in enumerate(eta_combos, 1):
        etas_remaining = set(iivs.names).difference(combo)
        model_name = f'iivsearch_no_of_etas_candidate{i}'
        model_description = _create_feature_str(param_dict, etas_remaining)
        task_copy = Task('copy', copy, model_name, model_description)
        wf.add_task(task_copy)

        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)

        task_remove_eta = Task('remove_eta', remove_eta, combo)
        wf.add_task(task_remove_eta, predecessors=task_update_inits)

    wf_fit = modelfit.create_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit)
    return wf


def brute_force_block_structure(base_model):
    wf = Workflow()

    iivs = base_model.random_variables.iiv
    eta_combos = _get_eta_combinations(iivs, as_blocks=True)
    param_dict = _get_param_names(base_model)
    model_no = 1

    for combo in eta_combos:
        model_description = _create_feature_str(param_dict, combo, as_blocks=True)
        if _is_current_block_structure(iivs, combo):
            base_model.description = model_description
            continue

        model_name = f'iivsearch_block_structure_candidate{model_no}'
        task_copy = Task('copy', copy, model_name, model_description)
        wf.add_task(task_copy)

        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)

        task_joint_dist = Task('create_eta_blocks', create_eta_blocks, combo)
        wf.add_task(task_joint_dist, predecessors=task_update_inits)

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


def _get_param_names(model):
    sset = model.statements

    param_dict = dict()
    for eta in model.random_variables.iiv:
        s = _find_assignment(sset, eta)
        if len(s.expression.free_symbols) > 1:
            param_dict[eta.name] = s.symbol.name
        else:
            s = _find_assignment(sset, s.symbol)
            param_dict[eta.name] = s.symbol.name
    return param_dict


def _find_assignment(sset, symb_target):
    for s in sset:
        if isinstance(s, ODESystem):
            continue
        expr_symbs = [symb.name for symb in s.expression.free_symbols]
        if symb_target.name in expr_symbs:
            return s


def _create_feature_str(param_dict, combo, as_blocks=False):
    if as_blocks:
        blocks = []
        for block in combo:
            features = ','.join([param_dict[eta] for eta in block])
            blocks.append(f'[{features}]')
        return '+'.join(blocks)
    else:
        features = ','.join([param_dict[eta] for eta in combo])
        return f'[{features}]'


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


def copy(name, description, model):
    model_copy = copy_model(model, name)
    model_copy.description = description
    return model_copy
