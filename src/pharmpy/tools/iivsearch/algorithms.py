from typing import Dict, List, Set, Tuple

import pharmpy.tools.modelfit as modelfit
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.set.partitions import partitions
from pharmpy.internals.set.subsets import non_empty_subsets
from pharmpy.model import Model, RandomVariables
from pharmpy.modeling import create_joint_distribution, remove_iiv, split_joint_distribution
from pharmpy.modeling.expressions import get_rv_parameters
from pharmpy.results import mfr
from pharmpy.tools.common import update_initial_estimates
from pharmpy.workflows import Task, Workflow


def brute_force_no_of_etas(base_model, index_offset=0, keep=[]):
    wf = Workflow()

    base_model = base_model.replace(description=create_description(base_model))

    iivs = base_model.random_variables.iiv
    iiv_names = iivs.names
    if len(keep) > 0:
        iiv_names = set(iiv_names) - _get_eta_from_parameter(base_model, keep)

    for i, to_remove in enumerate(non_empty_subsets(iiv_names), 1):
        model_name = f'iivsearch_run{i + index_offset}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)

        task_remove_eta = Task('remove_eta', remove_eta, to_remove)
        wf.add_task(task_remove_eta, predecessors=task_update_inits)

        task_update_description = Task('update_description', update_description)
        wf.add_task(task_update_description, predecessors=task_remove_eta)

    wf_fit = modelfit.create_fit_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit)
    return wf


def brute_force_block_structure(base_model, index_offset=0):
    wf = Workflow()

    base_model = base_model.replace(description=create_description(base_model))

    iivs = base_model.random_variables.iiv
    model_no = 1 + index_offset

    for block_structure in _rv_block_structures(iivs):
        if _is_rv_block_structure(iivs, block_structure):
            continue

        model_name = f'iivsearch_run{model_no}'
        task_copy = Task('copy', copy, model_name)
        wf.add_task(task_copy)

        task_update_inits = Task('update_inits', update_initial_estimates)
        wf.add_task(task_update_inits, predecessors=task_copy)

        task_joint_dist = Task('create_eta_blocks', create_eta_blocks, block_structure)
        wf.add_task(task_joint_dist, predecessors=task_update_inits)

        task_update_description = Task('update_description', update_description)
        wf.add_task(task_update_description, predecessors=task_joint_dist)

        model_no += 1

    wf_fit = modelfit.create_fit_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit)
    return wf


def _rv_block_structures(etas: RandomVariables):
    # NOTE All possible partitions of etas into block structures
    return partitions(etas.names)


def _is_rv_block_structure(etas: RandomVariables, partition: Tuple[Tuple[str, ...], ...]):
    parts = set(partition)
    return all(map(lambda dist: dist.names in parts, etas))


def _create_param_dict(model: Model, dists: RandomVariables) -> Dict[str, str]:
    param_subs = {
        parameter.symbol: parameter.init for parameter in model.parameters if parameter.fix
    }
    param_dict = {}
    # FIXME temporary workaround, should handle IIV on eps
    symbs_before_ode = [symb.name for symb in model.statements.before_odes.free_symbols]
    for eta in dists.names:
        if subs(dists[eta].get_variance(eta), param_subs, simultaneous=True) != 0:
            # Skip etas that are before ODE
            if eta not in symbs_before_ode:
                continue
            param_dict[eta] = get_rv_parameters(model, eta)[0]
    return param_dict


def create_description(model: Model, iov: bool = False) -> str:
    if iov:
        dists = model.random_variables.iov
    else:
        dists = model.random_variables.iiv

    param_dict = _create_param_dict(model, dists)
    if len(param_dict) == 0:
        return '[]'

    blocks, same = [], []
    for dist in dists:
        rvs_names = dist.names
        param_names = [
            param_dict[name] for name in rvs_names if name not in same and name in param_dict.keys()
        ]
        if param_names:
            blocks.append(f'[{",".join(param_names)}]')

        if iov:
            same_names = []
            for name in rvs_names:
                same_names.extend(dists.get_rvs_with_same_dist(name).names)
            same.extend(same_names)

    return '+'.join(blocks)


def remove_eta(etas, model):
    model = remove_iiv(model, etas)
    return model


def create_eta_blocks(partition: Tuple[Tuple[str, ...], ...], model: Model):
    for part in partition:
        if len(part) == 1:
            model = split_joint_distribution(model, part)
        else:
            model = create_joint_distribution(
                model, list(part), individual_estimates=mfr(model).individual_estimates
            )
    return model


def copy(name, model):
    model_copy = model.replace(name=name)
    return model_copy


def update_description(model):
    description = create_description(model)
    model = model.replace(description=description)
    return model


def _get_eta_from_parameter(model: Model, parameters: List[str]) -> Set[str]:
    # returns list of eta names from parameter names
    iiv_set = set()
    iiv_names = model.random_variables.names
    for iiv_name in iiv_names:
        param = get_rv_parameters(model, iiv_name)
        if set(param).issubset(parameters) and len(param) > 0:
            iiv_set.add(iiv_name)
    return iiv_set
