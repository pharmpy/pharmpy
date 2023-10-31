from typing import Dict, List, Set, Tuple

import pharmpy.tools.modelfit as modelfit
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.set.partitions import partitions
from pharmpy.internals.set.subsets import non_empty_subsets
from pharmpy.model import Model, RandomVariables
from pharmpy.modeling import (
    create_joint_distribution,
    get_omegas,
    remove_iiv,
    split_joint_distribution,
)
from pharmpy.modeling.expressions import get_rv_parameters
from pharmpy.tools.common import update_initial_estimates
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import mfr


def brute_force_no_of_etas(base_model, index_offset=0, keep=None):
    wb = WorkflowBuilder(name='brute_force_no_of_etas')

    base_model = base_model.replace(description=create_description(base_model))

    iivs = base_model.random_variables.iiv
    iiv_names = iivs.names
    if keep:
        iiv_names = _remove_sublist(iiv_names, _get_eta_from_parameter(base_model, keep))

    # Remove fixed etas
    fixed_etas = _get_fixed_etas(base_model)
    iiv_names = _remove_sublist(iiv_names, fixed_etas)

    for i, to_remove in enumerate(non_empty_subsets(iiv_names), 1):
        model_name = f'iivsearch_run{i + index_offset}'
        task_candidate_entry = Task(
            'candidate_entry', create_no_of_etas_candidate_entry, model_name, to_remove
        )
        wb.add_task(task_candidate_entry)

    wf_fit = modelfit.create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit)
    return Workflow(wb)


def brute_force_block_structure(base_model, index_offset=0):
    wb = WorkflowBuilder(name='brute_force_block_structure')

    base_model = base_model.replace(description=create_description(base_model))

    iivs = base_model.random_variables.iiv
    model_no = 1 + index_offset

    fixed_etas = _get_fixed_etas(base_model)
    iiv_names = _remove_sublist(iivs.names, fixed_etas)

    for block_structure in _rv_block_structures(iiv_names):
        if _is_rv_block_structure(iivs, block_structure, fixed_etas):
            continue

        model_name = f'iivsearch_run{model_no}'
        task_candidate_entry = Task(
            'candidate_entry', create_block_structure_candidate_entry, model_name, block_structure
        )
        wb.add_task(task_candidate_entry)

        model_no += 1

    wf_fit = modelfit.create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit)
    return Workflow(wb)


def create_no_of_etas_candidate_entry(name, to_remove, model_entry):
    candidate_model = update_initial_estimates(model_entry.model, model_entry.modelfit_results)
    candidate_model = remove_iiv(candidate_model, to_remove)
    candidate_model = candidate_model.replace(
        name=name, description=create_description(candidate_model)
    )

    return ModelEntry.create(model=candidate_model, modelfit_results=None, parent=model_entry.model)


def create_block_structure_candidate_entry(name, block_structure, model_entry):
    candidate_model = update_initial_estimates(model_entry.model, model_entry.modelfit_results)
    candidate_model = create_eta_blocks(block_structure, candidate_model)
    candidate_model = candidate_model.replace(
        name=name, description=create_description(candidate_model)
    )

    return ModelEntry.create(model=candidate_model, modelfit_results=None, parent=model_entry.model)


def _rv_block_structures(etas: RandomVariables):
    # NOTE: All possible partitions of etas into block structures
    return partitions(etas)


def _is_rv_block_structure(
    etas: RandomVariables, partition: Tuple[Tuple[str, ...], ...], fixed_etas
):
    parts = set(partition)
    # Remove fixed etas from etas
    list_of_tuples = list(
        filter(
            None, list(map(lambda dist: tuple(_remove_sublist(list(dist.names), fixed_etas)), etas))
        )
    )
    return all(map(lambda dist: dist in parts, list_of_tuples))


def _create_param_dict(model: Model, dists: RandomVariables) -> Dict[str, str]:
    param_subs = {
        parameter.symbol: parameter.init for parameter in model.parameters if parameter.fix
    }
    param_dict = {}
    # FIXME: Temporary workaround, should handle IIV on eps
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


def create_eta_blocks(partition: Tuple[Tuple[str, ...], ...], model: Model):
    for part in partition:
        if len(part) == 1:
            model = split_joint_distribution(model, part)
        else:
            model = create_joint_distribution(
                model, list(part), individual_estimates=mfr(model).individual_estimates
            )
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


def _get_fixed_etas(model: Model) -> List[str]:
    fixed_omegas = get_omegas(model).fixed.names
    iivs = model.random_variables.iiv
    if len(fixed_omegas) > 0:
        fixed_etas = [
            iiv.names for iiv in iivs if str(list(iiv.variance.free_symbols)[0]) in fixed_omegas
        ]
        return [item for tup in fixed_etas for item in tup]
    else:
        return []


def _remove_sublist(list_a, list_b):
    return [x for x in list_a if x not in list_b]
