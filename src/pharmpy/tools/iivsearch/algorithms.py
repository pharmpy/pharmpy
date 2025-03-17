from collections import defaultdict
from typing import Optional

import pharmpy.tools.modelfit as modelfit
from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
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
from pharmpy.tools.run import calculate_mbic_penalty, get_rankval
from pharmpy.workflows import ModelEntry, ModelfitResults, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import mfr


def td_exhaustive_no_of_etas(base_model, index_offset=0, keep=None, param_mapping=None):
    wb = WorkflowBuilder(name='td_exhaustive_no_of_etas')

    base_model = base_model.replace(
        description=create_description(base_model, param_dict=param_mapping)
    )
    eta_names = get_eta_names(base_model, keep, param_mapping)

    for i, to_remove in enumerate(non_empty_subsets(eta_names), 1):
        model_name = f'iivsearch_run{i + index_offset}'
        if param_mapping:
            etas = param_mapping.keys()
            param_names = param_mapping.values()
        else:
            etas = tuple()
            param_names = tuple()
        task_candidate_entry = Task(
            'candidate_entry',
            create_no_of_etas_candidate_entry,
            model_name,
            to_remove,
            etas,
            param_names,
            False,
            None,
        )
        wb.add_task(task_candidate_entry)

    wf_fit = modelfit.create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit)
    return Workflow(wb)


def get_eta_names(model, keep, param_mapping):
    iiv_symbs = model.random_variables.iiv.free_symbols
    etas = model.statements.before_odes.free_symbols.intersection(iiv_symbs)
    # Extract to have correct order, necessary for create_joint_distribution
    eta_names = model.random_variables[etas].names
    if keep and param_mapping:
        keep = tuple(k for k, v in param_mapping.items() if v in keep)

    if keep:
        eta_names = _remove_sublist(eta_names, _get_eta_from_parameter(model, keep))

    # Remove fixed etas
    fixed_etas = _get_fixed_etas(model)
    etas = _remove_sublist(eta_names, fixed_etas)
    return etas


def bu_stepwise_no_of_etas(
    base_model,
    strictness,
    index_offset=0,
    input_model_entry=None,
    list_of_algorithms=None,
    rank_type=None,
    E_p=None,
    E_q=None,
    keep=None,
    param_mapping=None,
    clearance_parameter="",
):
    wb = WorkflowBuilder(name='bu_stepwise_no_of_etas')
    stepwise_task = Task(
        "stepwise_BU_task",
        stepwise_BU_algorithm,
        base_model,
        index_offset,
        strictness,
        input_model_entry,
        list_of_algorithms,
        rank_type,
        E_p,
        E_q,
        keep,
        param_mapping,
        clearance_parameter,
    )
    wb.add_task(stepwise_task)
    return wb


def stepwise_BU_algorithm(
    context,
    base_model,
    index_offset,
    strictness,
    input_model_entry,
    list_of_algorithms,
    rank_type,
    E_p,
    E_q,
    keep,
    param_mapping,
    clearance_parameter,
    base_model_entry,
):
    base_model = base_model.replace(
        description=create_description(base_model, param_dict=param_mapping)
    )

    iivs = base_model.random_variables.iiv
    iiv_names = iivs.names  # All ETAs in the base model

    # Remove fixed etas
    fixed_etas = _get_fixed_etas(base_model)
    iiv_names = _remove_sublist(iiv_names, fixed_etas)

    if keep and param_mapping:
        keep = tuple(k for k, v in param_mapping.items() if v in keep)

    base_parameter = _extract_clearance_parameter(
        base_model, param_mapping, clearance_parameter, iiv_names
    )

    if keep:
        parameters_to_ignore = _get_eta_from_parameter(base_model, keep)
    else:
        parameters_to_ignore = {base_parameter}

    # Create and run first model with a single ETA on base_parameter
    bu_base_model_wb = WorkflowBuilder(name='create_and_fit_BU_base_model')
    to_be_removed = [i for i in iiv_names if i not in parameters_to_ignore]
    model_name = f'iivsearch_run{1 + index_offset}'
    index_offset += 1
    if param_mapping:
        etas = param_mapping.keys()
        param_names = param_mapping.values()
    else:
        etas = tuple()
        param_names = tuple()

    bu_base_entry = Task(
        'candidate_entry',
        create_no_of_etas_candidate_entry,
        model_name,
        to_be_removed,
        etas,
        param_names,
        True,
        input_model_entry,
        base_model_entry,
    )
    bu_base_model_wb.add_task(bu_base_entry)
    wf_fit = modelfit.create_fit_workflow(n=len(bu_base_model_wb.output_tasks))
    bu_base_model_wb.insert_workflow(wf_fit)
    best_model_entry = context.call_workflow(Workflow(bu_base_model_wb), 'fit_BU_base_model')
    # Filter IIV names to contain all combination with the base parameter in it
    iiv_names_to_add = list(non_empty_subsets(iiv_names))
    if parameters_to_ignore != {""}:
        iiv_names_to_add = [
            i for i in iiv_names_to_add if all(p in i for p in parameters_to_ignore)
        ]

    # Invert the list to REMOVE ETAs from the base model instead of adding to the
    # single ETA model
    iiv_names_to_remove = [tuple(i for i in iiv_names if i not in x) for x in iiv_names_to_add]

    # Remove largest step removing all ETAs but base_parameter
    max_step = max(len(element) for element in iiv_names_to_remove)
    if base_parameter:
        iiv_names_to_remove = [i for i in iiv_names_to_remove if len(i) != max_step]

    # Dictionary of all possible candidates of each step
    step_dict = defaultdict(list)
    for step in iiv_names_to_remove:
        step_dict[max_step - len(step) + 1].append(step)
    # Assert to be sorted in correct order
    step_dict = dict(sorted(step_dict.items()))

    search_space = []
    if any('no_of_etas' in algorithm for algorithm in list_of_algorithms):
        search_space.append('iiv_diag')
    if any('block' in algorithm for algorithm in list_of_algorithms):
        search_space.append('iiv_block')

    previous_index = index_offset
    previous_removed = to_be_removed
    all_modelentries = [best_model_entry]
    for step_number, steps in step_dict.items():
        effect_dict = {}
        temp_wb = WorkflowBuilder(name=f'stepwise_bu_{step_number}')
        for to_remove in steps:
            if all(e in previous_removed for e in to_remove):  # Filter unwanted effects
                model_name = f'iivsearch_run{previous_index + 1}'
                effect_dict[model_name] = to_remove
                task_candidate_entry = Task(
                    'candidate_entry',
                    create_no_of_etas_candidate_entry,
                    model_name,
                    to_remove,
                    etas,
                    param_names,
                    False,
                    best_model_entry,
                    base_model_entry,
                )
                temp_wb.add_task(task_candidate_entry)
                previous_index += 1
        wf_fit = modelfit.create_fit_workflow(n=len(temp_wb.output_tasks))
        temp_wb.insert_workflow(wf_fit, predecessors=temp_wb.output_tasks)
        task_gather = Task('gather', lambda *model_entries: model_entries)
        temp_wb.add_task(task_gather, predecessors=temp_wb.output_tasks)
        new_candidate_modelentries = context.call_workflow(
            Workflow(temp_wb), f'td_exhaustive_no_of_etas-fit-{step_number}'
        )
        all_modelentries.extend(new_candidate_modelentries)
        old_best_name = best_model_entry.model.name
        for me in new_candidate_modelentries:
            if rank_type == 'mbic':
                rank_name = 'bic'
            else:
                rank_name = rank_type
            rankval_me = get_rankval(
                me.model, me.modelfit_results, strictness, rank_type=rank_name, bic_type='iiv'
            )
            rankval_best = get_rankval(
                best_model_entry.model,
                best_model_entry.modelfit_results,
                strictness,
                rank_type=rank_name,
                bic_type='iiv',
            )
            if not np.isnan(rankval_me) and not np.isnan(rankval_best):
                if rank_type == 'mbic':
                    rankval_me += calculate_mbic_penalty(
                        me.model, search_space, base_model=base_model, keep=keep, E_p=E_p, E_q=E_q
                    )
                    rankval_best += calculate_mbic_penalty(
                        best_model_entry.model,
                        search_space,
                        base_model=base_model,
                        keep=keep,
                        E_p=E_p,
                        E_q=E_q,
                    )
                if rankval_best > rankval_me:
                    best_model_entry = me
                    previous_removed = effect_dict[me.model.name]
        if old_best_name == best_model_entry.model.name:
            return all_modelentries

    return all_modelentries


def td_exhaustive_block_structure(base_model, index_offset=0, param_mapping=None):
    wb = WorkflowBuilder(name='td_exhaustive_block_structure')

    base_model = base_model.replace(
        description=create_description(base_model, param_dict=param_mapping)
    )

    model_no = 1 + index_offset

    fixed_etas = _get_fixed_etas(base_model)
    eta_names = get_eta_names(base_model, [], param_mapping)
    etas_base_model = base_model.random_variables[eta_names]
    for block_structure in _rv_block_structures(eta_names):
        if _is_rv_block_structure(etas_base_model, block_structure, fixed_etas):
            continue

        model_name = f'iivsearch_run{model_no}'
        if param_mapping:
            etas = param_mapping.keys()
            param_names = param_mapping.values()
        else:
            etas = tuple()
            param_names = tuple()
        task_candidate_entry = Task(
            'candidate_entry',
            create_block_structure_candidate_entry,
            model_name,
            block_structure,
            etas,
            param_names,
        )
        wb.add_task(task_candidate_entry)

        model_no += 1

    wf_fit = modelfit.create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit)
    return Workflow(wb)


def create_no_of_etas_candidate_entry(
    name, to_remove, etas, param_names, base_parent, best_model_entry, base_model_entry
):
    if best_model_entry is None:
        best_model_entry = base_model_entry
    param_mapping = {k: v for k, v in zip(etas, param_names)}
    candidate_model = update_initial_estimates(
        base_model_entry.model, best_model_entry.modelfit_results
    )
    candidate_model = remove_iiv(candidate_model, to_remove)
    candidate_model = candidate_model.replace(name=name)
    candidate_model = candidate_model.replace(
        description=create_description(candidate_model, param_dict=param_mapping)
    )

    if base_parent:
        parent = base_model_entry.model
    else:
        parent = best_model_entry.model

    return ModelEntry.create(model=candidate_model, modelfit_results=None, parent=parent)


def create_block_structure_candidate_entry(name, block_structure, etas, param_names, model_entry):
    param_mapping = {k: v for k, v in zip(etas, param_names)}
    candidate_model = update_initial_estimates(model_entry.model, model_entry.modelfit_results)
    candidate_model = create_eta_blocks(
        block_structure, candidate_model, model_entry.modelfit_results
    )
    candidate_model = candidate_model.replace(
        name=name, description=create_description(candidate_model, param_dict=param_mapping)
    )

    return ModelEntry.create(model=candidate_model, modelfit_results=None, parent=model_entry.model)


def _extract_clearance_parameter(model, param_mapping, clearance_parameter, iiv_names):
    if param_mapping:  # Linearized model
        cl_eta_list = list(k for k in param_mapping if param_mapping[k] == clearance_parameter)
    else:
        cl_eta_list = list(_get_eta_from_parameter(model, [clearance_parameter]))
    if cl_eta_list and cl_eta_list[0] in iiv_names:
        base_parameter = cl_eta_list[0]
    else:
        base_parameter = ""  # Start with no ETAs at all

    return base_parameter


def _rv_block_structures(etas):
    # NOTE: All possible partitions of etas into block structures
    return partitions(etas)


def _is_rv_block_structure(
    etas: RandomVariables, partition: tuple[tuple[str, ...], ...], fixed_etas
):
    parts = set(partition)
    # Remove fixed etas from etas
    list_of_tuples = list(
        filter(
            None, list(map(lambda dist: tuple(_remove_sublist(list(dist.names), fixed_etas)), etas))
        )
    )
    return all(map(lambda dist: dist in parts, list_of_tuples))


def _create_param_dict(model: Model, dists: RandomVariables) -> dict[str, str]:
    param_subs = {
        parameter.symbol: parameter.init for parameter in model.parameters if parameter.fix
    }
    param_dict = {}
    # FIXME: Temporary workaround, should handle IIV on eps
    symbs_before_ode = [symb.name for symb in model.statements.before_odes.free_symbols]
    for eta in dists.names:
        if dists[eta].get_variance(eta).subs(param_subs) != 0:
            # Skip etas that are before ODE
            if eta not in symbs_before_ode:
                continue
            param_dict[eta] = get_rv_parameters(model, eta)[0]
    return param_dict


def create_description(
    model: Model, iov: bool = False, param_dict: Optional[dict[str, str]] = None
) -> str:
    if iov:
        dists = model.random_variables.iov
    else:
        dists = model.random_variables.iiv

    if not param_dict:
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

    description = '+'.join(blocks)
    return description


def create_eta_blocks(partition: tuple[tuple[str, ...], ...], model: Model, res: ModelfitResults):
    for part in partition:
        if len(part) == 1:
            model = split_joint_distribution(model, part)
        else:
            model = create_joint_distribution(
                model, list(part), individual_estimates=mfr(res).individual_estimates
            )
    return model


def _get_eta_from_parameter(model: Model, parameters: list[str]) -> set[str]:
    # returns list of eta names from parameter names
    # ETA names in parameters are allowed and will be returned as is
    iiv_set = set()
    iiv_names = model.random_variables.iiv.names

    for p in parameters:
        if p in iiv_names:
            iiv_set.add(p)
    for iiv_name in iiv_names:
        if _is_iiv_on_ruv(model, iiv_name):
            # Do not concider IIVs used on RUV
            continue
        param = get_rv_parameters(model, iiv_name)
        if set(param).issubset(parameters) and len(param) > 0:
            iiv_set.add(iiv_name)
    return iiv_set


def _is_iiv_on_ruv(model, name):
    error = model.statements.error
    for s in reversed(error):
        if Expr.symbol(name) in s.free_symbols:
            expr = error.full_expression(s.symbol)
            if not set(model.random_variables.epsilons.symbols).isdisjoint(expr.free_symbols):
                return True
    return False


def _get_fixed_etas(model: Model) -> list[str]:
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
