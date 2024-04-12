from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pharmpy.tools.modelfit as modelfit
from pharmpy.internals.set.partitions import partitions
from pharmpy.internals.set.subsets import non_empty_subsets
from pharmpy.model import Model, RandomVariables
from pharmpy.modeling import (
    calculate_bic,
    create_joint_distribution,
    find_clearance_parameters,
    get_omegas,
    remove_iiv,
    split_joint_distribution,
)
from pharmpy.modeling.expressions import get_rv_parameters
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.run import is_strictness_fulfilled
from pharmpy.workflows import (
    ModelEntry,
    ModelfitResults,
    Task,
    Workflow,
    WorkflowBuilder,
    call_workflow,
)
from pharmpy.workflows.results import mfr


def td_exhaustive_no_of_etas(base_model, index_offset=0, keep=None):
    wb = WorkflowBuilder(name='td_exhaustive_no_of_etas')

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
            'candidate_entry', create_no_of_etas_candidate_entry, model_name, to_remove, False, None
        )
        wb.add_task(task_candidate_entry)

    wf_fit = modelfit.create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit)
    return Workflow(wb)


def bu_stepwise_no_of_etas(
    base_model,
    strictness,
    index_offset=0,
    input_model_entry=None,
    keep=None,
):
    wb = WorkflowBuilder(name='bu_stepwise_no_of_etas')
    stepwise_task = Task(
        "stepwise_BU_task",
        stepwise_BU_algorithm,
        base_model,
        index_offset,
        strictness,
        input_model_entry,
        keep,
    )
    wb.add_task(stepwise_task)
    return wb


def stepwise_BU_algorithm(
    context, base_model, index_offset, strictness, input_model_entry, keep, base_model_entry
):
    base_model = base_model.replace(description=create_description(base_model))

    iivs = base_model.random_variables.iiv
    iiv_names = iivs.names  # All ETAs in the base model

    # Remove fixed etas
    fixed_etas = _get_fixed_etas(base_model)
    iiv_names = _remove_sublist(iiv_names, fixed_etas)

    # Remove alle ETAs except for clearance (if possible)
    cl = find_clearance_parameters(base_model)[0]  # FIXME : Handle multiple clearance?
    cl_eta = list(_get_eta_from_parameter(base_model, [str(cl)]))[0]
    if cl_eta in iiv_names:
        base_parameter = cl_eta
    else:
        base_parameter = sorted(iiv_names)[0]  # No clearance --> fallback to alphabetical order

    if keep:
        parameters_to_ignore = _get_eta_from_parameter(base_model, keep)
    else:
        parameters_to_ignore = {base_parameter}

    # Create and run first model with a single ETA on base_parameter
    bu_base_model_wb = WorkflowBuilder(name='create_and_fit_BU_base_model')
    to_be_removed = [i for i in iiv_names if i not in parameters_to_ignore]
    model_name = f'iivsearch_run{1 + index_offset}'
    index_offset += 1
    bu_base_entry = Task(
        'candidate_entry',
        create_no_of_etas_candidate_entry,
        model_name,
        to_be_removed,
        True,
        input_model_entry,
        base_model_entry,
    )
    bu_base_model_wb.add_task(bu_base_entry)
    wf_fit = modelfit.create_fit_workflow(n=len(bu_base_model_wb.output_tasks))
    bu_base_model_wb.insert_workflow(wf_fit)
    best_model_entry = call_workflow(Workflow(bu_base_model_wb), 'fit_BU_base_model', context)

    # Filter IIV names to contain all combination with the base parameter in it
    iiv_names_to_add = list(non_empty_subsets(iiv_names))

    iiv_names_to_add = [i for i in iiv_names_to_add if all(p in i for p in parameters_to_ignore)]

    # Invert the list to REMOVE ETAs from the base model instead of adding to the
    # single ETA model
    iiv_names_to_remove = [tuple(i for i in iiv_names if i not in x) for x in iiv_names_to_add]

    # Remove largest step removing all ETAs but base_parameter
    max_step = max(len(element) for element in iiv_names_to_remove)
    iiv_names_to_remove = [i for i in iiv_names_to_remove if len(i) != max_step]

    # Dictionary of all possible candidates of each step
    step_dict = defaultdict(list)
    for step in iiv_names_to_remove:
        step_dict[max_step - len(step) + 1].append(step)
    # Assert to be sorted in correct order
    step_dict = dict(sorted(step_dict.items()))

    number_of_predicted = len(iiv_names)
    number_of_expected = number_of_predicted / 2
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
        new_candidate_modelentries = call_workflow(
            Workflow(temp_wb), f'td_exhaustive_no_of_etas-fit-{step_number}', context
        )
        all_modelentries.extend(new_candidate_modelentries)
        old_best_name = best_model_entry.model.name
        for me in new_candidate_modelentries:
            if is_strictness_fulfilled(me.modelfit_results, me.model, strictness):
                bic_me = calculate_bic(
                    me.model,
                    me.modelfit_results.ofv,
                    type='iiv',
                    multiple_testing=True,
                    mult_test_p=number_of_predicted,
                    mult_test_e=number_of_expected,
                )
                bic_best = calculate_bic(
                    best_model_entry.model,
                    best_model_entry.modelfit_results.ofv,
                    type='iiv',
                    multiple_testing=True,
                    mult_test_p=number_of_predicted,
                    mult_test_e=number_of_expected,
                )
                if bic_best > bic_me:
                    best_model_entry = me
                    previous_removed = effect_dict[me.model.name]
        if old_best_name == best_model_entry.model.name:
            return all_modelentries

    return all_modelentries


def td_exhaustive_block_structure(base_model, index_offset=0):
    wb = WorkflowBuilder(name='td_exhaustive_block_structure')

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


def create_no_of_etas_candidate_entry(
    name, to_remove, base_parent, best_model_entry, base_model_entry
):
    if best_model_entry is None:
        best_model_entry = base_model_entry
    candidate_model = remove_iiv(base_model_entry.model, to_remove)
    candidate_model = update_initial_estimates(candidate_model, best_model_entry.modelfit_results)
    candidate_model = candidate_model.replace(
        name=name, description=create_description(candidate_model)
    )

    if base_parent:
        parent = base_model_entry.model
    else:
        parent = best_model_entry.model

    return ModelEntry.create(model=candidate_model, modelfit_results=None, parent=parent)


def create_block_structure_candidate_entry(name, block_structure, model_entry):
    candidate_model = update_initial_estimates(model_entry.model, model_entry.modelfit_results)
    candidate_model = create_eta_blocks(
        block_structure, candidate_model, model_entry.modelfit_results
    )
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
        if dists[eta].get_variance(eta).subs(param_subs) != 0:
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


def create_eta_blocks(partition: Tuple[Tuple[str, ...], ...], model: Model, res: ModelfitResults):
    for part in partition:
        if len(part) == 1:
            model = split_joint_distribution(model, part)
        else:
            model = create_joint_distribution(
                model, list(part), individual_estimates=mfr(res).individual_estimates
            )
    return model


def _get_eta_from_parameter(model: Model, parameters: List[str]) -> Set[str]:
    # returns list of eta names from parameter names
    iiv_set = set()
    iiv_names = model.random_variables.iiv.names
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
