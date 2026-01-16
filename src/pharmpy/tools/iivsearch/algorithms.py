import itertools
from collections import defaultdict
from collections.abc import Sequence
from typing import Optional, cast

import pharmpy.tools.modelfit as modelfit
from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.internals.set.partitions import partitions
from pharmpy.internals.set.subsets import non_empty_subsets
from pharmpy.mfl import IIV, Covariance, ModelFeatures
from pharmpy.model import Model, RandomVariables
from pharmpy.modeling import (
    create_joint_distribution,
    get_omegas,
    remove_iiv,
    set_description,
    split_joint_distribution,
)
from pharmpy.modeling.expressions import get_rv_parameters
from pharmpy.modeling.mfl import (
    expand_model_features,
    get_model_features,
    transform_into_search_space,
)
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.modelrank import ModelRankResults
from pharmpy.tools.run import run_subtool
from pharmpy.workflows import ModelEntry, ModelfitResults, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import mfr


def td_exhaustive(type, base_model_entry, mfl, index_offset, as_fullblock, param_mapping=None):
    assert mfl.is_expanded()

    wb = WorkflowBuilder(name=f'td_exhaustive_{type}')

    base_model = base_model_entry.model
    base_model = base_model.replace(description=create_description(base_model))

    if param_mapping:
        base_features = get_base_features_linearized(type, base_model, param_mapping)
    else:
        base_features = get_model_features(base_model, type=type)

    if type == 'iiv':
        combinations = get_iiv_combinations(mfl.iiv, base_features)
    else:
        combinations = get_covariance_combinations(mfl.covariance, base_features)

    for i, features in enumerate(combinations, 1):
        model_name = f'iivsearch_run{index_offset + i}'
        if param_mapping:
            task_candidate_entry = Task(
                f'create_{model_name}',
                create_candidate_linearized,
                model_name,
                features,
                type,
                param_mapping,
                base_model_entry,
            )
        else:
            task_candidate_entry = Task(
                f'create_{model_name}',
                create_candidate,
                model_name,
                features,
                type,
                as_fullblock,
                base_model_entry,
            )
        wb.add_task(task_candidate_entry)

    if len(wb.output_tasks) == 0:
        return None

    wf_fit = modelfit.create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit)
    wb.gather(wb.output_tasks)

    return Workflow(wb)


def bu_stepwise_no_of_etas_mfl(
    context, base_model_entry, mfl, index_offset, as_fullblock, rank_options
):
    mfl = expand_model_features(base_model_entry.model, mfl.iiv)
    iivs = mfl.iiv.filter(filter_on='optional')

    rank_results = []
    mes = []

    selected_model_entry = base_model_entry
    for i in range(len(iivs)):
        wb_step = WorkflowBuilder(name=f'step{i}')
        base_features = get_model_features(selected_model_entry.model, type='iiv')
        to_test = iivs.force_optional() - base_features.iiv
        for j, iiv in enumerate(to_test, 1):
            candidate_number = index_offset + len(mes) + j
            model_name = f'iivsearch_run{candidate_number}'
            task_candidate_entry = Task(
                f'create_{model_name}',
                create_candidate,
                model_name,
                base_features + iiv,
                'iiv',
                as_fullblock,
                selected_model_entry,
            )
            wb_step.add_task(task_candidate_entry)
            wf_fit = modelfit.create_fit_workflow(n=1)
            wb_step.insert_workflow(wf_fit, predecessors=[task_candidate_entry])

        wb_step.gather(wb_step.output_tasks)
        wf_step = Workflow(wb_step)
        mes_step = context.call_workflow(wf_step, unique_name=f'run_candidates_step{i}')

        rank_res = rank_models(
            context,
            rank_options,
            selected_model_entry,
            mes_step,
        )

        rank_results.append(rank_res)
        mes.extend(mes_step)

        if rank_res.final_model == selected_model_entry.model:
            break

        mes_all = (selected_model_entry,) + mes_step
        selected_model_entry = get_best_model_entry(mes_all, rank_res.final_model)

    return rank_results, tuple(mes)


def bu_stepwise_no_of_etas_linearized_mfl(
    context, linbase_model_entry, mfl, index_offset, rank_options, param_mapping
):
    mfl = expand_model_features(linbase_model_entry.parent, mfl.iiv)

    to_keep = mfl.iiv - mfl.iiv.filter(filter_on='optional')
    base_model_entry = run_base_model_entry_linearized(
        context, index_offset, to_keep, param_mapping, linbase_model_entry
    )
    base_features = get_base_features_linearized('iiv', base_model_entry.model, param_mapping)

    rank_results = []
    mes = [base_model_entry]

    selected_model_entry = base_model_entry
    to_test = mfl.iiv.filter(filter_on='optional').force_optional()

    i = 1
    while True:
        if not to_test:
            break

        wb_step = WorkflowBuilder(name=f'step{i}')
        for j, eta in enumerate(to_test, 1):
            to_keep = base_features + eta
            n = index_offset + len(mes) + j
            model_name = f'iivsearch_run{n}'
            task_candidate_entry = Task(
                f'create_{model_name}',
                create_candidate_linearized,
                model_name,
                to_keep,
                'iiv',
                param_mapping,
                linbase_model_entry,
            )
            wb_step.add_task(task_candidate_entry)
            wf_fit = modelfit.create_fit_workflow(n=1)
            wb_step.insert_workflow(wf_fit, predecessors=[task_candidate_entry])

        wb_step.gather(wb_step.output_tasks)
        wf_step = Workflow(wb_step)
        mes_step = context.call_workflow(wf_step, unique_name=f'run_candidates_step{i}')

        rank_res = rank_models(
            context,
            rank_options,
            selected_model_entry,
            mes_step,
        )

        rank_results.append(rank_res)
        mes.extend(mes_step)

        if rank_res.final_model == selected_model_entry.model:
            break

        mes_all = (selected_model_entry,) + mes_step
        selected_model_entry = get_best_model_entry(mes_all, rank_res.final_model)
        base_features = get_base_features_linearized(
            'iiv', selected_model_entry.model, param_mapping
        )
        to_test -= base_features
        i += 1

    return rank_results, tuple(mes)


def get_base_features_linearized(type, base_model, param_mapping):
    rvs = base_model.random_variables.iiv
    rvs = [
        dist
        for dist in rvs
        if not set(dist.parameter_names).intersection(base_model.parameters.fixed.names)
    ]
    iivs, covs = [], []
    for dist in rvs:
        dist_iivs = [
            IIV.create(parameter=param_mapping[eta], fp='exp', optional=False) for eta in dist.names
        ]
        iivs.extend(dist_iivs)
        if len(dist_iivs) > 1:
            pairs = [
                (p1.parameter, p2.parameter) for p1, p2 in itertools.combinations(dist_iivs, 2)
            ]
            dist_covs = [
                Covariance.create(type='iiv', parameters=pair, optional=False) for pair in pairs
            ]
            covs.extend(dist_covs)
    if type == 'iiv':
        base_features = ModelFeatures.create(iivs)
    else:
        base_features = ModelFeatures.create(covs)
    return base_features


def get_iiv_combinations(mfl, base_features):
    assert mfl.is_expanded()

    mfl_optional = mfl.filter(filter_on='optional')
    mfl_forced = mfl - mfl_optional
    params_forced = {feature.parameter for feature in mfl_forced}

    combinations = []
    for subset in ((),) + tuple(non_empty_subsets(mfl_optional)):
        mfl_subset = ModelFeatures.create(subset)
        mfl_all = mfl_forced + mfl_subset

        if params_forced.intersection(feature.parameter for feature in mfl_subset):
            continue
        if not mfl_all.force_optional().is_single_model():
            continue
        if mfl_all.force_optional() == base_features:
            continue

        combinations.append(mfl_all)

    combinations = tuple(sorted(combinations, key=len, reverse=True))

    return combinations


def get_covariance_combinations(mfl, base_features):
    assert mfl.is_expanded()

    mfl = mfl.filter(filter_on='optional')

    combinations = []
    for subset in non_empty_subsets(mfl):
        mfl_subset = ModelFeatures.create(subset)
        if mfl_subset.force_optional() == base_features:
            continue
        if not _is_valid_block_combination(mfl_subset):
            continue
        combinations.append(mfl_subset)
    if base_features:
        empty_subset = ModelFeatures.create([])
        return (empty_subset,) + tuple(combinations)
    else:
        return tuple(combinations)


def _is_valid_block_combination(features: Sequence[Covariance]):
    # FIXME: make more general
    params = sorted({p for f in features for p in f.parameters})

    idx = {p: i for i, p in enumerate(params)}
    adj = np.zeros((len(params), len(params)), dtype=bool)

    for feature in features:
        p1, p2 = feature.parameters
        i, j = idx[p1], idx[p2]
        adj[i, j] = True
        adj[j, i] = True
        adj[i, i] = True
        adj[j, j] = True

    if (~adj).any():
        return False
    else:
        return True


def create_candidate(name, mfl, type, as_fullblock, base_model_entry):
    base_model, base_res = base_model_entry.model, base_model_entry.modelfit_results
    candidate_model = update_initial_estimates(base_model, base_res)
    candidate_model = candidate_model.replace(name=name)
    ies = base_res.individual_estimates
    candidate_model = transform_into_search_space(
        candidate_model, mfl.force_optional(), type=type, individual_estimates=ies
    )
    if as_fullblock and len(mfl.iiv) > 1:
        candidate_model = create_joint_distribution(candidate_model, individual_estimates=ies)
    description = create_description_mfl(get_model_features(candidate_model), type)
    candidate_model = candidate_model.replace(description=description)

    return ModelEntry.create(model=candidate_model, parent=base_model)


def create_candidate_linearized(name, mfl, type, param_mapping, base_model_entry):
    base_model, base_res = base_model_entry.model, base_model_entry.modelfit_results
    candidate_model = base_model.replace(name=name)
    candidate_model = update_initial_estimates(candidate_model, base_res)
    param_to_eta = {k: v for v, k in param_mapping.items()}
    if type == 'iiv':
        to_keep = [param_to_eta[iiv.parameter] for iiv in mfl.iiv]
        to_remove = [eta for eta in base_model.random_variables.iiv.names if eta not in to_keep]
        for parameter in to_remove:
            candidate_model = remove_iiv(candidate_model, to_remove=parameter)
    else:
        covariances = Covariance.get_covariance_blocks(mfl.covariance)
        ies = base_res.individual_estimates
        for block in covariances:
            etas = [param_to_eta[param] for param in block]
            candidate_model = create_joint_distribution(
                candidate_model, etas, individual_estimates=ies
            )

    iivs = get_base_features_linearized('iiv', candidate_model, param_mapping)
    covs = get_base_features_linearized('covariance', candidate_model, param_mapping)
    description = create_description_mfl(iivs + covs, type)
    candidate_model = set_description(candidate_model, description)
    return ModelEntry.create(model=candidate_model, parent=base_model)


def run_base_model_entry_linearized(
    context, index_offset, to_keep, param_mapping, linbase_model_entry
):
    wb = WorkflowBuilder(name='base_model')

    candidate_number = index_offset + 1
    model_name = f'iivsearch_run{candidate_number}'
    task_base_entry = Task(
        f'create_{model_name}',
        create_candidate_linearized,
        model_name,
        to_keep,
        'iiv',
        param_mapping,
        linbase_model_entry,
    )

    wb.add_task(task_base_entry)
    wf_fit = modelfit.create_fit_workflow(n=1)
    wb.insert_workflow(wf_fit, predecessors=[task_base_entry])

    wb.gather(wb.output_tasks)

    base_model_entry = context.call_workflow(Workflow(wb), unique_name='run_base')[0]

    return base_model_entry


def create_description_mfl(mfl, type):
    assert type in ['iiv', 'covariance']

    blocks = Covariance.get_covariance_blocks(mfl.covariance)
    params_in_blocks = {p for b in blocks for p in b}
    blocks += tuple((iiv.parameter,) for iiv in mfl.iiv if iiv.parameter not in params_in_blocks)

    description = '+'.join(f"[{','.join(block)}]" for block in blocks)

    if type == 'iiv':
        fp_groups = defaultdict(list)
        for iiv in mfl.iiv:
            if iiv.fp != 'EXP':
                fp_groups[iiv.fp].append(iiv.parameter)

        if fp_groups:
            fps = [f"{fp}:{','.join(params)}" for fp, params in fp_groups.items()]
            fp_description = ';'.join(fps)
            description += f' ({fp_description})'

    return description


def rank_models(
    context,
    rank_options,
    base_model_entry,
    model_entries,
):
    models = [base_model_entry.model] + [me.model for me in model_entries]
    results = [base_model_entry.modelfit_results] + [me.modelfit_results for me in model_entries]

    rank_res = run_subtool(
        tool_name='modelrank',
        ctx=context,
        models=models,
        results=results,
        ref_model=base_model_entry.model,
        rank_type=rank_options.rank_type,
        alpha=rank_options.cutoff,
        strictness=rank_options.strictness,
        parameter_uncertainty_method=rank_options.parameter_uncertainty_method,
    )

    return rank_res


def get_best_model_entry(model_entries, final_model):
    best_model_entry = [me for me in model_entries if me.model == final_model]
    assert len(best_model_entry) == 1
    return best_model_entry[0]


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
    cutoff=None,
    E_p=None,
    E_q=None,
    parameter_uncertainty_method=None,
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
        cutoff,
        E_p,
        E_q,
        parameter_uncertainty_method,
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
    cutoff,
    E_p,
    E_q,
    parameter_uncertainty_method,
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

    # FIXME: remove once search space is properly handled
    from .tool import get_mbic_search_space

    if rank_type == 'mbic':
        search_space = get_mbic_search_space(base_model, keep, E_p, E_q)
    else:
        search_space = None
    rank_type = rank_type + '_iiv' if rank_type in ('bic', 'mbic') else rank_type

    E = (E_p, E_q) if E_p is not None or E_q is not None else None
    modelrank_opts = {
        'search_space': search_space,
        'rank_type': rank_type,
        'alpha': cutoff,
        'strictness': strictness,
        'E': E,
        'parameter_uncertainty_method': parameter_uncertainty_method,
    }

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

        selected_model_entry = select_model_entry(
            context, best_model_entry, new_candidate_modelentries, modelrank_opts
        )
        if selected_model_entry.model != best_model_entry.model:
            best_model_entry = selected_model_entry
            previous_removed = effect_dict[best_model_entry.model.name]

        if old_best_name == best_model_entry.model.name:
            return all_modelentries

    return all_modelentries


def select_model_entry(context, base_model_entry, model_entries, modelrank_opts):
    models_to_rank = [base_model_entry.model] + [me.model for me in model_entries]
    results_to_rank = [base_model_entry.modelfit_results] + [
        me.modelfit_results for me in model_entries
    ]

    rank_res = run_subtool(
        tool_name='modelrank',
        ctx=context,
        models=models_to_rank,
        results=results_to_rank,
        ref_model=base_model_entry.model,
        rank_type=modelrank_opts['rank_type'],
        alpha=modelrank_opts['alpha'],
        strictness=modelrank_opts['strictness'],
        search_space=modelrank_opts['search_space'],
        E=modelrank_opts['E'],
        parameter_uncertainty_method=modelrank_opts['parameter_uncertainty_method'],
    )
    rank_res = cast(ModelRankResults, rank_res)
    if rank_res.final_model and rank_res.final_model != base_model_entry.model:
        mes_best = [me for me in model_entries if me.model == rank_res.final_model]
        assert len(mes_best) == 1
        selected_me = mes_best[0]
    else:
        selected_me = base_model_entry

    return selected_me


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


def _get_eta_from_parameter(model: Model, parameters: Sequence[str]) -> set[str]:
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
