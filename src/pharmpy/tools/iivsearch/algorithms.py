import itertools
from collections import defaultdict
from collections.abc import Sequence

import pharmpy.tools.modelfit as modelfit
from pharmpy.deps import numpy as np
from pharmpy.internals.set.subsets import non_empty_subsets
from pharmpy.mfl import IIV, Covariance, ModelFeatures
from pharmpy.model import Model
from pharmpy.modeling import (
    create_joint_distribution,
    remove_iiv,
    set_description,
)
from pharmpy.modeling.mfl import (
    expand_model_features,
    get_model_features,
    transform_into_search_space,
)
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.run import run_subtool
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder


def td_exhaustive(
    type, base_model_entry, mfl, index_offset, as_fullblock=False, param_mapping=None
):
    assert mfl.is_expanded()

    wb = WorkflowBuilder(name=f'td_exhaustive_{type}')

    base_model = base_model_entry.model
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


def bu_stepwise_no_of_etas(
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
            selected_model_entry.model,
            [selected_model_entry] + list(mes_step),
        )

        rank_results.append(rank_res)
        mes.extend(mes_step)

        if rank_res.final_model == selected_model_entry.model:
            break

        mes_all = (selected_model_entry,) + mes_step
        selected_model_entry = get_best_model_entry(mes_all, rank_res.final_model)

    return rank_results, tuple(mes)


def bu_stepwise_no_of_etas_linearized(
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
            selected_model_entry.model,
            [selected_model_entry] + list(mes_step),
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

    mfl_optional = mfl.filter(filter_on='optional')
    mfl_forced = mfl - mfl_optional

    combinations = []
    for subset in ((),) + tuple(non_empty_subsets(mfl_optional)):
        mfl_subset = ModelFeatures.create(subset)
        mfl_all = mfl_forced + mfl_subset.force_optional()

        if mfl_all == base_features:
            continue
        if not _is_valid_block_combination(mfl_subset):
            continue

        combinations.append(mfl_all)

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
    description = create_description(get_model_features(candidate_model), type)
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
    description = create_description(iivs + covs, type)
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


def create_description(model_or_mfl, type):
    assert type in ['iiv', 'covariance']
    if isinstance(model_or_mfl, Model):
        mfl = get_model_features(model_or_mfl, type=type)
        if type == 'covariance':
            mfl += get_model_features(model_or_mfl, type='iiv')
    else:
        mfl = model_or_mfl

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
    ref_model,
    model_entries,
):
    models = [me.model for me in model_entries]
    results = [me.modelfit_results for me in model_entries]

    rank_res = run_subtool(
        tool_name='modelrank',
        ctx=context,
        models=models,
        results=results,
        ref_model=ref_model,
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
