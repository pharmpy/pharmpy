import re
from functools import partial
from typing import Literal, Optional

from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps.scipy import stats
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import (
    Assignment,
    EstimationStep,
    ExecutionSteps,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
)
from pharmpy.modeling import (
    add_population_parameter,
    add_time_after_dose,
    create_symbol,
    get_mdv,
    has_proportional_error_model,
    set_combined_error_model,
    set_iiv_on_ruv,
    set_initial_estimates,
    set_power_on_ruv,
    set_proportional_error_model,
)
from pharmpy.modeling.blq import has_blq_transformation
from pharmpy.modeling.error import remove_error_model, set_time_varying_error_model
from pharmpy.tools.common import (
    create_plots,
    summarize_tool,
    table_final_eta_shrinkage,
    update_initial_estimates,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import (
    is_strictness_fulfilled,
    summarize_errors_from_entries,
    summarize_modelfit_results_from_entries,
)
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

from .results import RUVSearchResults, calculate_results

SKIP = frozenset(('IIV_on_RUV', 'power', 'combined', 'time_varying'))


def create_workflow(
    model: Model,
    results: ModelfitResults,
    groups: int = 4,
    p_value: float = 0.001,
    skip: Optional[list[Literal[tuple(SKIP)]]] = None,
    max_iter: int = 3,
    dv: Optional[int] = None,
    strictness: str = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
):
    """Run the ruvsearch tool. For more details, see :ref:`ruvsearch`.

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results of model
    groups : int
        The number of bins to use for the time varying models
    p_value : float
        The p-value to use for the likelihood ratio test
    skip : list of {'IIV_on_RUV', 'power', 'combined', 'time_varying'}
        A list of models to not attempt.
    max_iter :  int
        Number of iterations to run (1, 2, or 3). For models with BLQ only one iteration is supported.
    dv : int
        Which DV to assess the error model for.
    strictness : str
        Strictness criteria

    Returns
    -------
    RUVSearchResults
        Ruvsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_ruvsearch, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_ruvsearch(model=model, results=results)      # doctest: +SKIP

    """

    wb = WorkflowBuilder(name="ruvsearch")
    start_task = Task(
        'start_ruvsearch', start, model, results, groups, p_value, skip, max_iter, dv, strictness
    )
    wb.add_task(start_task)
    task_results = Task('results', _results)
    wb.add_task(task_results, predecessors=[start_task])
    return Workflow(wb)


def create_iteration_workflow(model_entry, groups, cutoff, skip, current_iteration, strictness, dv):
    wb = WorkflowBuilder()

    start_task = Task('start_iteration', _start_iteration, model_entry)
    wb.add_task(start_task)

    task_base_model = Task(
        'create_base_model', partial(_create_base_model, current_iteration=current_iteration, dv=dv)
    )
    wb.add_task(task_base_model, predecessors=start_task)

    tasks = []
    if 'IIV_on_RUV' not in skip:
        task_iiv = Task(
            'create_iiv_on_ruv_model',
            partial(_create_iiv_on_ruv_model, current_iteration=current_iteration),
        )
        tasks.append(task_iiv)
        wb.add_task(task_iiv, predecessors=task_base_model)

    if 'power' not in skip and 'combined' not in skip:
        task_power = Task(
            'create_power_model',
            partial(_create_power_model, current_iteration=current_iteration),
        )
        wb.add_task(task_power, predecessors=task_base_model)
        tasks.append(task_power)
        task_combined = Task(
            'create_combined_error_model',
            partial(_create_combined_model, current_iteration=current_iteration),
        )
        wb.add_task(task_combined, predecessors=task_base_model)
        tasks.append(task_combined)

    if 'time_varying' not in skip:
        for i in range(1, groups):
            tvar = partial(
                _create_time_varying_model,
                groups=groups,
                i=i,
                current_iteration=current_iteration,
            )
            task = Task(f"create_time_varying_model{i}", tvar)
            tasks.append(task)
            wb.add_task(task, predecessors=task_base_model)

    fit_wf = create_fit_workflow(n=1 + len(tasks))
    wb.insert_workflow(fit_wf, predecessors=[task_base_model] + tasks)
    post_pro = partial(
        post_process,
        cutoff=cutoff,
        current_iteration=current_iteration,
        strictness=strictness,
        dv=dv,
        groups=groups,
    )
    task_post_process = Task('post_process', post_pro)
    wb.add_task(task_post_process, predecessors=[start_task] + fit_wf.output_tasks)

    return Workflow(wb)


def proportional_error_workflow(model_entry):
    wb = WorkflowBuilder()

    prop_start = Task('Check_proportional', _start_iteration, model_entry)
    wb.add_task(prop_start)

    convert_to_prop_task = Task("convert_to_proportional", _change_proportional_model)
    wb.add_task(convert_to_prop_task, predecessors=prop_start)

    fit_wf = create_fit_workflow(n=1)
    wb.insert_workflow(fit_wf, predecessors=convert_to_prop_task)
    return Workflow(wb)


def _change_proportional_model(model_entry):
    model = model_entry.model
    model = model.replace(
        name='prop_error',
        description='Input model with proportional error model',
    )
    model = set_proportional_error_model(model)
    return ModelEntry.create(model, modelfit_results=None)


def start(context, input_model, input_res, groups, p_value, skip, max_iter, dv, strictness):
    context.log_info("Starting tool ruvsearch")
    cutoff = float(stats.chi2.isf(q=p_value, df=1))
    if skip is None:
        skip = []

    input_model = input_model.replace(name="input", description="")
    input_model_entry = ModelEntry.create(input_model, modelfit_results=input_res)
    context.store_input_model_entry(input_model_entry)
    context.log_info(f"Input model OFV: {input_res.ofv:.3f}")

    if not has_proportional_error_model(input_model_entry.model):
        context.log_info("Fitting input model with proportional error")
        proportional_workflow = proportional_error_workflow(input_model_entry)
        model_entry = context.call_workflow(proportional_workflow, 'Convert_error_model')
        prop_model_entry = model_entry
        selected_model_entries = [input_model_entry, model_entry]
        context.log_info(
            f"Input model with proportional error OFV: {model_entry.modelfit_results.ofv:.3f}"
        )
    else:
        model_entry = input_model_entry
        prop_model_entry = None
        selected_model_entries = [model_entry]

    cwres_models = []
    for current_iteration in range(1, max_iter + 1):
        context.log_info(f"Starting iteration {current_iteration}")
        wf = create_iteration_workflow(
            model_entry, groups, cutoff, skip, current_iteration, strictness=strictness, dv=dv
        )
        res, best_model_entry, selected_model_name = context.call_workflow(
            wf, f'results{current_iteration}'
        )
        cwres_models.append(res.cwres_models)

        if not selected_model_name.startswith('base'):
            selected_model_entries.append(best_model_entry)

        model_entry = best_model_entry
        context.log_info(f"Best model after iteration OFV: {model_entry.modelfit_results.ofv:.3f}")

        if selected_model_name.startswith('base'):
            break
        elif selected_model_name.startswith('time_varying'):
            skip.append('time_varying')
        else:
            skip.append(selected_model_name)

    # Check that there actually occured an improvement from the initial model.
    delta_ofv = input_model_entry.modelfit_results.ofv - model_entry.modelfit_results.ofv
    if delta_ofv < cutoff:
        model_entry = input_model_entry
        changing = "input"
    else:
        changing = None
    if prop_model_entry is not None:
        delta_ofv = prop_model_entry.modelfit_results.ofv - model_entry.modelfit_results.ofv
        if delta_ofv < cutoff:
            model_entry = prop_model_entry
            changing = "prop_error"
    if changing is not None and changing != model_entry.model.name:
        context.log_info(
            f"The {changing} model with OFV {model_entry.modelfit_results.ofv:.3f} was better than the selected model"
        )

    tables = create_result_tables(selected_model_entries, cutoff, strictness)
    plots = create_plots(model_entry.model, model_entry.modelfit_results)
    final_model = model_entry.model.replace(name="final")

    res = RUVSearchResults(
        cwres_models=pd.concat(cwres_models),
        final_model=final_model,
        final_results=model_entry.modelfit_results,
        summary_models=tables['summary_models'],
        summary_tool=tables['summary_tool'],
        summary_errors=tables['summary_errors'],
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=table_final_eta_shrinkage(
            model_entry.model, model_entry.modelfit_results
        ),
    )

    context.store_final_model_entry(final_model)

    return res


def create_result_tables(model_entries, cutoff, strictness):
    sum_models = summarize_modelfit_results_from_entries(model_entries)
    sum_models['step'] = list(range(len(sum_models)))
    summf = sum_models.reset_index().set_index(['step', 'model'])
    summary_tool = _create_summary_tool(model_entries, cutoff, strictness)
    summary_errors = summarize_errors_from_entries(model_entries)
    tables = {
        'summary_models': summf,
        'summary_tool': summary_tool,
        'summary_errors': summary_errors,
    }
    return tables


def _create_summary_tool(selected_model_entries, cutoff, strictness):
    selected_models = [model_entry.model for model_entry in selected_model_entries]
    model_names = [model.name for model in selected_models]
    iteration_map = {model.name: model_names.index(model.name) for model in selected_models}

    base_model_entry = selected_model_entries[0]
    ruvsearch_model_entries = selected_model_entries[1:]

    sum_tool = summarize_tool(
        ruvsearch_model_entries, base_model_entry, 'ofv', cutoff, strictness=strictness
    ).reset_index()
    sum_tool['step'] = sum_tool['model'].map(iteration_map)
    sum_tool_by_iter = sum_tool.set_index(['step', 'model']).sort_index()

    # FIXME: Workaround since rank_models will exclude ranking of base model since dofv will be 0
    sum_tool_by_iter.loc[(0, base_model_entry.model.name), 'ofv'] = (
        base_model_entry.modelfit_results.ofv
    )
    sum_tool_by_iter.loc[(0, base_model_entry.model.name), 'dofv'] = 0

    return sum_tool_by_iter.drop(columns=['rank'])


def _start_iteration(model_or_model_entry):
    return model_or_model_entry


def _results(context, res):
    context.log_info("Finishing tool ruvsearch")
    return res


def post_process(
    context, start_model_entry, *model_entries, cutoff, current_iteration, strictness, dv, groups
):
    res = calculate_results(model_entries)
    best_model_unfitted, selected_model_name = _create_best_model(
        start_model_entry, res, current_iteration, cutoff=cutoff, dv=dv, groups=groups
    )
    if best_model_unfitted is not None:
        fit_wf = create_fit_workflow(modelentries=[best_model_unfitted])
        best_model_entry = context.call_workflow(fit_wf, f'fit{current_iteration}')
        if best_model_entry.modelfit_results is not None:
            best_model_check = [
                best_model_entry.modelfit_results.ofv,
                best_model_entry.modelfit_results.residuals,
                best_model_entry.modelfit_results.predictions,
            ]
            if all(check is not None for check in best_model_check):
                delta_ofv = (
                    start_model_entry.modelfit_results.ofv - best_model_entry.modelfit_results.ofv
                )
                if (
                    is_strictness_fulfilled(
                        best_model_entry.model, best_model_entry.modelfit_results, strictness
                    )
                    and delta_ofv > cutoff
                ):
                    return (res, best_model_entry, selected_model_name)

    return (res, start_model_entry, f"base_{current_iteration}")


def _create_base_model(input_model_entry, current_iteration, dv):
    input_model = input_model_entry.model
    theta = Parameter('theta', 0.1)
    omega = Parameter('omega', 0.01, lower=0)
    sigma = Parameter('sigma', 1.0, lower=0)
    params = Parameters((theta, omega, sigma))

    eta_name = 'eta_base'
    eta = NormalDistribution.create(eta_name, 'iiv', 0, omega.symbol)
    sigma_name = 'epsilon'
    sigma = NormalDistribution.create(sigma_name, 'ruv', 0, sigma.symbol)
    rvs = RandomVariables.create([eta, sigma])

    y = Assignment.create(
        Expr.symbol('Y'), theta.symbol + Expr.symbol(eta_name) + Expr.symbol(sigma_name)
    )
    statements = Statements([y])

    name = f'base_{current_iteration}'

    est = EstimationStep.create('foce', interaction=True, maximum_evaluations=9999)

    base_model = Model.create(
        parameters=params,
        random_variables=rvs,
        statements=statements,
        name=name,
        description=name,
        execution_steps=ExecutionSteps.create([est]),
        dependent_variables={y.symbol: 1},
    )
    base_model = base_model.replace(dataset=_create_dataset(input_model_entry, dv))
    return ModelEntry.create(base_model, modelfit_results=None, parent=input_model)


def _create_iiv_on_ruv_model(base_model_entry, current_iteration):
    base_model = base_model_entry.model
    model = set_iiv_on_ruv(base_model)
    name = f'IIV_on_RUV_{current_iteration}'
    model = model.replace(name=name, description=name)
    return ModelEntry.create(model, modelfit_results=None, parent=base_model)


def _create_power_model(base_model_entry, current_iteration):
    base_model = base_model_entry.model
    model = set_power_on_ruv(base_model, ipred='IPRED', lower_limit=None, zero_protection=True)
    name = f'power_{current_iteration}'
    model = model.replace(name=name, description=name)
    return ModelEntry.create(model, modelfit_results=None, parent=base_model)


def _create_time_varying_model(base_model_entry, groups, i, current_iteration):
    base_model = base_model_entry.model
    quantile = i / groups
    cutoff = base_model.dataset['TAD'].quantile(q=quantile)
    model = set_time_varying_error_model(base_model, cutoff=cutoff, idv='TAD')
    name = f"time_varying{i}_{current_iteration}"
    model = model.replace(name=name, description=name)
    return ModelEntry.create(model, modelfit_results=None, parent=base_model)


def _create_combined_model(base_model_entry, current_iteration):
    base_model = base_model_entry.model
    model = remove_error_model(base_model)
    sset = model.statements
    ruv_prop = create_symbol(model, 'epsilon_p')
    ruv_add = create_symbol(model, 'epsilon_a')
    ipred = Expr.symbol('IPRED')
    s = sset[0]
    assert isinstance(s, Assignment)
    s = Assignment.create(s.symbol, s.expression + ruv_prop + ruv_add / ipred)

    prop_name = 'sigma_prop'
    model = add_population_parameter(model, prop_name, 1, lower=0)
    df = model.dataset
    assert df is not None
    df = df.copy()
    df['IPRED'] = df['IPRED'].replace(0, 2.225e-307)
    model = model.replace(dataset=df)
    ipred_min = df['IPRED'].min()
    if ipred_min == 0 or np.isnan(ipred_min):
        sigma_add_init = 0.001
    else:
        sigma_add_init = abs(ipred_min) / 2
    add_name = 'sigma_add'
    model = add_population_parameter(model, add_name, sigma_add_init, lower=0)

    eps_prop = NormalDistribution.create(ruv_prop.name, 'ruv', 0, Expr.symbol(prop_name))
    eps_add = NormalDistribution.create(ruv_add.name, 'ruv', 0, Expr.symbol(add_name))
    name = f'combined_{current_iteration}'
    model = model.replace(
        statements=s + sset[1:],
        random_variables=model.random_variables + [eps_prop, eps_add],
        name=name,
        description=name,
    )
    return ModelEntry.create(model, modelfit_results=None, parent=base_model)


def _create_dataset(input_model_entry: ModelEntry, dv):
    # Non-observations have already been filtered
    input_model, results = input_model_entry.model, input_model_entry.modelfit_results
    assert results is not None
    residuals = results.residuals
    assert residuals is not None
    input_dataset = input_model.dataset
    assert input_dataset is not None
    if dv is not None:
        try:
            dvid_name = input_model.datainfo.typeix['dvid'][0].name
        except IndexError:
            dvid_name = 'DVID'
        observation_label = input_model.datainfo.dv_column.name
        input_dataset_obs = input_dataset.query(f'{observation_label} != 0').reset_index(
            drop=True
        )  # filter non-observations
        indices_obs = input_dataset_obs.index[input_dataset_obs[f'{dvid_name}'] == dv].tolist()
        residuals = residuals.iloc[indices_obs]
    cwres = residuals['CWRES'].reset_index(drop=True)
    if has_blq_transformation(input_model):
        cwres = cwres.loc[cwres != 0]

    predictions = results.predictions
    assert predictions is not None
    if 'CIPREDI' in predictions:
        ipredcol = 'CIPREDI'
    elif 'IPRED' in predictions:
        ipredcol = 'IPRED'
    else:
        raise ValueError("Need CIPREDI or IPRED")
    if dv is not None:
        indices = input_model.dataset.index[input_model.dataset['DVID'] == dv].tolist()
        predictions = predictions.iloc[indices]
    ipred = predictions[ipredcol].reset_index(drop=True)

    mdv = get_mdv(input_model)
    if dv is not None:
        mdv = mdv.iloc[indices]
    mdv = mdv.reset_index(drop=True)

    label_id = input_model.datainfo.id_column.name
    if dv is not None:
        input_id = (
            input_dataset[label_id].iloc[indices].astype('int64').squeeze().reset_index(drop=True)
        )
    else:
        input_id = input_dataset[label_id].astype('int64').squeeze().reset_index(drop=True)

    input_model = add_time_after_dose(input_model)
    tad_label = input_model.datainfo.descriptorix['time after dose'][0].name
    if dv is not None:
        tad = input_model.dataset[tad_label].iloc[indices].squeeze().reset_index(drop=True)
    else:
        tad = input_model.dataset[tad_label].squeeze().reset_index(drop=True)

    df = pd.concat([mdv, input_id, tad, ipred], axis=1)
    df = df[df['MDV'] == 0].reset_index(drop=True)
    df = pd.concat([df, cwres], axis=1).rename(columns={'CWRES': 'DV', ipredcol: 'IPRED'})
    df = df.loc[df['DV'].notna()]
    return df


def _time_after_dose(model):
    if 'TAD' in model.dataset:
        pass
    else:
        model = add_time_after_dose(model)
    return model


def _create_best_model(model_entry, res, current_iteration, dv, groups, cutoff=3.84):
    if any(res.cwres_models['dofv'] > cutoff):
        model = update_initial_estimates(model_entry.model, model_entry.modelfit_results)
        idx = res.cwres_models['dofv'].idxmax()
        name = idx[0]

        if current_iteration == 1:
            base_description = ''
        else:
            base_description = model.description + '+'
        model = model.replace(
            name=f'best_ruvsearch_{current_iteration}', description=base_description + name
        )

        if name.startswith('power'):
            power = (
                res.cwres_models['parameters'].loc['power', 1, current_iteration].get('theta') + 1.0
            )
            if power < 0.01:
                # FIXME: Remove lower bound?
                power = 0.02
            model = set_power_on_ruv(model, dv=dv)
            model = set_initial_estimates(model, {'power1': power})
        elif name.startswith('IIV_on_RUV'):
            model = set_iiv_on_ruv(model, dv=dv)
            model = set_initial_estimates(
                model,
                {
                    'IIV_RUV1': res.cwres_models['parameters']
                    .loc['IIV_on_RUV', 1, current_iteration]
                    .get('omega')
                },
            )
        elif name.startswith('time_varying'):
            model = _time_after_dose(model)
            # Name is like time_varying10
            i = int(re.search(r"(\d+)$", name).group(1))
            quantile = i / groups
            df = _create_dataset(model_entry, dv=dv)
            tad = df['TAD']
            cutoff_tvar = tad.quantile(q=quantile)
            model = set_time_varying_error_model(model, cutoff=cutoff_tvar, idv='TAD', dv=dv)
            model = set_initial_estimates(
                model,
                {
                    'time_varying': res.cwres_models['parameters']
                    .loc[f"time_varying{i}", 1, current_iteration]
                    .get('theta')
                },
            )
        else:
            model = set_combined_error_model(model, dv=dv)
            model = set_initial_estimates(
                model,
                {
                    'sigma_prop': res.cwres_models['parameters']
                    .loc['combined', 1, current_iteration]
                    .get('sigma_prop'),
                    'sigma_add': res.cwres_models['parameters']
                    .loc['combined', 1, current_iteration]
                    .get('sigma_add'),
                },
            )

        best_model_entry = ModelEntry.create(model, modelfit_results=None, parent=model_entry.model)
        selected_model_name = name
    else:
        best_model_entry = None
        selected_model_name = None

    return best_model_entry, selected_model_name


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(model, results, groups, p_value, skip, max_iter, dv, strictness):
    if groups < 2:
        raise ValueError(f'Invalid `groups`: got `{groups}`, must be >= 2.')

    if not 0 < p_value <= 1:
        raise ValueError(f'Invalid `p_value`: got `{p_value}`, must be a float in range (0, 1].')

    if max_iter < 1 or max_iter > 3:
        raise ValueError(f'Invalid `max_iter`: got `{max_iter}`, must be int in range [1, 3].')

    if model is not None:
        if results is None:
            raise ValueError('Invalid `results`: modelfit results must be provided.')

        residuals = results.residuals
        if residuals is None or 'CWRES' not in residuals:
            raise ValueError(
                f'Invalid `results`: please check {model.name}.mod file to'
                f' make sure ID, TIME, CWRES are in $TABLE.'
            )

        predictions = results.predictions
        if predictions is None or ('CIPREDI' not in predictions and 'IPRED' not in predictions):
            raise ValueError(
                f'Invalid `results`: please check {model.name}.mod file to'
                f' make sure ID, TIME, CIPREDI (or IPRED) are in $TABLE.'
            )

        if has_blq_transformation(model) and max_iter > 1:
            raise ValueError(
                f'Invalid `max_iter`: got `{max_iter}`,only 1 iteration is supported '
                f'for models with BLQ transformation.'
            )

        if dv:
            if 'DVID' not in model.dataset.columns and 'dvid' not in model.datainfo.types:
                raise ValueError("No DVID column in dataset.")
            else:
                try:
                    dvid_name = model.datainfo.typeix['dvid'][0].name
                except IndexError:
                    dvid_name = 'DVID'
                if dv not in set(model.dataset[dvid_name]):
                    raise ValueError(f"No {dvid_name} = {dv} in dataset.")

        if Expr.symbol('TAD') in model.statements.lhs_symbols:
            raise ValueError("Invalid `model`: TAD must be a column")

    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                '`parameter_uncertainty_method` not set for model, cannot calculate relative standard errors.'
            )
