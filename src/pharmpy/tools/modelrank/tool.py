import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.workflows import (
    Context,
    ModelEntry,
    ModelfitResults,
    Results,
    Task,
    Workflow,
    WorkflowBuilder,
)

from ...modeling import add_parameter_uncertainty_step, set_description, set_name
from ..common import concat_summaries
from ..mfl.parse import parse as mfl_parse
from ..modelfit import create_fit_workflow
from ..modelsearch.filter import mfl_filtering
from ..run import summarize_modelfit_results_from_entries
from .ranking import get_rank_name, get_rank_values, rank_model_entries
from .strictness import get_strictness_expr, get_strictness_predicates, get_strictness_predicates_me

RANK_TYPES = frozenset(
    (
        'ofv',
        'lrt',
        'aic',
        'bic_mixed',
        'bic_iiv',
        'bic_random',
        'mbic_mixed',
        'mbic_iiv',
        'mbic_random',
    )
)


def create_workflow(
    models: list[Model],
    results: list[ModelfitResults],
    ref_model: Model,
    strictness: str = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
    rank_type: Literal[tuple(RANK_TYPES)] = 'ofv',
    alpha: Optional[float] = 0.05,
    search_space: Optional[Union[str, ModelFeatures]] = None,
    E: Optional[Union[float, str, tuple[float | str, float | str]]] = None,
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
):
    """Run ModelRank tool.

    Parameters
    ----------
    models : list[Model]
        Models to rank
    results : list[ModelfitResults]
        Modelfit results to rank on
    ref_model : Model
        Model to compare to
    strictness : str or None
        Strictness criteria
    rank_type : str
        Which ranking type should be used. Supported types are OFV, LRT, AIC, BIC (mixed, IIV,
        random), and mBIC (mixed, IIV, random). Default is OFV.
    alpha : float
        Cutoff p-value that is considered significant in likelihood ratio test. Default
        is None
    search_space : str, ModelFeatures or None
        Search space to test. Either as a string or a ModelFeatures object.
    E : float
        Expected number of predictors (used for mBIC). Must be set when using mBIC. Tuple
        if mBIC for IIV (both diagonals and off-diagonals)
    parameter_uncertainty_method : {'SANDWICH', 'SMAT', 'RMAT', 'EFIM'} or None
        Parameter uncertainty method. Will be used in ranking models if strictness includes
        parameter uncertainty

    Returns
    -------
    ModelRankResults
        ModelRank tool result object

    """
    wb = WorkflowBuilder(name='modelrank')
    start_task = Task(
        'start_rank',
        start,
        models,
        results,
        ref_model,
        strictness,
        rank_type,
        alpha,
        search_space,
        E,
        parameter_uncertainty_method,
    )
    wb.add_task(start_task)
    task_results = Task('results', _results)
    wb.add_task(task_results, predecessors=[start_task])

    return Workflow(wb)


def start(
    context: Context,
    models: list[Model],
    results: list[ModelfitResults],
    ref_model: Model,
    strictness: str,
    rank_type: str,
    alpha: Optional[float],
    search_space: Optional[str],
    E: Union[float, tuple[float]],
    parameter_uncertainty_method: Optional[str],
):
    context.log_info("Starting tool modelrank")

    wb = WorkflowBuilder()

    me_ref, mes_cand = prepare_model_entries(models, results, ref_model)
    for me in [me_ref] + mes_cand:
        context.store_model_entry(me)

    context.log_info("Ranking models")
    if parameter_uncertainty_method:
        rank_task = Task(
            'rank_models',
            rank_models_with_uncertainty,
            me_ref,
            mes_cand,
            strictness,
            rank_type,
            alpha,
            search_space,
            E,
            parameter_uncertainty_method,
        )
    else:
        rank_task = Task(
            'rank_models',
            rank_models,
            me_ref,
            mes_cand,
            strictness,
            rank_type,
            alpha,
            search_space,
            E,
        )
    wb.add_task(rank_task)

    res = context.call_workflow(wb, 'rank_models')

    return res


def prepare_model_entries(
    models: list[Model],
    results: list[ModelfitResults],
    ref_model: Model,
):
    me_cands = []
    me_ref = None

    for model, results in zip(models, results):
        me = ModelEntry.create(model, modelfit_results=results)
        if model == ref_model:
            assert me_ref is None
            me_ref = me
        else:
            me_cands.append(me)

    return me_ref, me_cands


def rank_models(
    me_ref: ModelEntry,
    mes_cand: list[ModelEntry],
    strictness: str,
    rank_type: str,
    alpha: Optional[float],
    search_space: Optional[str],
    E: Union[float, tuple[float]],
):
    expr = get_strictness_expr(strictness)
    me_predicates = get_strictness_predicates([me_ref] + mes_cand, expr)
    mes_to_rank = get_model_entries_to_rank(me_predicates, strict=True)

    me_rank_values = get_rank_values(
        me_ref, mes_cand, rank_type, alpha, search_space, E, mes_to_rank
    )
    me_rank_values_sorted = rank_model_entries(me_rank_values, rank_type)

    summary_strictness = create_table(me_predicates)
    summary_selection_criteria = create_table(me_rank_values)
    summary_ranking = create_ranking_table(me_ref, me_rank_values_sorted, rank_type)

    best_me = list(me_rank_values_sorted.keys())[0]

    res = ModelRankResults(
        summary_tool=summary_ranking,
        summary_strictness=summary_strictness,
        summary_selection_criteria=summary_selection_criteria,
        final_model=best_me.model,
        final_results=best_me.modelfit_results,
    )
    return res


def create_table(me_dict):
    col_names = list(list(me_dict.values())[0].keys())
    df_data = {col: [] for col in ['model'] + col_names}
    for me, predicates in me_dict.items():
        df_data['model'].append(me.model.name)
        for col in col_names:
            df_data[col].append(predicates[col])
    df = pd.DataFrame(df_data).set_index(['model'])
    return df


def create_ranking_table(me_ref, me_rank_values, rank_type):
    rank_name = get_rank_name(rank_type)
    col_names = [
        'model',
        'description',
        'n_params',
        'd_params',
        f'd{rank_name}',
        f'{rank_name}',
        'rank',
    ]
    df_data = {col: [] for col in col_names}
    params_ref = len(me_ref.model.parameters.nonfixed)
    for i, (me, predicates) in enumerate(me_rank_values.items(), 1):
        model = me.model
        n_params = len(me.model.parameters.nonfixed)
        rank = i if not np.isnan(predicates['rank_val']) or me.model is me_ref.model else pd.NA
        me_dict = {
            'model': model.name,
            'description': model.description,
            'n_params': n_params,
            'd_params': n_params - params_ref,
            f'd{rank_name}': predicates[f'd{rank_name}'],
            f'{rank_name}': predicates[f'{rank_name}'],
            'rank': rank,
        }
        for key, value in me_dict.items():
            df_data[key].append(value)

    df = pd.DataFrame(df_data).set_index(['model'])
    return df


def rank_models_with_uncertainty(
    context: Context,
    me_ref: ModelEntry,
    mes_cand: list[ModelEntry],
    strictness: str,
    rank_type: str,
    alpha: Optional[float],
    search_space: Optional[str],
    E: Union[float, tuple[float]],
    parameter_uncertainty_method: Optional[str],
):
    expr = get_strictness_expr(strictness)
    me_predicates = get_strictness_predicates([me_ref] + mes_cand, expr)
    mes_to_rank = get_model_entries_to_rank(me_predicates, strict=False)
    me_rank_values = get_rank_values(
        me_ref, mes_cand, rank_type, alpha, search_space, E, mes_to_rank
    )
    me_rank_values_sorted = rank_model_entries(me_rank_values, rank_type)

    no_cov_strictness = create_table(me_predicates)
    no_cov_selection_criteria = create_table(me_rank_values)
    no_cov_models = summarize_modelfit_results_from_entries([me_ref] + mes_cand)

    cov_strictness, cov_selection_criteria, cov_models = [], [], []
    mes_to_run = list(me_rank_values_sorted.keys())
    me_predicates_reeval = me_predicates.copy()
    i = 0
    best_me = None
    while mes_to_run:
        me = mes_to_run.pop(0)
        strictness_fulfilled = me_predicates[me]['strictness_fulfilled']
        if strictness_fulfilled:
            context.log_info(f'Model {me.model.name} already fulfilled strictness')
            best_me = me
            break
        elif strictness_fulfilled is False:
            break

        context.log_info(f'Running model {me.model.name} with parameter uncertainty')
        i += 1
        me_cov = run_candidate(context, me.model, i, parameter_uncertainty_method)

        predicates = get_strictness_predicates_me(me_cov, expr)
        me_predicates_reeval[me] = predicates

        strictness_fulfilled = predicates['strictness_fulfilled']
        assert strictness_fulfilled is not None

        rank_values = me_rank_values[me].copy()
        rank_values['rank_val'] = np.nan if not strictness_fulfilled else rank_values['rank_val']

        me_strictness = create_table({me_cov: predicates})
        me_selection_criteria = create_table({me_cov: rank_values})
        me_modelfit = summarize_modelfit_results_from_entries([me_cov])

        cov_strictness.append(me_strictness)
        cov_selection_criteria.append(me_selection_criteria)
        cov_models.append(me_modelfit)

        if strictness_fulfilled:
            context.log_info(
                f'Model {me.model.name} passed parameter uncertainty strictness criteria'
            )
            best_me = me
            break
        else:
            context.log_info(
                f'Model {me.model.name} did not pass parameter uncertainty strictness criteria, testing next model'
            )

    if best_me is None:
        context.log_warning('All models failed the strictness criteria')
        final_model, final_results = None, None
    else:
        final_model, final_results = best_me.model, best_me.modelfit_results

    keys = list(range(0, i + 1))

    summary_strictness = concat_summaries([no_cov_strictness] + cov_strictness, keys=keys)
    summary_selection_criteria = concat_summaries(
        [no_cov_selection_criteria] + cov_selection_criteria, keys=keys
    )
    summary_models = concat_summaries([no_cov_models] + cov_models, keys=keys)

    mes_to_rank = get_model_entries_to_rank(me_predicates_reeval, strict=False)
    me_rank_values = get_rank_values(
        me_ref, mes_cand, rank_type, alpha, search_space, E, mes_to_rank
    )
    me_rank_values_sorted = rank_model_entries(me_rank_values, rank_type)
    summary_ranking = create_ranking_table(me_ref, me_rank_values_sorted, rank_type)

    res = ModelRankResults(
        summary_tool=summary_ranking,
        summary_strictness=summary_strictness,
        summary_selection_criteria=summary_selection_criteria,
        summary_models=summary_models,
        final_model=final_model,
        final_results=final_results,
    )

    return res


def run_candidate(context, model_ref, i, parameter_uncertainty_method):
    name = f'modelrank_run{i}'
    candidate_task = Task(
        'cand', create_candidate_with_uncertainty, model_ref, name, parameter_uncertainty_method
    )

    wb = WorkflowBuilder()
    wb.add_task(candidate_task)
    wf_fit = create_fit_workflow(n=1)
    wb.insert_workflow(wf_fit)
    wf = Workflow(wb)

    cand_me = context.call_workflow(wf, f'fit_{model_ref.name}_cov')
    return cand_me


def create_candidate_with_uncertainty(base_model, name, parameter_uncertainty_method):
    model = set_name(base_model, name)
    model = set_description(model, f'{base_model.name} with uncertainty')
    model = add_parameter_uncertainty_step(
        model, parameter_uncertainty_method=parameter_uncertainty_method
    )
    me = ModelEntry.create(model, parent=base_model)
    return me


def get_model_entries_to_rank(me_predicates, strict=True):
    if strict:
        mes_to_rank = [
            me
            for me, predicates in me_predicates.items()
            if predicates['strictness_fulfilled'] is True
        ]
    else:
        mes_to_rank = [
            me
            for me, predicates in me_predicates.items()
            if predicates['strictness_fulfilled'] is not False
        ]

    return mes_to_rank


def _results(context: Context, res: Results):
    context.log_info("Finishing tool modelrank")
    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    models,
    results,
    ref_model,
    strictness,
    rank_type,
    alpha,
    search_space,
    E,
    parameter_uncertainty_method,
):
    if len(models) != len(results):
        raise ValueError(
            f'Length mismatch: `models` ({len(models)}) must be same length as '
            f'`results ({len(results)})`'
        )

    if ref_model not in models:
        raise ValueError(
            'Incorrect `ref_model`: reference model must be part of input' 'models and results'
        )

    if rank_type == 'lrt':
        for model in models:
            if model is ref_model:
                continue
            df = len(ref_model.parameters.nonfixed) - len(model.parameters.nonfixed)
            if df == 0:
                raise ValueError(
                    'Cannot perform LRT: at least one model candidate has 0'
                    'degrees of freedom compared to reference model'
                )

    if not parameter_uncertainty_method and "rse" in strictness.lower():
        if any(model.execution_steps[-1].parameter_uncertainty_method is None for model in models):
            raise ValueError(
                '`parameter_uncertainty_method` not set for one or more models, '
                'cannot calculate relative standard errors.'
            )

    if search_space:
        # FIXME: will be less messy once IIV/IOV is implemented in MFL
        if isinstance(search_space, str):
            try:
                statements = mfl_parse(search_space)
            except:  # noqa E722
                pattern_cov = r'COV\?*\(\[([\w,]*)\]*\)'
                pattern_var = r'(IIV|IOV)\?*\(\[(\w+,*)+\],\w+\)'
                if not re.match(pattern_cov, search_space) and not re.match(
                    pattern_var, search_space
                ):
                    raise ValueError(
                        f'Invalid `search_space`, could not be parsed: "{search_space}"'
                    )
                else:
                    statements = None
        else:
            statements = search_space.filter("pk").mfl_statement_list()

        if statements:
            modelsearch_statements = mfl_filtering(statements, 'modelsearch')
            bad_statements = list(
                filter(lambda statement: statement not in modelsearch_statements, statements)
            )

            if bad_statements:
                raise ValueError(
                    f'Invalid `search_space`: found unknown statement of type {type(bad_statements[0]).__name__}.'
                )

    if rank_type.startswith('mbic'):
        if search_space is None:
            raise ValueError('Argument `search_space` must be provided when using mbic')
        if E is None:
            raise ValueError('Value `E` must be provided when using mbic')
        if isinstance(E, float) and E <= 0.0:
            raise ValueError(f'Value `E` must be more than 0: got `{E}`')
        if isinstance(E, str) and not E.endswith('%'):
            raise ValueError(f'Value `E` must be denoted with `%`: got `{E}`')
    else:
        if E is not None:
            raise ValueError(f'E can only be provided when `rank_type` is mbic: got `{rank_type}`')


@dataclass(frozen=True)
class ModelRankResults(Results):
    summary_tool: Optional[Any] = None
    summary_strictness: Optional[Any] = None
    summary_selection_criteria: Optional[Any] = None
    summary_models: Optional[Any] = None
    final_model: Optional[Model] = None
    final_results: Optional[ModelfitResults] = None
