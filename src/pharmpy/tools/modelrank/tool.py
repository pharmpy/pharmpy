import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.tools.common import ToolResults
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.workflows import (
    Context,
    ModelEntry,
    ModelfitResults,
    Task,
    Workflow,
    WorkflowBuilder,
)

from ..mfl.parse import parse as mfl_parse
from ..modelsearch.filter import mfl_filtering
from .ranking import get_rank_type, get_rank_values, rank_model_entries
from .strictness import evaluate_strictness, get_strictness_expr, get_strictness_predicates

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
    cutoff: Optional[float] = None,
    search_space: Optional[Union[str, ModelFeatures]] = None,
    E: Optional[Union[float, str, tuple[float], tuple[str]]] = None,
    _parent_dict: Optional[dict[str, str]] = None,
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
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    search_space : str, ModelFeatures or None
        Search space to test. Either as a string or a ModelFeatures object.
    E : float
        Expected number of predictors (used for mBIC). Must be set when using mBIC. Tuple
        if mBIC for IIV (both diagonals and off-diagonals)
    _parent_dict : dict
        EXPERIMENTAL FEATURE, WILL BE REMOVED. Dictionary of parent and child models.

    Returns
    -------
    RankToolResults
        Rank tool result object

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
        cutoff,
        search_space,
        E,
        _parent_dict,
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
    cutoff: Optional[float],
    search_space: Optional[str],
    E: Union[float, tuple[float]],
    _parent_dict: Optional[dict[str, str]],
):
    context.log_info("Starting tool modelrank")

    wb = WorkflowBuilder()

    me_ref, mes_cand = prepare_model_entries(models, results, ref_model, _parent_dict)
    for me in [me_ref] + mes_cand:
        context.store_model_entry(me)

    context.log_info("Ranking models")
    rank_task = Task(
        'rank_models',
        rank_models,
        me_ref,
        mes_cand,
        strictness,
        rank_type,
        cutoff,
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
    parent_dict: Optional[dict[str, str]],
):
    me_cands = []
    me_ref = None

    model_dict = {model.name: model for model in models}

    if not parent_dict:
        parent_dict = {model.name: ref_model.name for model in models}

    for model, results in zip(models, results):
        if model.name in parent_dict.keys() and model.name != parent_dict[model.name]:
            parent = model_dict[parent_dict[model.name]]
        else:
            parent = None
        me = ModelEntry.create(model, modelfit_results=results, parent=parent)
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
    cutoff: Optional[float],
    search_space,
    E,
):
    expr = get_strictness_expr(strictness)
    me_predicates = {me: get_strictness_predicates(me, expr) for me in [me_ref] + mes_cand}

    mes_to_rank = []
    for me, predicates in me_predicates.items():
        if evaluate_strictness(expr, predicates):
            mes_to_rank.append(me)

    me_rank_values = get_rank_values(
        me_ref, mes_cand, rank_type, cutoff, search_space, E, mes_to_rank
    )

    summary_strictness = create_table(me_predicates)
    summary_selection_criteria = create_table(me_rank_values)

    mes_sorted_by_rank = rank_model_entries(me_rank_values, get_rank_type(rank_type))
    summary_ranking = create_ranking_table(me_ref, mes_sorted_by_rank, get_rank_type(rank_type))

    best_me = list(mes_sorted_by_rank.keys())[0]

    res = ModelRankToolResults(
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
    col_names = [
        'model',
        'description',
        'n_params',
        'd_params',
        f'd{rank_type}',
        f'{rank_type}',
        'rank',
        'parent_model',
    ]
    df_data = {col: [] for col in col_names}
    params_ref = len(me_ref.model.parameters.nonfixed)
    for i, (me, predicates) in enumerate(me_rank_values.items(), 1):
        model = me.model
        n_params = len(me.model.parameters.nonfixed)
        parent = me.parent.name if me.parent else ''
        rank = i if not np.isnan(predicates['rank_val']) else np.nan
        me_dict = {
            'model': model.name,
            'description': model.description,
            'n_params': n_params,
            'd_params': n_params - params_ref,
            f'd{rank_type}': predicates[f'd{rank_type}'],
            f'{rank_type}': predicates[f'{rank_type}'],
            'rank': rank,
            'parent_model': parent,
        }
        for key, value in me_dict.items():
            df_data[key].append(value)

    df = pd.DataFrame(df_data).set_index(['model'])
    return df


def _results(context: Context, res: ToolResults):
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
    cutoff,
    search_space,
    E,
    _parent_dict,
):
    if len(models) != len(results):
        raise ValueError(
            f'Length mismatch: `models` ({len(models)}) must be same length as '
            f'`results ({len(results)})`'
        )

    # FIXME: Remove when tool supports running models with parameter uncertainty step
    if strictness is not None and "rse" in strictness.lower():
        if any(model.execution_steps[-1].parameter_uncertainty_method is None for model in models):
            raise ValueError(
                '`parameter_uncertainty_method` not set for one or more models, '
                'cannot calculate relative standard errors.'
            )

    if search_space:
        if isinstance(search_space, str):
            try:
                statements = mfl_parse(search_space)
            except:  # noqa E722
                pattern_cov = r'COV\?\(\[*([\w,]*)\]*\)'
                pattern_var = r'(IIV|IOV)\?\(\[*([\w,]*)\]*,\w+\)'
                if not re.match(pattern_cov, search_space) and not re.match(
                    pattern_var, search_space
                ):
                    raise ValueError(
                        f'Invalid `search_space`, could not be parsed: "{search_space}"'
                    )
        else:
            statements = search_space.filter("pk").mfl_statement_list()

        modelsearch_statements = mfl_filtering(statements, 'modelsearch')
        bad_statements = list(
            filter(lambda statement: statement not in modelsearch_statements, statements)
        )

        if bad_statements:
            raise ValueError(
                f'Invalid `search_space`: found unknown statement of type {type(bad_statements[0]).__name__}.'
            )

    if rank_type != 'mbic_mixed' and E is not None:
        raise ValueError(f'E can only be provided when `rank_type` is mbic: got `{rank_type}`')
    if rank_type == 'mbic_mixed':
        if search_space is None:
            raise ValueError('Argument `search_space` must be provided when using mbic')
        if E is None:
            raise ValueError('Value `E` must be provided when using mbic')
        if isinstance(E, float) and E <= 0.0:
            raise ValueError(f'Value `E` must be more than 0: got `{E}`')
        if isinstance(E, str) and not E.endswith('%'):
            raise ValueError(f'Value `E` must be denoted with `%`: got `{E}`')


@dataclass(frozen=True)
class ModelRankToolResults(ToolResults):
    summary_tool: Optional[Any] = None
    summary_strictness: Optional[Any] = None
    summary_selection_criteria: Optional[Any] = None
    final_model: Optional[Model] = None
    final_results: Optional[ModelfitResults] = None
