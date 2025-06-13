from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.tools.common import Results
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.tools.run import calculate_mbic_penalty, rank_models_from_entries
from pharmpy.workflows import Context, ModelEntry, ModelfitResults, Task, Workflow, WorkflowBuilder

from ..mfl.parse import parse as mfl_parse
from ..modelsearch.filter import mfl_filtering

RANK_TYPES = frozenset(('ofv', 'lrt', 'aic', 'bic_mixed', 'bic_iiv', 'mbic_mixed'))


def create_workflow(
    models: list[Model],
    results: list[ModelfitResults],
    ref_model: Model,
    strictness: str = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
    rank_type: Literal[tuple(RANK_TYPES)] = 'ofv',
    cutoff: Optional[float] = None,
    search_space: Optional[Union[str, ModelFeatures]] = None,
    E: Optional[Union[float, str, tuple[float], tuple[str]]] = None,
    _parent_dict: dict[str, str] = None,
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
    rank_type : {'ofv', 'lrt', 'aic', 'bic_mixed', 'bic_iiv'}
        Which ranking type should be used. Default is OFV .
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
    search_space,
    E,
    _parent_dict,
):
    context.log_info("Starting tool rank")

    wb = WorkflowBuilder()

    me_ref, mes_cand = prepare_model_entries(models, results, ref_model)
    for me in [me_ref] + mes_cand:
        context.store_model_entry(me)

    if rank_type == 'mbic':
        penalties = prepare_penalties(search_space, E, me_ref, mes_cand)
    else:
        penalties = None

    context.log_info("Ranking models")
    rank_task = Task(
        'rank_models',
        rank_models,
        me_ref,
        mes_cand,
        strictness,
        rank_type,
        cutoff,
        penalties,
        _parent_dict,
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


def prepare_penalties(search_space, E, me_ref, mes_cand):
    if isinstance(E, tuple):
        E_kwargs = {'E_p': E[0], 'E_q': E[1]}
    else:
        E_kwargs = {'E_p': E}
    penalties = [
        calculate_mbic_penalty(me.model, search_space, **E_kwargs) for me in [me_ref] + mes_cand
    ]
    return penalties


def rank_models(
    me_ref: ModelEntry,
    mes_cand: list[ModelEntry],
    strictness: str,
    rank_type: str,
    cutoff: Optional[float],
    penalties: Optional[list[float]],
    _parent_dict: dict[str, str],
):
    kwargs = get_rank_type_kwargs(rank_type)
    df_rank = rank_models_from_entries(
        me_ref, mes_cand, strictness=strictness, cutoff=cutoff, penalties=penalties, **kwargs
    )
    summary_tool = _modify_rank_table(me_ref, mes_cand, df_rank, _parent_dict)

    best_model_name = summary_tool['rank'].idxmin()
    best_me = next(filter(lambda me: me.model.name == best_model_name, mes_cand), me_ref)

    res = RankToolResults(
        summary_tool=summary_tool, final_model=best_me.model, final_results=best_me.modelfit_results
    )
    return res


def _modify_rank_table(
    me_ref: ModelEntry,
    mes_cand: list[ModelEntry],
    df_rank: pd.DataFrame,
    _parent_dict: dict[str, str],
):
    rows = {}
    params_ref = len(me_ref.model.parameters.nonfixed)
    for model_entry in [me_ref] + mes_cand:
        model = model_entry.model
        description = model.description
        if _parent_dict:
            parent = _parent_dict[model.name]
        else:
            parent = None
        n_params = len(model.parameters.nonfixed)
        d_params = n_params - params_ref
        rows[model.name] = (description, n_params, d_params, parent)

    colnames = ['description', 'n_params', 'd_params', 'parent_model']
    index = pd.Index(rows.keys(), name='model')
    df_descr = pd.DataFrame(rows.values(), index=index, columns=colnames)

    df = pd.concat([df_descr, df_rank], axis=1)
    df = df.reindex(df_rank.index)
    df['parent_model'] = df.pop('parent_model')

    return df


def get_rank_type_kwargs(rank_type: str):
    if rank_type in ('bic_mixed', 'mbic_mixed'):
        kwargs = {'rank_type': 'bic', 'bic_type': 'mixed'}
    elif rank_type == 'bic_iiv':
        kwargs = {'rank_type': 'bic', 'bic_type': 'iiv'}
    else:
        kwargs = {'rank_type': rank_type}
    return kwargs


def _results(context: Context, res: Results):
    context.log_info("Finishing tool rank")
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
                raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')
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
class RankToolResults(Results):
    summary_tool: Optional[Any] = None
    final_model: Optional[Model] = None
    final_results: Optional[ModelfitResults] = None
