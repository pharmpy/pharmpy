from dataclasses import dataclass
from typing import Any, Literal, Optional

from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.tools.common import Results
from pharmpy.tools.run import rank_models_from_entries
from pharmpy.workflows import Context, ModelEntry, ModelfitResults, Task, Workflow, WorkflowBuilder

RANK_TYPES = frozenset(('ofv', 'lrt', 'aic', 'bic_mixed', 'bic_iiv'))


def create_workflow(
    model_ref: Model,  # Use ref value instead of a model object?
    results_ref: ModelfitResults,
    models_cand: list[Model],
    results_cand: list[ModelfitResults],
    strictness: str = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
    rank_type: Literal[tuple(RANK_TYPES)] = 'ofv',
    cutoff: Optional[float] = None,
):
    """Run Rank tool.

    Parameters
    ----------
    model_ref : Model
        Pharmpy model to use as reference (e.g. when calculating dOFV etc)
    results_ref : ModelfitResults
        Results for reference model
    models_cand : list[Model]
        Candidate models to rank
    results_cand : list[ModelfitResults]
        Candidate modelfit results
    strictness : str or None
        Strictness criteria
    rank_type : {'ofv', 'lrt', 'aic', 'bic_mixed', 'bic_iiv'}
        Which ranking type should be used. Default is OFV .
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)

    Returns
    -------
    RankToolResults
        Rank tool result object

    """
    wb = WorkflowBuilder(name='rank')
    start_task = Task(
        'start_rank',
        start,
        model_ref,
        results_ref,
        models_cand,
        results_cand,
        strictness,
        rank_type,
        cutoff,
    )
    wb.add_task(start_task)
    task_results = Task('results', _results)
    wb.add_task(task_results, predecessors=[start_task])

    return Workflow(wb)


def start(
    context: Context,
    model_ref: Model,
    results_ref: ModelfitResults,
    models_cand: list[Model],
    results_cand: list[ModelfitResults],
    strictness: str,
    rank_type: str,
    cutoff: Optional[float],
):
    context.log_info("Starting tool rank")

    wb = WorkflowBuilder()

    me_ref, mes_cand = prepare_model_entries(model_ref, results_ref, models_cand, results_cand)
    for me in [me_ref] + mes_cand:
        context.store_model_entry(me)

    context.log_info("Ranking models")
    rank_task = Task('rank_models', rank_models, me_ref, mes_cand, strictness, rank_type, cutoff)
    wb.add_task(rank_task)

    res = context.call_workflow(wb, 'rank_models')

    return res


def prepare_model_entries(
    model_ref: Model,
    results_ref: ModelfitResults,
    models_cand: list[Model],
    results_cand: list[ModelfitResults],
):
    me_ref = ModelEntry.create(model_ref, modelfit_results=results_ref)
    me_cands = [
        ModelEntry.create(m, modelfit_results=res) for m, res in zip(models_cand, results_cand)
    ]
    return me_ref, me_cands


def rank_models(
    me_ref: ModelEntry,
    mes_cand: list[ModelEntry],
    strictness: str,
    rank_type: str,
    cutoff: Optional[float],
):
    kwargs = get_rank_type_kwargs(rank_type)
    df_rank = rank_models_from_entries(
        me_ref, mes_cand, strictness=strictness, cutoff=cutoff, **kwargs
    )
    summary_tool = _modify_rank_table(me_ref, mes_cand, df_rank)

    best_model_name = summary_tool['rank'].idxmin()
    best_me = next(filter(lambda me: me.model.name == best_model_name, mes_cand), me_ref)

    res = RankToolResults(
        summary_tool=summary_tool, final_model=best_me.model, final_results=best_me.modelfit_results
    )
    return res


def _modify_rank_table(me_ref: ModelEntry, mes_cand: list[ModelEntry], df_rank: pd.DataFrame):
    rows = {}
    params_ref = len(me_ref.model.parameters.nonfixed)
    for model_entry in [me_ref] + mes_cand:
        model = model_entry.model
        description = model.description
        n_params = len(model.parameters.nonfixed)
        d_params = n_params - params_ref
        rows[model.name] = (description, n_params, d_params)

    colnames = ['description', 'n_params', 'd_params']
    index = pd.Index(rows.keys(), name='model')
    df_descr = pd.DataFrame(rows.values(), index=index, columns=colnames)

    df = pd.concat([df_descr, df_rank], axis=1)

    return df.reindex(df_rank.index)


def get_rank_type_kwargs(rank_type: str):
    if rank_type == 'bic_mixed':
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
    model_ref, results_ref, models_cand, results_cand, strictness, rank_type, cutoff
):
    if len(models_cand) != len(results_cand):
        raise ValueError(
            f'Length mismatch: `models_cand` ({len(models_cand)}) must be same length as '
            f'`results_cand ({len(results_cand)})`'
        )

    # FIXME: Remove when tool supports running models with parameter uncertainty step
    if strictness is not None and "rse" in strictness.lower():
        if any(
            model.execution_steps[-1].parameter_uncertainty_method is None
            for model in [model_ref] + models_cand
        ):
            raise ValueError(
                '`parameter_uncertainty_method` not set for one or more models, '
                'cannot calculate relative standard errors.'
            )


@dataclass(frozen=True)
class RankToolResults(Results):
    summary_tool: Optional[Any] = None
    final_model: Optional[Model] = None
    final_results: Optional[ModelfitResults] = None
