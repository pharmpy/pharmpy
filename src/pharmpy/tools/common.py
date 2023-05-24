from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Type, TypeVar

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results
from pharmpy.modeling import update_inits
from pharmpy.tools import rank_models, summarize_errors
from pharmpy.workflows import ToolDatabase

from .funcs import summarize_individuals, summarize_individuals_count_table

DataFrame = Any  # NOTE should be pd.DataFrame but we want lazy loading


def update_initial_estimates(model):
    if model.modelfit_results is None:
        return model
    if not model.modelfit_results.minimization_successful:
        if model.modelfit_results.termination_cause != 'rounding_errors':
            return model

    try:
        model = update_inits(
            model, model.modelfit_results.parameter_estimates, move_est_close_to_bounds=True
        )
    except (ValueError, np.linalg.LinAlgError):
        pass
    return model


@dataclass(frozen=True)
class ToolResults(Results):
    summary_tool: Optional[Any] = None
    summary_models: Optional[pd.DataFrame] = None
    summary_individuals: Optional[pd.DataFrame] = None
    summary_individuals_count: Optional[pd.DataFrame] = None
    summary_errors: Optional[pd.DataFrame] = None
    final_model_name: Optional[str] = None
    models: Sequence[Model] = ()
    tool_database: Optional[ToolDatabase] = None


T = TypeVar('T', bound=ToolResults)


def create_results(
    res_class: Type[T],
    input_model,
    base_model,
    res_models,
    rank_type,
    cutoff,
    bic_type='mixed',
    **rest,
) -> T:
    summary_tool = summarize_tool(res_models, base_model, rank_type, cutoff, bic_type)
    summary_individuals, summary_individuals_count = summarize_tool_individuals(
        [base_model] + res_models,
        summary_tool['description'],
        summary_tool[f'd{"ofv" if rank_type == "lrt" else rank_type}'],
    )
    summary_errors = summarize_errors(
        [base_model.modelfit_results] + [m.modelfit_results for m in res_models]
    )

    best_model_name = summary_tool['rank'].idxmin()
    best_model = next(filter(lambda model: model.name == best_model_name, res_models), base_model)

    if base_model.name != input_model.name:
        models = [base_model] + res_models
    else:
        models = res_models

    # FIXME: remove best_model, input_model, models when there is function to read db
    res = res_class(
        summary_tool=summary_tool,
        summary_individuals=summary_individuals,
        summary_individuals_count=summary_individuals_count,
        summary_errors=summary_errors,
        final_model_name=best_model.name,
        models=models,
        **rest,
    )

    return res


def summarize_tool(
    models,
    start_model,
    rank_type,
    cutoff,
    bic_type='mixed',
) -> DataFrame:
    models_all = [start_model] + models

    df_rank = rank_models(
        start_model,
        models,
        errors_allowed=['rounding_errors'],
        rank_type=rank_type,
        cutoff=cutoff,
        bic_type=bic_type,
    )

    model_dict = {model.name: model for model in models_all}
    rows = {}

    for model in models_all:
        description, parent_model = model.description, model.parent_model
        n_params = len(model.parameters.nonfixed)
        if model.name == start_model.name:
            d_params = 0
        else:
            d_params = n_params - len(model_dict[parent_model].parameters.nonfixed)
        rows[model.name] = (description, n_params, d_params, parent_model)

    colnames = ['description', 'n_params', 'd_params', 'parent_model']
    index = pd.Index(rows.keys(), name='model')
    df_descr = pd.DataFrame(rows.values(), index=index, columns=colnames)

    df = pd.concat([df_descr, df_rank], axis=1)
    df['parent_model'] = df.pop('parent_model')

    df_sorted = df.reindex(df_rank.index)

    assert df_sorted is not None
    return df_sorted


def summarize_tool_individuals(models, description_col, rank_type_col):
    summary_individuals = summarize_individuals(models)
    summary_individuals = summary_individuals.join(description_col, how='inner')
    col_to_move = summary_individuals.pop('description')
    summary_individuals.insert(0, 'description', col_to_move)

    suminds_count = summarize_individuals_count_table(df=summary_individuals)
    if suminds_count is None:
        summary_individuals_count = None
    else:
        suminds_count.insert(0, description_col.name, description_col)
        suminds_count.insert(1, rank_type_col.name, rank_type_col)
        suminds_count['parent_model'] = suminds_count.pop('parent_model')
        summary_individuals_count = suminds_count.sort_values(
            by=[rank_type_col.name], ascending=False
        )
    return summary_individuals, summary_individuals_count
