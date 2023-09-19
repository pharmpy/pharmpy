from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence, Type, TypeVar

from pharmpy.model import Model, Results
from pharmpy.modeling import update_inits
from pharmpy.tools import rank_models, summarize_errors
from pharmpy.workflows import ToolDatabase

from .funcs import summarize_individuals, summarize_individuals_count_table

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    from pharmpy.deps import numpy as np
    from pharmpy.deps import pandas as pd

DataFrame = Any  # NOTE: Should be pd.DataFrame but we want lazy loading

RANK_TYPES = frozenset(('ofv', 'lrt', 'aic', 'bic', 'mbic'))


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
    final_model: Optional[Model] = None
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
    if rank_type == 'lrt':
        delta_name = 'dofv'
    elif rank_type == 'mbic':
        delta_name = 'dbic'
    else:
        delta_name = f'd{rank_type}'
    summary_individuals, summary_individuals_count = summarize_tool_individuals(
        [base_model] + res_models,
        summary_tool['description'],
        summary_tool[delta_name],
    )
    summary_errors = summarize_errors(
        [base_model.modelfit_results] + [m.modelfit_results for m in res_models]
    )

    best_model_name = summary_tool['rank'].idxmin()
    best_model = next(filter(lambda model: model.name == best_model_name, res_models), base_model)

    if base_model.name != input_model.name:
        models = [base_model] + res_models
    else:
        # Check if any resulting models exist
        if res_models:
            models = res_models
        else:
            models = None

    # FIXME: Remove best_model, input_model, models when there is function to read db
    res = res_class(
        summary_tool=summary_tool,
        summary_individuals=summary_individuals,
        summary_individuals_count=summary_individuals_count,
        summary_errors=summary_errors,
        final_model=best_model,
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
    if rank_type == 'mbic':
        rank_type = 'bic'
        multiple_testing = True
        n_expected_models = len(models)
    else:
        multiple_testing = False
        n_expected_models = None

    models_all = [start_model] + models

    df_rank = rank_models(
        start_model,
        models,
        errors_allowed=['rounding_errors'],
        rank_type=rank_type,
        cutoff=cutoff,
        bic_type=bic_type,
        multiple_testing=multiple_testing,
        mult_test_p=n_expected_models,
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
