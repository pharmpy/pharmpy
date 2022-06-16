import warnings

import numpy as np
import pandas as pd

from pharmpy.modeling import (
    rank_models,
    summarize_errors,
    summarize_individuals,
    summarize_individuals_count_table,
    summarize_modelfit_results,
    update_inits,
)


def update_initial_estimates(model):
    try:
        update_inits(model, move_est_close_to_bounds=True)
    except (ValueError, np.linalg.LinAlgError):
        warnings.warn(f'{model.name}: Could not update initial estimates, using original estimates')
        pass
    return model


def create_results(
    res_class, input_model, base_model, res_models, rank_type, cutoff, bic_type='mixed'
):
    summary_tool = summarize_tool(res_models, base_model, rank_type, cutoff, bic_type)
    summary_models = summarize_modelfit_results([base_model] + res_models).reindex(
        summary_tool.index
    )
    summary_individuals, summary_individuals_count = summarize_tool_individuals(
        [base_model] + res_models, summary_tool['description'], summary_tool[f'd{rank_type}']
    )
    summary_errors = summarize_errors([base_model] + res_models)

    best_model_name = summary_tool['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = base_model

    if base_model.name != input_model.name:
        models = [base_model] + res_models
    else:
        models = res_models

    # FIXME: remove best_model, input_model, models when there is function to read db
    res = res_class(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals=summary_individuals,
        summary_individuals_count=summary_individuals_count,
        summary_errors=summary_errors,
        best_model=best_model,
        input_model=input_model,
        models=models,
    )

    return res


def summarize_tool(
    models,
    start_model,
    rank_type,
    cutoff,
    bic_type='mixed',
) -> pd.DataFrame:
    models_all = [start_model] + models

    df_rank, _ = rank_models(
        start_model, models, strictness=[], rank_type=rank_type, cutoff=cutoff, bic_type=bic_type
    )

    rows = {model.name: [model.description, model.parent_model] for model in models_all}
    colnames = ['description', 'parent_model']
    index = pd.Index(rows.keys(), name='model')
    df_descr = pd.DataFrame(rows.values(), index=index, columns=colnames)

    df = pd.concat([df_descr, df_rank], axis=1)
    df['parent_model'] = df.pop('parent_model')

    if rank_type == 'lrt':
        rank_type_name = 'ofv'
    else:
        rank_type_name = rank_type

    df_sorted = df.sort_values(by=[f'd{rank_type_name}'], ascending=False)

    assert df_sorted is not None
    return df_sorted


def summarize_tool_individuals(models, description_col, rank_type_col):
    summary_individuals = summarize_individuals(models)
    summary_individuals = summary_individuals.join(description_col, how='inner')
    col_to_move = summary_individuals.pop('description')
    summary_individuals.insert(0, 'description', col_to_move)

    suminds_count = summarize_individuals_count_table(df=summary_individuals)
    suminds_count.insert(0, description_col.name, description_col)
    suminds_count.insert(1, rank_type_col.name, rank_type_col)
    suminds_count['parent_model'] = suminds_count.pop('parent_model')
    summary_individuals_count = suminds_count.sort_values(by=[rank_type_col.name], ascending=False)
    return summary_individuals, summary_individuals_count
