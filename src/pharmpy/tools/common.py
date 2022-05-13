import warnings

import numpy as np
import pandas as pd

import pharmpy.tools.rankfuncs as rankfuncs
from pharmpy.modeling import summarize_individuals, summarize_individuals_count_table, update_inits


def update_initial_estimates(model):
    try:
        update_inits(model, move_est_close_to_bounds=True)
    except (ValueError, np.linalg.LinAlgError):
        warnings.warn(f'{model.name}: Could not update initial estimates, using original estimates')
        pass
    return model


def summarize_tool(
    models,
    start_model,
    rankfunc_name,
    cutoff,
    bic_type='mixed',
):
    models_all = [start_model] + models
    rankfunc = getattr(rankfuncs, rankfunc_name)

    kwargs = dict()
    if cutoff is not None:
        kwargs['cutoff'] = cutoff
    if rankfunc_name == 'bic':
        kwargs['bic_type'] = bic_type
    ranking, diff_dict = rankfunc(start_model, models_all, **kwargs)
    ranking_by_name = [model.name for model in ranking]  # Using list of models is very slow

    model_names = []
    rows = []
    for model in models_all:
        model_names.append(model.name)
        parent_model = model.parent_model
        diff = diff_dict[model.name]
        desc = model.description
        if model.name in ranking_by_name:
            ranks = ranking_by_name.index(model.name) + 1
        else:
            ranks = np.nan
        rows.append([desc, diff, ranks, parent_model])

    # FIXME: in ranks, if any row has NaN the rank converts to float
    colnames = ['description', f'd{rankfunc_name}', 'rank', 'parent_model']
    index = pd.Index(model_names, name='model')
    df = pd.DataFrame(rows, index=index, columns=colnames)
    df_sorted = df.sort_values(by=[f'd{rankfunc_name}'], ascending=False)

    return df_sorted


def summarize_tool_individuals(models, description_col, rankfunc_col):
    summary_individuals = summarize_individuals(models)
    summary_individuals = summary_individuals.join(description_col, how='inner')
    col_to_move = summary_individuals.pop('description')
    summary_individuals.insert(0, 'description', col_to_move)

    suminds_count = summarize_individuals_count_table(df=summary_individuals)
    suminds_count.insert(0, description_col.name, description_col)
    suminds_count.insert(1, rankfunc_col.name, rankfunc_col)
    col_to_move = suminds_count.pop('parent_model')
    suminds_count['parent_model'] = col_to_move
    summary_individuals_count = suminds_count.sort_values(by=[rankfunc_col.name], ascending=False)
    return summary_individuals, summary_individuals_count
