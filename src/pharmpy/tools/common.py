import warnings

import numpy as np
import pandas as pd

import pharmpy.tools.rankfuncs as rankfuncs
from pharmpy.modeling import update_inits


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

    index = []
    rows = []
    for model in models_all:
        index.append(model.name)
        parent_model = model.parent_model
        diff = diff_dict[model.name]
        desc = model.description
        if model.name in ranking_by_name:
            ranks = ranking_by_name.index(model.name) + 1
        else:
            ranks = np.nan
        rows.append([parent_model, diff, desc, ranks])

    # FIXME: in ranks, if any row has NaN the rank converts to float
    colnames = ['parent_model', f'd{rankfunc_name}', 'description', 'rank']
    df = pd.DataFrame(rows, index=index, columns=colnames)

    return df
