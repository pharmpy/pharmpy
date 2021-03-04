import itertools

import numpy as np
import pandas as pd


def exhaustive(base_model, transformation_funcs, run_func, rank_func):
    trans_indices = itertools.chain.from_iterable(
        itertools.combinations(list(range(len(transformation_funcs))), i + 1)
        for i in range(len(transformation_funcs))
    )
    torun = []
    df = pd.DataFrame(
        index=pd.RangeIndex(stop=2 ** len(transformation_funcs) - 1),
        columns=['features', 'dofv', 'rank'],
    )
    for n, indices in enumerate(trans_indices):
        model = base_model.copy()
        model.name = f'candidate{n}'
        for i in indices:
            transformation_funcs[i](model)
        df.loc[n]['features'] = indices
        torun.append(model)
    run_func(torun)
    for i, model in enumerate(torun):
        df.loc[i]['dofv'] = base_model.modelfit_results.ofv - model.modelfit_results.ofv
    ranks = rank_func(base_model, torun)
    for i, ranked_model in enumerate(ranks):
        idx = torun.index(ranked_model)
        df.loc[idx]['rank'] = i + 1
    return df


def stepwise(base_model, transformation_funcs, run_func, rank_func):
    remaining = list(range(len(transformation_funcs)))
    start_model = base_model
    current_features = []
    features_col = []
    dofv_col = []
    for step in range(len(transformation_funcs)):
        torun = []
        for i in remaining:
            model = start_model.copy()
            model.name = f'step_{step}_{i}'
            func = transformation_funcs[i]
            func(model)
            torun.append(model)
            features_col.append(tuple(current_features + [i]))
        run_func(torun)
        for model in torun:
            dofv = start_model.modelfit_results.ofv - model.modelfit_results.ofv
            dofv_col.append(dofv)
        ranks = rank_func(start_model, torun)
        if not ranks:
            break
        start_model = ranks[0]
        idx = torun.index(start_model)
        current_features.append(remaining[idx])
        del remaining[idx]
    df = pd.DataFrame({'features': features_col, 'dofv': dofv_col, 'rank': np.nan})
    best_features = tuple(current_features)
    best_df_index = features_col.index(best_features)
    df.at[best_df_index, 'rank'] = 1
    return df
