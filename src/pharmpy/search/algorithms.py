import itertools

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
