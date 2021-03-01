import itertools


def exhaustive(base_model, transformation_funcs, run_func, rank_func):
    trans_indices = itertools.chain.from_iterable(
        itertools.combinations(list(range(len(transformation_funcs))), i + 1)
        for i in range(len(transformation_funcs))
    )
    torun = []
    for indices in trans_indices:
        model = base_model.copy()
        for i in indices:
            transformation_funcs[i](model)
        torun.append(model)
    run_func(torun)
    return rank_func(base_model, torun)
