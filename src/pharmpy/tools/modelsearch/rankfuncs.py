import numpy as np

# All functions used for comparing a set of candidate models and
# ranking them take the same base arguments
# base - the base model
# candidates - a list of candidate models
# returns a list which is a subset of the candidates with the best ranked at the beginning
# models can be removed from candidates if they don't pass some criteria


def ofv(base, candidates, cutoff=3.84):
    try:
        base_ofv = base.modelfit_results.ofv
    except AttributeError:
        base_ofv = np.nan
    return rank_models('ofv', base_ofv, candidates, cutoff)


def aic(base, candidates, cutoff=3.84):
    try:
        base_aic = base.modelfit_results.aic
    except AttributeError:
        base_aic = np.nan
    return rank_models('aic', base_aic, candidates, cutoff)


def bic(base, candidates, cutoff=3.84):
    try:
        base_bic = base.modelfit_results.bic
    except AttributeError:
        base_bic = np.nan
    return rank_models('bic', base_bic, candidates, cutoff)


def rank_models(rank_type, base_value, candidates, cutoff):
    filtered = [
        model
        for model in candidates
        if model.modelfit_results is not None
        and base_value - getattr(model.modelfit_results, rank_type) >= cutoff
    ]
    srtd = sorted(filtered, key=lambda model: getattr(model.modelfit_results, rank_type))
    return srtd
