import numpy as np

from pharmpy.modeling import calculate_aic, calculate_bic

# All functions used for comparing a set of candidate models and
# ranking them take the same base arguments
# base - the base model
# candidates - a list of candidate models
# returns a list which is a subset of the candidates with the best ranked at the beginning
# models can be removed from candidates if they don't pass some criteria


def ofv(base, candidates, cutoff=3.84, rank_by_not_worse=False):
    return rank_models('ofv', base, candidates, cutoff, rank_by_not_worse=rank_by_not_worse)


def aic(base, candidates, cutoff=None):
    return rank_models('aic', base, candidates, cutoff)


def bic(base, candidates, cutoff=None, bic_type='mixed'):
    return rank_models('bic', base, candidates, cutoff, bic_type=bic_type)


def create_diff_dict(rank_type, base, candidates, bic_type='mixed'):
    delta_dict = dict()
    for model in candidates:
        if base.modelfit_results is not None and model.modelfit_results is not None:
            if rank_type == 'aic':
                delta = calculate_aic(base) - calculate_aic(model)
            elif rank_type == 'bic':
                if not bic_type:
                    bic_type = 'mixed'
                delta = calculate_bic(base, bic_type) - calculate_bic(model, bic_type)
            else:
                delta = base.modelfit_results.ofv - model.modelfit_results.ofv
        else:
            delta = np.nan
        delta_dict[model.name] = delta
    return delta_dict


def rank_models(
    rank_type, base, candidates, cutoff=None, rank_by_not_worse=False, bic_type='mixed'
):
    delta_dict = create_diff_dict(rank_type, base, candidates, bic_type)
    if cutoff is not None:
        if rank_type == 'ofv' and rank_by_not_worse:
            cutoff = -cutoff
        filtered = [model for model in candidates if delta_dict[model.name] >= cutoff]
    else:
        filtered = [model for model in candidates if model.modelfit_results is not None]

    def fn(model):
        if rank_type == 'aic':
            return calculate_aic(model)
        elif rank_type == 'bic':
            bic = calculate_bic(model, bic_type)
            return bic
        else:
            return getattr(model.modelfit_results, rank_type)

    srtd = sorted(filtered, key=fn)
    return srtd
