from typing import Union

from pharmpy.deps import numpy as np
from pharmpy.modeling import calculate_aic, calculate_bic

# All functions used for comparing a set of candidate models and
# ranking them take the same base arguments
# base - the base model
# candidates - a list of candidate models
# returns a list which is a subset of the candidates with the best ranked at the beginning
# models can be removed from candidates if they don't pass some criteria


def ofv(base, candidates, cutoff=3.84):
    return rank_models('ofv', base, candidates, cutoff)


def aic(base, candidates, cutoff=None):
    return rank_models('aic', base, candidates, cutoff)


def bic(base, candidates, cutoff=None, bic_type: Union[None, str] = 'mixed'):
    return rank_models('bic', base, candidates, cutoff, bic_type=bic_type)


def rank_models(rank_type, base, candidates, cutoff=None, bic_type: Union[None, str] = 'mixed'):
    diff_dict = _create_diff_dict(rank_type, base, candidates, bic_type)
    if cutoff is None:
        filtered = [model for model in candidates if model.modelfit_results is not None]
    else:
        filtered = [model for model in candidates if diff_dict[model.name] >= cutoff]

    def fn(model):
        if rank_type == 'aic':
            return calculate_aic(model, model.modelfit_results.ofv)
        elif rank_type == 'bic':
            return calculate_bic(model, model.modelfit_results.ofv, bic_type)
        else:
            return model.modelfit_results.ofv

    srtd = sorted(filtered, key=fn)  # FIXME: if same rankfunc value?
    return srtd, diff_dict


def _create_diff_dict(rank_type, base, candidates, bic_type):
    diff_dict = {}
    for model in candidates:
        # FIXME: way to handle if start model fails
        if base.modelfit_results is None or model.modelfit_results is None:
            diff = np.nan
        elif rank_type == 'aic':
            diff = calculate_aic(base, base.modelfit_results.ofv) - calculate_aic(
                model, model.modelfit_results.ofv
            )
        elif rank_type == 'bic':
            diff = calculate_bic(base, base.modelfit_results.ofv, bic_type) - calculate_bic(
                model, model.modelfit_results.ofv, bic_type
            )
        else:
            diff = base.modelfit_results.ofv - model.modelfit_results.ofv
        diff_dict[model.name] = diff
    return diff_dict
