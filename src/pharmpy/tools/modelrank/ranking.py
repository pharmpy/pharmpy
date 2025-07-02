from typing import Literal, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.modeling import calculate_aic, calculate_bic
from pharmpy.modeling.lrt import cutoff as lrt_cutoff
from pharmpy.modeling.lrt import degrees_of_freedom as lrt_df
from pharmpy.tools.run import calculate_mbic_penalty
from pharmpy.workflows import ModelEntry


def get_rank_values(
    me_ref: ModelEntry,
    mes_cand: list[ModelEntry],
    rank_type: str,
    p_value: Optional[float],
    search_space: Optional[str],
    E: Optional[float],
    mes_to_rank: list[ModelEntry],
) -> dict[ModelEntry, dict[str, float]]:

    if rank_type == 'lrt':
        me_dict = {me.model: me for me in [me_ref] + mes_cand}
        rank_val = 'ofv'
        ref = perform_lrt(me_ref, me_ref, p_value)
        cands = {me: perform_lrt(me, me_dict[me.parent], p_value) for me in mes_cand}
        rank_values = {me_ref: ref, **cands}
    else:
        if rank_type == 'ofv':
            rank_func = get_ofv
            rank_kwargs = dict()
            rank_val = 'ofv'
        elif rank_type == 'aic':
            rank_func = get_aic
            rank_kwargs = dict()
            rank_val = 'aic'
        else:
            rank_func = get_bic
            rank_kwargs = {'rank_type': rank_type, 'search_space': search_space, 'E': E}
            rank_val = 'bic'
        ref = rank_func(me_ref, None, **rank_kwargs)
        cands = {me: rank_func(me, ref[rank_val], **rank_kwargs) for me in mes_cand}
        rank_values = {me_ref: ref, **cands}

    for me, values in rank_values.items():
        if me in mes_to_rank:
            if rank_type == 'lrt' and not values['significant']:
                values['rank_val'] = np.nan
            else:
                values['rank_val'] = values[rank_val]
        else:
            values['rank_val'] = np.nan

    return rank_values


def perform_lrt(me, me_parent, p_value) -> dict[str, Union[float, int, bool]]:
    rank_dict = dict()
    rank_dict['df'] = lrt_df(me_parent, me)
    if isinstance(p_value, tuple):
        alpha = p_value[0] if rank_dict['df'] >= 0 else p_value[1]
    else:
        alpha = p_value
    rank_dict['alpha'] = alpha
    rank_dict['p_value'] = lrt_cutoff(me_parent.model, me.model, alpha)
    likelihood = me.modelfit_results.ofv
    rank_dict['dofv'] = me_parent.modelfit_results.ofv - likelihood
    rank_dict['ofv'] = likelihood
    if rank_dict['dofv'] >= rank_dict['p_value']:
        rank_dict['significant'] = True
    else:
        rank_dict['significant'] = False

    return rank_dict


def rank_model_entries(me_rank_values, sort_by):
    rank_values_no_nan = {
        me: vals for me, vals in me_rank_values.items() if not np.isnan(vals['rank_val'])
    }
    ranking = dict(sorted(rank_values_no_nan.items(), key=lambda x: x[1]['rank_val']))
    rank_values_nan = {
        me: vals for me, vals in me_rank_values.items() if me not in rank_values_no_nan.keys()
    }
    ranking_nan = dict(sorted(rank_values_nan.items(), key=lambda x: x[1][sort_by]))
    ranking = {**ranking, **ranking_nan}
    return ranking


def get_rank_type(rank_type):
    if rank_type == 'lrt':
        return 'ofv'
    elif 'bic' in rank_type:
        return 'bic'
    else:
        return rank_type


def get_ofv(me, ref_value) -> dict[str, float]:
    rank_dict = dict()
    likelihood = me.modelfit_results.ofv
    if ref_value:
        rank_dict['dofv'] = ref_value - likelihood
    else:
        rank_dict['dofv'] = 0
    rank_dict['ofv'] = likelihood
    return rank_dict


def get_aic(me, ref_value) -> dict[str, Union[float, int]]:
    rank_dict = dict()
    likelihood = me.modelfit_results.ofv
    rank_dict['ofv'] = likelihood
    aic = calculate_aic(me.model, likelihood)
    rank_dict['aic_penalty'] = aic - likelihood
    if ref_value:
        rank_dict['daic'] = ref_value - aic
    else:
        rank_dict['daic'] = 0
    rank_dict['aic'] = aic
    return rank_dict


def get_bic(me, ref_value, rank_type, search_space, E=None) -> dict[str, Union[float, int]]:
    rank_dict = dict()
    likelihood = me.modelfit_results.ofv
    rank_dict['ofv'] = likelihood
    bic = calculate_bic(me.model, likelihood, type=_get_bic_type(rank_type))
    rank_dict['bic_penalty'] = bic - likelihood
    if 'mbic' in rank_type:
        E_kwargs = get_mbic_E_values(E)
        mbic_penalty = calculate_mbic_penalty(me.model, search_space, **E_kwargs)
        bic += mbic_penalty
        rank_dict['mbic_penalty'] = mbic_penalty
    if ref_value:
        rank_dict['dbic'] = ref_value - bic
    else:
        rank_dict['dbic'] = 0
    rank_dict['bic'] = bic
    return rank_dict


def get_mbic_E_values(E):
    if isinstance(E, tuple):
        E_kwargs = {'E_p': E[0], 'E_q': E[1]}
    else:
        E_kwargs = {'E_p': E}
    return E_kwargs


def _get_bic_type(rank_type: str) -> Literal['mixed', 'iiv', 'fixed', 'random']:
    bic_type = rank_type.split('_')[-1]
    assert bic_type in ('mixed', 'iiv', 'fixed', 'random')
    return bic_type
