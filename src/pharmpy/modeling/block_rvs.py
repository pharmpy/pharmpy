"""
:meta private:
"""

import numpy as np
from sympy.stats.joint_rv_types import MultivariateNormalDistribution

from pharmpy import math
from pharmpy.modeling.help_functions import _format_input_list, _get_etas
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariables, VariabilityLevel


def create_rv_block(model, list_of_rvs=None):
    """
    Creates a full or partial block structure of etas. The etas must be IIVs and cannot
    be fixed. Initial estimates for covariance between the etas is dependent on whether
    the model has results from a previous results. In that case, the correlation will
    be calculated from individual estimates, otherwise correlation will be set to 10%.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_of_rvs : list
        List of etas to create a block structure from. If None, all etas that are IIVs and
        non-fixed will be used (full block). None is default.
    """
    if list_of_rvs and len(list_of_rvs) == 1:
        raise ValueError('At least two random variables are needed')

    rvs_full = model.random_variables
    rvs = _get_etas(model, list_of_rvs)

    if list_of_rvs is not None:
        for rv in rvs:
            if isinstance(rv.pspace.distribution, MultivariateNormalDistribution):
                rvs_same_dist = rvs_full.get_rvs_from_same_dist(rv)
                if not all(rv in rvs for rv in rvs_same_dist):
                    rvs_full.extract_from_block(rv)

    rvs_block = RandomVariables()
    for rv in rvs_full:
        if rv.name in [rv.name for rv in rvs]:
            rvs_block.add(rv)

    pset = _merge_rvs(model, rvs_block)

    rvs_new = RandomVariables()
    are_consecutive = rvs_full.are_consecutive(rvs_block)

    for rv in rvs_full:
        if rv.name not in [rv.name for rv in rvs_block.etas]:
            rvs_new.add(rv)
        elif are_consecutive:
            rvs_new.add(rvs_block[rv.name])

    if not are_consecutive:
        {rvs_new.add(rv) for rv in rvs_block}  # Add new block last

    model.random_variables = rvs_new
    model.parameters = pset

    return model


def split_rv_block(model, list_of_rvs=None):
    """
    Splits an block structure given a list of etas to separate.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_of_rvs : str, list
        Name/names of etas to split from block structure. If None, all etas that are IIVs and
        non-fixed will become single. None is default.
    """
    rvs_full = model.random_variables
    list_of_rvs = _format_input_list(list_of_rvs)
    rvs_block = _get_etas(model, list_of_rvs)
    pset = model.parameters

    cov_matrix_before = rvs_full.covariance_matrix()
    for rv in rvs_block:
        rvs_full.extract_from_block(rv)

    cov_matrix_after = rvs_full.covariance_matrix()

    diff = cov_matrix_before - cov_matrix_after
    param_names = [elem for elem in diff if elem != 0]

    for param in set(param_names):
        pset.discard(param)

    model.random_variables = rvs_full
    model.parameters = pset

    return model


def _merge_rvs(model, rvs):
    sset, pset = model.statements, model.parameters

    rv_to_param = dict()

    for rv in rvs:
        statements = sset.find_assignment(rv.name, is_symbol=False, last=False)
        parameter_names = '_'.join([s.symbol.name for s in statements])
        rv_to_param[rv.name] = parameter_names

    cov_to_params = rvs.merge_normal_distributions(create_cov_params=True, rv_to_param=rv_to_param)

    for rv in rvs:
        rv.variability_level = VariabilityLevel.IIV

    for cov_name, param_names in cov_to_params.items():
        parent_params = (pset[param_names[0]], pset[param_names[1]])
        covariance_init = _choose_param_init(model, rvs, parent_params)
        param_new = Parameter(cov_name, covariance_init)
        pset.add(param_new)

    return pset


def _choose_param_init(model, rvs, params):
    res = model.modelfit_results
    rvs_names = [rv.name for rv in rvs]

    etas = []
    for i in range(len(rvs)):
        elem = rvs.covariance_matrix().row(i).col(i)[0]
        if str(elem) in [p.name for p in params]:
            etas.append(rvs_names[i])

    sd = np.array([np.sqrt(params[0].init), np.sqrt(params[1].init)])
    init_default = round(0.1 * sd[0] * sd[1], 7)

    if res is not None:
        try:
            ie = res.individual_estimates
            if not all(eta in ie.columns for eta in etas):
                return init_default
        except KeyError:
            return init_default
        eta_corr = ie[etas].corr()
        cov = math.corr2cov(eta_corr.to_numpy(), sd)
        cov[cov == 0] = 0.0001
        cov = math.nearest_posdef(cov)
        return round(cov[1][0], 7)
    else:
        return init_default
