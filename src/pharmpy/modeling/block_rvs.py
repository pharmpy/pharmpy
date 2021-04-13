"""
:meta private:
"""

import numpy as np

from pharmpy import math
from pharmpy.modeling.help_functions import _get_etas
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariables


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
    rvs = model.random_variables
    if list_of_rvs is None:
        iiv_rvs = model.random_variables.iiv
        nonfix = RandomVariables()
        for rv in iiv_rvs:
            for name in iiv_rvs.parameter_names:
                if model.parameters[name].fix:
                    break
            else:
                nonfix.append(rv)
        list_of_rvs = nonfix.names
    else:
        for name in list_of_rvs:
            if name in rvs and rvs[name].level == 'IOV':
                raise ValueError(
                    f'{name} describes IOV: Joining IOV random variables is currently not supported'
                )
    if len(list_of_rvs) == 1:
        raise ValueError('At least two random variables are needed')

    sset = model.statements
    paramnames = []
    for rv in list_of_rvs:
        statements = sset.find_assignment(rv, is_symbol=False, last=False)
        parameter_names = '_'.join([s.symbol.name for s in statements])
        paramnames.append(parameter_names)

    cov_to_params = rvs.join(list_of_rvs, name_template='IIV_{}_IIV_{}', param_names=paramnames)

    pset = model.parameters
    for cov_name, param_names in cov_to_params.items():
        parent_params = (pset[param_names[0]], pset[param_names[1]])
        covariance_init = _choose_param_init(model, rvs, parent_params)
        param_new = Parameter(cov_name, covariance_init)
        pset.append(param_new)

    return model


def split_rv_block(model, list_of_rvs=None):
    """
    Splits a block structure given a list of etas to separate.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_of_rvs : str, list
        Name/names of etas to split from block structure. If None, all etas that are IIVs and
        non-fixed will become single. None is default.
    """
    rvs = model.random_variables
    list_of_rvs = _get_etas(model, list_of_rvs)

    parameters_before = rvs.parameter_names
    rvs.unjoin(list_of_rvs)
    parameters_after = rvs.parameter_names

    removed_parameters = set(parameters_before) - set(parameters_after)
    pset = model.parameters
    for param in removed_parameters:
        del pset[param]

    return model


def _choose_param_init(model, rvs, params):
    res = model.modelfit_results
    rvs_names = [rv.name for rv in rvs]

    etas = []
    for i in range(len(rvs)):
        elem = rvs.covariance_matrix.row(i).col(i)[0]
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
