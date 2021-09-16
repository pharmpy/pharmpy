"""
:meta private:
"""

import numpy as np

from pharmpy import math
from pharmpy.modeling.help_functions import _get_etas
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariables


def create_joint_distribution(model, rvs=None):
    """
    Combines some or all etas into a joint distribution.

    The etas must be IIVs and cannot
    be fixed. Initial estimates for covariance between the etas is dependent on whether
    the model has results from a previous results. In that case, the correlation will
    be calculated from individual estimates, otherwise correlation will be set to 10%.

    Parameters
    ----------
    model : Model
        Pharmpy model
    rvs : list
        Sequence of etas or names of etas to combine. If None, all etas that are IIVs and
        non-fixed will be used (full block). None is default.

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, create_joint_distribution
    >>> model = load_example_model("pheno")
    >>> model.random_variables.etas
    ETA(1) ~ 𝒩 (0, OMEGA(1,1))
    ETA(2) ~ 𝒩 (0, OMEGA(2,2))
    >>> create_joint_distribution(model, ['ETA(1)', 'ETA(2)'])      # doctest: +ELLIPSIS
    <...>
    >>> model.random_variables.etas
    ⎡ETA(1)⎤     ⎧⎡0⎤  ⎡ OMEGA(1,1)   IIV_CL_IIV_V⎤⎫
    ⎢      ⎥ ~ 𝒩 ⎪⎢ ⎥, ⎢                          ⎥⎪
    ⎣ETA(2)⎦     ⎩⎣0⎦  ⎣IIV_CL_IIV_V   OMEGA(2,2) ⎦⎭

    See also
    --------
    split_joint_distribution : split etas into separate distributions
    """
    all_rvs = model.random_variables
    if rvs is None:
        iiv_rvs = model.random_variables.iiv
        nonfix = RandomVariables()
        for rv in iiv_rvs:
            for name in iiv_rvs.parameter_names:
                if model.parameters[name].fix:
                    break
            else:
                nonfix.append(rv)
        rvs = nonfix.names
    else:
        for name in rvs:
            if name in all_rvs and all_rvs[name].level == 'IOV':
                raise ValueError(
                    f'{name} describes IOV: Joining IOV random variables is currently not supported'
                )
    if len(rvs) == 1:
        raise ValueError('At least two random variables are needed')

    sset = model.statements
    paramnames = []
    for rv in rvs:
        statements = sset.find_assignment(rv, is_symbol=False, last=False)
        parameter_names = '_'.join([s.symbol.name for s in statements])
        paramnames.append(parameter_names)

    cov_to_params = all_rvs.join(rvs, name_template='IIV_{}_IIV_{}', param_names=paramnames)

    pset = model.parameters
    for cov_name, param_names in cov_to_params.items():
        parent_params = (pset[param_names[0]], pset[param_names[1]])
        covariance_init = _choose_param_init(model, all_rvs, parent_params)
        param_new = Parameter(cov_name, covariance_init)
        pset.append(param_new)

    return model


def split_joint_distribution(model, rvs=None):
    """
    Splits etas following a joint distribution into separate distributions.

    Parameters
    ----------
    model : Model
        Pharmpy model
    rvs : str, list
        Name/names of etas to separate. If None, all etas that are IIVs and
        non-fixed will become single. None is default.

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> create_joint_distribution(model, ['ETA(1)', 'ETA(2)'])      # doctest: +ELLIPSIS
    <...>
    >>> model.random_variables.etas
    ⎡ETA(1)⎤     ⎧⎡0⎤  ⎡ OMEGA(1,1)   IIV_CL_IIV_V⎤⎫
    ⎢      ⎥ ~ 𝒩 ⎪⎢ ⎥, ⎢                          ⎥⎪
    ⎣ETA(2)⎦     ⎩⎣0⎦  ⎣IIV_CL_IIV_V   OMEGA(2,2) ⎦⎭
    >>> split_joint_distribution(model, ['ETA(1)', 'ETA(2)'])       # doctest: +ELLIPSIS
    <...>
    >>> model.random_variables.etas
    ETA(1) ~ 𝒩 (0, OMEGA(1,1))
    ETA(2) ~ 𝒩 (0, OMEGA(2,2))

    See also
    --------
    create_joint_distribution : combine etas into a join distribution
    """
    all_rvs = model.random_variables
    rvs = _get_etas(model, rvs)

    parameters_before = all_rvs.parameter_names
    all_rvs.unjoin(rvs)
    parameters_after = all_rvs.parameter_names

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
