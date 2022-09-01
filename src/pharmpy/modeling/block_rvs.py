"""
:meta private:
"""
import warnings

import numpy as np
import sympy

from pharmpy import math
from pharmpy.modeling.help_functions import _get_etas
from pharmpy.parameters import Parameter, Parameters
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
    ETA(1) ~ N(0, OMEGA(1,1))
    ETA(2) ~ N(0, OMEGA(2,2))
    >>> create_joint_distribution(model, ['ETA(1)', 'ETA(2)'])      # doctest: +ELLIPSIS
    <...>
    >>> model.random_variables.etas
    ⎡ETA(1)⎤    ⎧⎡0⎤  ⎡ OMEGA(1,1)   IIV_CL_IIV_V⎤⎫
    ⎢      ⎥ ~ N⎪⎢ ⎥, ⎢                          ⎥⎪
    ⎣ETA(2)⎦    ⎩⎣0⎦  ⎣IIV_CL_IIV_V   OMEGA(2,2) ⎦⎭

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
        parameter_names = '_'.join(
            [s.symbol.name for s in sset if sympy.Symbol(rv) in s.rhs_symbols]
        )
        paramnames.append(parameter_names)

    cov_to_params = all_rvs.join(rvs, name_template='IIV_{}_IIV_{}', param_names=paramnames)
    pset_new = model.parameters
    for cov_name, param_names in cov_to_params.items():
        parent1, parent2 = model.parameters[param_names[0]], model.parameters[param_names[1]]
        covariance_init = _choose_param_init(model, all_rvs, parent1, parent2)
        param_new = Parameter(cov_name, covariance_init)
        pset_new += param_new
    model.parameters = Parameters(pset_new)

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
    ⎡ETA(1)⎤    ⎧⎡0⎤  ⎡ OMEGA(1,1)   IIV_CL_IIV_V⎤⎫
    ⎢      ⎥ ~ N⎪⎢ ⎥, ⎢                          ⎥⎪
    ⎣ETA(2)⎦    ⎩⎣0⎦  ⎣IIV_CL_IIV_V   OMEGA(2,2) ⎦⎭
    >>> split_joint_distribution(model, ['ETA(1)', 'ETA(2)'])       # doctest: +ELLIPSIS
    <...>
    >>> model.random_variables.etas
    ETA(1) ~ N(0, OMEGA(1,1))
    ETA(2) ~ N(0, OMEGA(2,2))

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
    model.parameters = Parameters([p for p in model.parameters if p.name not in removed_parameters])
    return model


def _choose_param_init(model, rvs, parent1, parent2):
    res = model.modelfit_results

    etas = []
    for rv in rvs:
        if rvs.get_variance(rv).name in (parent1.name, parent2.name):
            etas.append(rv.name)

    sd = np.array([np.sqrt(parent1.init), np.sqrt(parent2.init)])
    init_default = round(0.1 * sd[0] * sd[1], 7)

    last_estimation_step = [est for est in model.estimation_steps if not est.evaluation][-1]
    if last_estimation_step.method == 'FO':
        return init_default
    elif res is not None:
        try:
            ie = res.individual_estimates
            if not all(eta in ie.columns for eta in etas):
                return init_default
        except KeyError:
            return init_default
        # NOTE Use pd.corr() and not pd.cov(). SD is chosen from the final estimates, if cov is used
        # it will be calculated from the EBEs.
        eta_corr = ie[etas].corr()
        if eta_corr.isnull().values.any():
            warnings.warn(
                f'Correlation of individual estimates between {parent1.name} and '
                f'{parent2.name} is NaN, returning default initial estimate'
            )
            return init_default
        cov = math.corr2cov(eta_corr.to_numpy(), sd)
        cov[cov == 0] = 0.0001
        cov = math.nearest_postive_semidefinite(cov)
        init_cov = cov[1][0]
        return round(init_cov, 7)
    else:
        return init_default
