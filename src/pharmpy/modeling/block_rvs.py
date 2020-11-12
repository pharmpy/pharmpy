import numpy as np
from sympy.stats.joint_rv_types import MultivariateNormalDistribution

from pharmpy import math
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariables, VariabilityLevel


class RVInputException(Exception):
    pass


def create_rv_block(model, list_of_rvs=None):
    """
    Creates a full or partial block structure of etas. The etas must be IIVs and cannot
    be fixed.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_of_rvs : list
        List of etas to create a block structure from. If None, all etas that are IIVs and
        non-fixed will be used (full block). None is default.
    """
    rvs_full = model.random_variables
    rvs = _get_rvs(model, list_of_rvs)

    if list_of_rvs is not None:
        for rv in rvs:
            if isinstance(rv.pspace.distribution, MultivariateNormalDistribution):
                rv_extracted = rvs_full.extract_from_block(rv)
                rvs.discard(rv)
                rvs.add(rv_extracted)

    pset = _merge_rvs(model, rvs)

    rvs_new = RandomVariables()

    for rv in rvs_full:
        if rv.name not in [rv.name for rv in rvs.etas]:
            rvs_new.add(rv)

    for rv in rvs:
        rvs_new.add(rv)

    model.random_variables = rvs_new
    model.parameters = pset

    model.modelfit_results = None

    return model


def _get_rvs(model, list_of_rvs):
    full_block = False

    if list_of_rvs is None:
        list_of_rvs = [rv for rv in model.random_variables.etas]
        full_block = True
    elif len(list_of_rvs) == 1:
        raise RVInputException('At least two random variables are needed')

    rvs = []
    for rv_str in list_of_rvs:
        try:
            rv = model.random_variables[rv_str]
        except KeyError:
            raise RVInputException(f'Random variable does not exist: {rv_str}')

        if _has_fixed_params(model, rv):
            if not full_block:
                raise RVInputException(f'Random variable cannot be set to fixed: {rv}')
            continue
        if rv.variability_level == VariabilityLevel.IOV:
            if not full_block:
                raise RVInputException(f'Random variable cannot be IOV: {rv}')
            continue

        rvs.append(rv)

    return RandomVariables(rvs)


def _has_fixed_params(model, rv):
    param_names = model.random_variables.get_eta_params(rv)

    for p in model.parameters:
        if p.name in param_names and p.fix:
            return True
    return False


def _merge_rvs(model, rvs):
    pset = model.parameters

    cov_to_params = rvs.merge_normal_distributions(create_cov_params=True)

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
    sd = np.array([np.sqrt(params[0].init), np.sqrt(params[1].init)])

    if res is not None:
        ie = res.individual_estimates
        eta_corr = ie[rvs_names].corr()
        cov = math.corr2cov(eta_corr.to_numpy(), sd)
        cov[cov == 0] = 0.0001
        cov = math.nearest_posdef(cov)
        return round(cov[1][0], 7)
    else:
        return round(0.1 * sd[0] * sd[1], 7)
