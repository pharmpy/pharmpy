import warnings

import sympy

from pharmpy.parameter import Parameter
from pharmpy.random_variables import JointNormalSeparate, RandomVariables, VariabilityLevel
from pharmpy.symbols import symbol as S


class RVInputException(Exception):
    pass


def create_rv_block(model, list_of_rvs=None):
    rvs = _get_rvs(model, list_of_rvs)

    rv_map = _create_rv_map(model, rvs)

    cov = sympy.zeros(len(list_of_rvs))
    pset = model.parameters

    for row in range(len(list_of_rvs)):
        for col in range(row + 1):
            if row == col:
                cov[row, col] = rv_map[rvs[row]]
            else:
                covariance_init = _choose_param_init(model)
                param = Parameter(f'block_rv_{row}_{col}', covariance_init)
                pset.add(param)
                cov[row, col] = S(param.name)
                cov[col, row] = S(param.name)

    model.parameters = pset

    dist_new = _create_distribution(rvs, cov)
    rvs_new = RandomVariables()

    for rv in model.random_variables:
        try:
            rv_new = dist_new[rv.name]
            rvs_new.add(rv_new)
        except KeyError:
            rvs_new.add(rv)

    model.random_variables = rvs_new

    return model


def _get_rvs(model, list_of_rvs):
    rvs_all = model.random_variables

    if list_of_rvs is None:
        return RandomVariables(rvs_all)
    elif len(list_of_rvs) == 1:
        warnings.warn('Cannot combine only one random variable, using all.')
        return RandomVariables(rvs_all)
    else:
        rvs = []
        for rv_str in list_of_rvs:
            try:
                rv = rvs_all[rv_str]
            except KeyError:
                raise RVInputException(f'Random variable does not exist: {rv_str}')
            rvs.append(rv)
        return RandomVariables(rvs)


def _create_rv_map(model, rvs):
    rv_map = dict()
    for rv, dist in rvs.distributions():
        param = model.parameters[str(list(dist.free_symbols)[0])]
        rv = rv[0]
        if param.fix:
            raise RVInputException(f'Random variable cannot be set to fixed: {rv}')
        if rv.variability_level == VariabilityLevel.IOV:
            raise RVInputException(f'Random variable cannot be IOV: {rv}')
        rv_map[rv] = S(param.name)
    return rv_map


def _choose_param_init(model):
    return 0.1


def _create_distribution(rvs, cov):
    means = []
    names = []

    for rv in rvs:
        means.append(0)
        names.append(rv.name)

    rvs_new = JointNormalSeparate(names, means, cov)

    for rv in rvs_new:
        rv.variability_level = VariabilityLevel.IIV

    return RandomVariables(rvs_new)
