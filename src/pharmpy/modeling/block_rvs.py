import warnings

import sympy

from pharmpy.parameter import Parameter
from pharmpy.random_variables import JointNormalSeparate, RandomVariables, VariabilityLevel


class RVInputException(Exception):
    pass


def create_rv_block(model, list_of_rvs=None):
    rvs = _get_rvs(model, list_of_rvs)

    rv_map = _create_rv_map(model, rvs)

    means = [0, 0]
    cov = _create_cov_matrix(rvs, list_of_rvs, rv_map)

    rvs_new = JointNormalSeparate([rvs[0].name, rvs[1].name], means, cov)
    for rv in rvs_new:
        rv.variability_level = VariabilityLevel.IIV
    rvs_new = RandomVariables(rvs_new)

    covariance_init = _choose_param_init(model)

    pset = model.parameters
    for i in range(len(rvs)):
        param = Parameter(f'block_rv{i+1}', covariance_init)
        pset.add(param)

    print('\n')
    print(rvs_new)

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
        rv_map[rv] = param
    return rv_map


def _create_cov_matrix(rvs, list_of_rvs, rv_map):
    cov = sympy.zeros(len(list_of_rvs))

    for row in range(len(list_of_rvs)):
        for col in range(len(list_of_rvs)):
            if row == col:
                cov[row, col] = rv_map[rvs[row]].name

    return cov


def _choose_param_init(model):
    return 0.1
