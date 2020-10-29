import re

import sympy
from sympy.stats import Normal
from sympy.stats.joint_rv_types import MultivariateNormalDistribution

from pharmpy.parameter import Parameter
from pharmpy.random_variables import JointNormalSeparate, RandomVariables, VariabilityLevel
from pharmpy.symbols import symbol as S


class RVInputException(Exception):
    pass


def create_rv_block(model, list_of_rvs=None):
    rvs = _get_rvs(model, list_of_rvs)

    pset, cov = _create_cov_matrix(model, rvs)

    model.parameters = pset

    dist_new = _create_distribution(rvs, cov)

    rvs_new = RandomVariables()

    for rv in model.random_variables:
        if rv not in rvs:
            rvs_new.add(rv)

    for rv in dist_new:
        rvs_new.add(rv)

    model.random_variables = rvs_new

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
        if isinstance(rv.pspace.distribution, MultivariateNormalDistribution):
            joined_rvs = model.random_variables.get_joined_rvs(rv)
            _extract_rv_from_block(rv, joined_rvs)
        rvs.append(rv)

    return RandomVariables(rvs)


def _extract_rv_from_block(rv, rest_of_block):
    dist = rv.pspace.distribution
    m = dist.args[1]
    eta_number = re.match(r'ETA\(([0-9])\)', rv.name).group(1)
    rvs_new = []

    to_remove = None

    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            if row == col:
                elem = m.row(row).col(col)[0]
                omega_number = re.match(r'OMEGA\(([0-9]),[0-9]\)', elem.name).group(1)
                if omega_number == eta_number:
                    rv_new = Normal(rv.name, 0, sympy.sqrt(elem))
                    rv_new.variability_level = VariabilityLevel.IIV
                    rvs_new.append(rv_new)
                    to_remove = row
                    break

    cov = m.row_del(to_remove).col_del(to_remove)

    if len(cov) == 1:
        rv_new = Normal(rv.name, 0, sympy.sqrt(cov[0]))
        rv_new.variability_level = VariabilityLevel.IIV
        rvs_new.append(rv_new)
    else:
        means = sympy.zeros(m.shape[0] - 1)
        dist_new = JointNormalSeparate(rest_of_block, means, cov)

        for rv in dist_new:
            rv.variability_level = VariabilityLevel.IIV

        rvs_new += dist_new

    return rvs_new


def _has_fixed_params(model, rv):
    param_names = model.random_variables.get_eta_params(rv)

    for p in model.parameters:
        if p.name in param_names and p.fix:
            return True
    return False


def _create_cov_matrix(model, rvs):
    cov = sympy.zeros(len(rvs))
    pset = model.parameters
    rv_map = _create_rv_map(model, rvs)

    for row in range(len(rvs)):  # TODO: keep old covariances?
        for col in range(row + 1):
            if row == col:
                cov[row, col] = rv_map[rvs[row]]
            else:
                covariance_init = _choose_param_init(model)
                param = Parameter(f'block_rv_{row}_{col}', covariance_init)
                pset.add(param)
                cov[row, col] = S(param.name)
                cov[col, row] = S(param.name)

    return pset, cov


def _create_rv_map(model, rvs):
    rv_map = dict()
    for rv, dist in rvs.distributions():
        param = model.parameters[str(list(dist.free_symbols)[0])]
        rv = rv[0]
        rv_map[rv] = S(param.name)
    return rv_map


def _choose_param_init(model):
    return 0.9


def _create_distribution(rvs, cov):
    means = []
    names = []

    for rv in rvs:
        means.append(0)
        names.append(rv.name)

    dist_new = JointNormalSeparate(names, means, cov)

    for rv in dist_new:
        rv.variability_level = VariabilityLevel.IIV

    return dist_new
