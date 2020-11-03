from sympy.stats.joint_rv_types import MultivariateNormalDistribution

from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariables, VariabilityLevel


class RVInputException(Exception):
    pass


def create_rv_block(model, list_of_rvs=None):
    rvs_full = model.random_variables
    rvs = _get_rvs(model, list_of_rvs)

    for rv in rvs:
        if isinstance(rv.pspace.distribution, MultivariateNormalDistribution):
            rv_extracted = rvs_full.extract_from_block(rv)
            rvs.discard(rv)
            rvs.add(rv_extracted)

    pset = _merge_rvs(model, rvs)

    model.parameters = pset

    rvs_new = RandomVariables()

    for rv in rvs_full:
        if rv.name not in [rv.name for rv in rvs.etas]:
            rvs_new.add(rv)

    for rv in rvs:
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

    params = rvs.merge_normal_distributions(create_cov_params=True)

    for rv in rvs:
        rv.variability_level = VariabilityLevel.IIV

    for p_name in params.keys():
        covariance_init = _choose_param_init(model)
        param_new = Parameter(p_name, covariance_init)
        pset.add(param_new)

    return pset


def _choose_param_init(model):
    return 0.9
