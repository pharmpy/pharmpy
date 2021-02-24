import warnings

from pharmpy.random_variables import VariabilityLevel


def _get_epsilons(model, list_of_eps):
    rvs = model.random_variables

    if list_of_eps is None:
        return rvs.ruv_rvs
    else:
        eps = []
        for eps_str in list_of_eps:
            try:
                eps.append(rvs[eps_str.upper()])
            except KeyError:
                warnings.warn(f'Epsilon "{eps_str}" does not exist')
        return eps


def _get_etas(model, list_of_etas, include_symbols=False, fixed_allowed=False, iov_allowed=False):
    rvs = model.random_variables
    all_valid_etas = False

    if list_of_etas is None:
        list_of_etas = [eta.name for eta in rvs.etas]
        all_valid_etas = True

    etas = []
    for eta_str in list_of_etas:
        try:
            eta = rvs[eta_str.upper()]
            if not fixed_allowed and _has_fixed_params(model, eta):
                if not all_valid_etas:
                    raise ValueError(f'Random variable cannot be set to fixed: {eta}')
                continue
            if not iov_allowed and eta.variability_level == VariabilityLevel.IOV:
                if not all_valid_etas:
                    raise ValueError(f'Random variable cannot be IOV: {eta}')
                continue
            if eta not in etas:
                etas.append(eta)
        except KeyError:
            if include_symbols:
                etas_symbs = _get_eta_symbs(eta_str, rvs, model.statements)
                etas += [eta for eta in etas_symbs if eta not in etas]
                continue
            raise KeyError(f'Random variable does not exist: {eta_str}')
    return etas


def _get_eta_symbs(eta_str, rvs, sset):
    try:
        exp_symbs = sset.find_assignment(eta_str).expression.free_symbols
    except AttributeError:
        raise KeyError(f'Symbol "{eta_str}" does not exist')
    return [rvs[str(e)] for e in exp_symbs.intersection(rvs.free_symbols)]


def _has_fixed_params(model, rv):
    param_names = model.random_variables.get_eta_params(rv)

    for p in model.parameters:
        if p.name in param_names and p.fix:
            return True
    return False


def _format_input_list(list_of_names):
    if list_of_names and isinstance(list_of_names, str):
        list_of_names = [list_of_names]
    return list_of_names


def _format_options(list_of_options, no_of_variables):
    options = []
    for option in list_of_options:
        if isinstance(option, str) or not option:
            option = [option] * no_of_variables
        options.append(option)

    return options
