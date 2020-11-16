import warnings

from pharmpy.random_variables import RandomVariables
from pharmpy.symbols import symbol as S


def remove_iiv(model, list_of_etas=None):
    etas = _get_etas(model, list_of_etas)
    omegas = RandomVariables(etas).variance_parameters()
    rvs, pset, sset = model.random_variables, model.parameters, model.statements

    for eta in etas:
        statement = sset.find_assignment(eta.name, is_symbol=False)
        statement.expression = statement.expression.subs(S(eta.name), 0)

        rvs.discard(eta)

    pset -= [pset[omega.name] for omega in omegas]

    model.random_variables = rvs
    model.parameters = pset
    model.statements = sset

    return model


def _get_etas(model, list_of_etas):  # TODO: move to model?
    rvs = model.random_variables

    if list_of_etas is None:
        return rvs.etas
    else:
        etas = []
        for eta in list_of_etas:
            try:
                etas.append(rvs[eta.upper()])
            except KeyError:
                warnings.warn(f'Random variable "{eta}" does not exist')
        return etas
