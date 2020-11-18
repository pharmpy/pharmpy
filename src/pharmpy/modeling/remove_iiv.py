import warnings

from pharmpy.random_variables import RandomVariables
from pharmpy.symbols import symbol as S


def remove_iiv(model, list_of_etas=None):
    rvs, pset, sset = model.random_variables, model.parameters, model.statements
    etas = _get_etas(model, list_of_etas)

    for eta in etas:
        rvs.discard(eta)

        statement = sset.find_assignment(eta.name, is_symbol=False)
        statement.expression = statement.expression.subs(S(eta.name), 0)

    model.random_variables = rvs
    model.parameters = pset
    model.statements = sset

    return model


def _get_etas(model, list_of_etas):
    rvs = model.random_variables

    if list_of_etas is None:
        return RandomVariables(rvs.etas)
    else:
        etas = []
        for eta in list_of_etas:
            try:
                etas.append(rvs[eta.upper()])
            except KeyError:
                warnings.warn(f'Random variable "{eta}" does not exist')
        return RandomVariables(etas)
