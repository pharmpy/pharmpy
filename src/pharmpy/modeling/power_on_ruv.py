import re
import warnings

from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment
from pharmpy.symbols import symbol as S


def power_on_ruv(model, list_of_eps=None):
    eps = _get_epsilons(model, list_of_eps)
    rvs, pset, sset = model.random_variables, model.parameters, model.statements

    for i, e in enumerate(eps):
        eta_no = int(re.findall(r'\d', e.name)[0])

        symbol = S(f'EPSP{eta_no}')
        theta_name = str(model.create_symbol(stem='power', force_numbering=True))
        theta = Parameter(theta_name, 0.01)
        pset.add(theta)

        expression = e * (model.individual_prediction_symbol ** S(theta.name))
        assignment = Assignment(symbol, expression)

        assignment_first = sset.find_assignment(e.name, is_symbol=False, last=False)[0]
        index = sset.index(assignment_first)

        sset.subs({e.name: symbol.name})
        sset.insert(index, assignment)

    model.random_variables = rvs
    model.parameters = pset
    model.statements = sset

    model.modelfit_results = None

    return model


def _get_epsilons(model, list_of_eps):
    rvs = model.random_variables

    if list_of_eps is None:
        return rvs.ruv_rvs
    else:
        eps = []
        for e in list_of_eps:
            try:
                eps.append(rvs[e.upper()])
            except KeyError:
                warnings.warn(f'Epsilon "{e}" does not exist')
        return eps
