import warnings

import sympy
import sympy.stats as stats

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.symbols import symbol as S


def iiv_on_ruv(model, list_of_eps=None):
    """
    Multiplies epsilons with exponential (new) etas.

    Parameters
    ----------
    model : Model
        Pharmpy model to apply IIV on epsilons.
    list_of_eps : list
        List of epsilons to multiply with exponential etas. If None, all epsilons will
        be chosen. None is default.
    """
    eps = _get_epsilons(model, list_of_eps)

    rvs, pset, sset = model.random_variables, model.parameters, model.statements

    for i, e in enumerate(eps):
        omega = S(f'IIV_RUV{i + 1}')
        pset.add(Parameter(str(omega), 0.01))

        eta = stats.Normal(f'RV_{e}', 0, sympy.sqrt(omega))
        eta.variability_level = VariabilityLevel.IIV
        rvs.add(eta)

        statement = sset.find_assignment(e.name, is_symbol=False)
        statement.expression = statement.expression.subs(
            S(e.name), S(e.name) * sympy.exp(S(eta.name))
        )

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
