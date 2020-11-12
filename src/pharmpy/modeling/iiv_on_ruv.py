import warnings

import sympy
import sympy.stats as stats

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.symbols import symbol as S


def iiv_on_ruv(model, list_of_epsilons=None):
    """
    Multiplies epsilons with exponential (new) etas.

    Parameters
    ----------
    model : Model
        Pharmpy model to apply IIV on epsilons.
    list_of_epsilons : list
        List of epsilons to multiply with exponential etas. If None, all epsilons will
        be chosen. None is default.
    """
    epsilons = _get_epsilons(model, list_of_epsilons)

    rvs, pset, sset = model.random_variables, model.parameters, model.statements

    for eps in epsilons:
        omega = S(f'IIV_RUV_{eps}')
        pset.add(Parameter(str(omega), 0.1))

        eta = stats.Normal(f'RV_{eps}', 0, sympy.sqrt(omega))
        eta.variability_level = VariabilityLevel.IIV
        rvs.add(eta)

        statement = sset.find_assignment(eps.name, is_symbol=False)
        statement.expression = statement.expression.subs(
            S(eps.name), S(eps.name) * sympy.exp(S(eta.name))
        )

    model.random_variables = rvs
    model.parameters = pset
    model.statements = sset

    model.modelfit_results = None

    return model


def _get_epsilons(model, list_of_epsilons):
    rvs = model.random_variables

    if list_of_epsilons is None:
        return rvs.ruv_rvs
    else:
        epsilons = []
        for eps in list_of_epsilons:
            try:
                epsilons.append(rvs[eps.upper()])
            except KeyError:
                warnings.warn(f'Epsilon "{eps}" does not exist')
        return epsilons
