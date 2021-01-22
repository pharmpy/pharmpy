"""
:meta private:
"""

import warnings

import sympy
import sympy.stats as stats

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.symbols import symbol as S


def iiv_on_ruv(model, list_of_eps=None, same_eta=True, eta_names=None):
    """
    Multiplies epsilons with exponential (new) etas. Initial estimates for new etas are 0.09.

    Parameters
    ----------
    model : Model
        Pharmpy model to apply IIV on epsilons.
    list_of_eps : list
        List of epsilons to multiply with exponential etas. If None, all epsilons will
        be chosen. None is default.
    same_eta : bool
        Boolean of whether all RUVs from input should use the same new ETA or if one ETA
        should be created for each RUV. True is default.
    """
    eps = _get_epsilons(model, list_of_eps)

    rvs, pset, sset = model.random_variables, model.parameters, model.statements

    if same_eta:
        eta = _create_eta(pset, 1, eta_names)
        rvs.add(eta)
        eta_dict = {e: eta for e in eps}
    else:
        etas = [_create_eta(pset, i + 1, eta_names) for i in range(len(eps))]
        for eta in etas:
            rvs.add(eta)
        eta_dict = {e: eta for e, eta in zip(eps, etas)}

    for e in eps:
        statement = sset.find_assignment(e.name, is_symbol=False)
        statement.expression = statement.expression.subs(
            S(e.name), S(e.name) * sympy.exp(S(eta_dict[e].name))
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


def _create_eta(pset, number, eta_names):
    omega = S(f'IIV_RUV{number}')
    pset.add(Parameter(str(omega), 0.09))

    if eta_names:
        eta_name = eta_names[number - 1]
    else:
        eta_name = f'RV{number}'

    eta = stats.Normal(eta_name, 0, sympy.sqrt(omega))
    eta.variability_level = VariabilityLevel.IIV

    return eta
