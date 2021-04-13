"""
:meta private:
"""

import sympy

from pharmpy.modeling.help_functions import _format_input_list, _get_epsilons
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariable
from pharmpy.symbols import symbol as S


def iiv_on_ruv(model, list_of_eps=None, same_eta=True, eta_names=None):
    """
    Multiplies epsilons with exponential (new) etas. Initial estimates for new etas are 0.09.

    Parameters
    ----------
    model : Model
        Pharmpy model to apply IIV on epsilons.
    list_of_eps : str, list
        Name/names of epsilons to multiply with exponential etas. If None, all epsilons will
        be chosen. None is default.
    same_eta : bool
        Boolean of whether all RUVs from input should use the same new ETA or if one ETA
        should be created for each RUV. True is default.
    eta_names: str, list
        Custom names of new etas. Must be equal to the number epsilons or 1 if same eta.
    """
    list_of_eps = _format_input_list(list_of_eps)
    eps = _get_epsilons(model, list_of_eps)

    if eta_names and len(eta_names) != len(eps):
        raise ValueError(
            'The number of provided eta names must be equal to the number of epsilons.'
        )

    rvs, pset, sset = model.random_variables, model.parameters, model.statements

    if same_eta:
        eta = _create_eta(pset, 1, eta_names)
        rvs.append(eta)
        eta_dict = {e: eta for e in eps}
    else:
        etas = [_create_eta(pset, i + 1, eta_names) for i in range(len(eps))]
        for eta in etas:
            rvs.append(eta)
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


def _create_eta(pset, number, eta_names):
    omega = S(f'IIV_RUV{number}')
    pset.append(Parameter(str(omega), 0.09))

    if eta_names:
        eta_name = eta_names[number - 1]
    else:
        eta_name = f'ETA_RV{number}'

    eta = RandomVariable.normal(eta_name, 'iiv', 0, omega)
    return eta
