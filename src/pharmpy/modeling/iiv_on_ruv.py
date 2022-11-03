"""
:meta private:
"""

from pharmpy.deps import sympy
from pharmpy.model import NormalDistribution, Parameter, Parameters
from pharmpy.modeling.help_functions import _format_input_list, _get_epsilons


def set_iiv_on_ruv(model, list_of_eps=None, same_eta=True, eta_names=None):
    """
    Multiplies epsilons with exponential (new) etas.

    Initial variance for new etas is 0.09.

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
    eta_names : str, list
        Custom names of new etas. Must be equal to the number epsilons or 1 if same eta.

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_iiv_on_ruv(model)   # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("Y")
                  ETA_RV1
    Y = EPS(1)⋅W⋅ℯ        + F

    See also
    --------
    set_power_on_ruv

    """
    list_of_eps = _format_input_list(list_of_eps)
    eps = _get_epsilons(model, list_of_eps)

    if eta_names and len(eta_names) != len(eps):
        raise ValueError(
            'The number of provided eta names must be equal to the number of epsilons.'
        )

    rvs, pset, sset = model.random_variables, list(model.parameters), model.statements

    if same_eta:
        eta = _create_eta(pset, 1, eta_names)
        rvs = rvs + eta
        eta_dict = {e: eta for e in eps}
    else:
        etas = [_create_eta(pset, i + 1, eta_names) for i in range(len(eps))]
        rvs = rvs + etas
        eta_dict = dict(zip(eps, etas))

    for e in eps:
        sset = sset.subs(
            {
                sympy.Symbol(e.names[0]): sympy.Symbol(e.names[0])
                * sympy.exp(sympy.Symbol(eta_dict[e].names[0]))
            }
        )

    model.random_variables = rvs
    model.parameters = Parameters(pset)
    model.statements = sset

    return model


def _create_eta(pset, number, eta_names):
    omega = sympy.Symbol(f'IIV_RUV{number}')
    pset.append(Parameter(str(omega), 0.09))

    if eta_names:
        eta_name = eta_names[number - 1]
    else:
        eta_name = f'ETA_RV{number}'

    eta = NormalDistribution.create(eta_name, 'iiv', 0, omega)
    return eta
