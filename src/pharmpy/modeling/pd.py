"""
:meta private:
"""
from pharmpy.deps import sympy
from pharmpy.model import Assignment, Model, output

from .error import set_proportional_error_model
from .odes import add_individual_parameter, set_initial_estimates


def set_direct_effect(model: Model, expr: str):
    """Add an effect to a model

    Parameters
    ----------
    model : Model
        Pharmpy model
    expr : str
        Name of PD effect function. The function can either be
        * baseline: E = E0
        * step effect: Emax = theta if C > 0 else 0, E = E0 + Emax
        * linear: E = E0 + S * C
        * Emax: E = E0 + Emax * C / (EC50 + C)
        * sigmoidal: E = Emax * C^n / (EC50^n + C^n)
        * log-lin: E = m * log(C + C0)
        Valid strings are: baseline, linear, Emax, sigmoid, step, loglin

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------

    """
    e0 = sympy.Symbol("E0")
    model = add_individual_parameter(model, e0.name)
    if expr in ["Emax", "sigmoid", "step"]:
        emax = sympy.Symbol("E_max")
        model = add_individual_parameter(model, emax.name)
    if expr in ["Emax", "sigmoid"]:
        ec50 = sympy.Symbol("EC_50")
        model = add_individual_parameter(model, ec50.name)

    # vc = sympy.Symbol("VC") # must be changed later
    vc, cl = _get_central_volume_and_cl(model)
    conc = model.statements.ode_system.central_compartment.amount / vc

    # Add effect E
    if expr == "baseline":
        E = Assignment(sympy.Symbol('E'), e0)
    elif expr == "linear":
        s = sympy.Symbol("S")  # slope
        model = add_individual_parameter(model, s.name)
        E = Assignment(sympy.Symbol('E'), e0 + s * conc)
    elif expr == "Emax":
        E = Assignment(sympy.Symbol("E"), e0 + emax * conc / (ec50 + conc))
    elif expr == "step":
        E = Assignment(sympy.Symbol("E"), sympy.Piecewise((0, conc < 0), (emax, True)))
    elif expr == "sigmoid":
        n = sympy.Symbol("n")
        model = add_individual_parameter(model, n.name)
        model = set_initial_estimates(model, {"POP_n": 1})
        E = Assignment(sympy.Symbol("E"), emax * conc**n / (ec50**n + conc**n))
    elif expr == "loglin":
        m = sympy.Symbol("m")  # slope
        model = add_individual_parameter(model, m.name)
        E = Assignment(sympy.Symbol("E"), m * sympy.log(conc + sympy.exp(e0 / m)))
    else:
        raise ValueError(f'Unknown model "{expr}".')

    # Add dependent variable Y_2
    y_2 = sympy.Symbol('Y_2')
    y = Assignment(y_2, E.symbol)
    dvs = model.dependent_variables.replace(y_2, 2)
    model = model.replace(statements=model.statements + E + y, dependent_variables=dvs)

    # Add error model
    model = set_proportional_error_model(model, dv=2, zero_protection=False)
    return model


def _get_central_volume_and_cl(model):
    odes = model.statements.ode_system
    central_comp = odes.central_compartment
    elimination_rate = odes.get_flow(central_comp, output)
    numer, denom = elimination_rate.as_numer_denom()
    if denom != 1:
        vc = denom
        cl = numer
    else:
        raise ValueError('Model is not suitable')
    return vc, cl
