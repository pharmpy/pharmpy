"""
:meta private:
"""

from __future__ import annotations

from typing import Literal, Optional

from pharmpy.basic import Expr
from pharmpy.model import (
    Assignment,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Model,
    Statements,
    get_and_check_odes,
    output,
)
from pharmpy.modeling import get_central_volume_and_clearance, set_initial_condition

from .error import set_proportional_error_model
from .odes import add_individual_parameter, set_initial_estimates

PDTypes = Literal['linear', 'emax', 'sigmoid', 'step', 'loglin']


def add_effect_compartment(model: Model, expr: PDTypes):
    r"""Add an effect compartment.

    Implemented PD models are:


    * Linear:

        .. math:: E = B \cdot (1 + \text{slope} \cdot C)

    * Emax:

        .. math:: E = B \cdot \Bigg(1 + \frac {E_{max} \cdot C } { EC_{50} + C}  \Bigg)

    * Step effect:

        .. math:: E = \Biggl \lbrace {B \quad \text{if C} \leq 0 \atop B \cdot (1+ E_{max}) \quad \text{else}}

    * Sigmoidal:

        .. math::  E=\Biggl \lbrace {B \cdot \Bigl(1+\frac{E_{max} \cdot C^n}{EC_{50}^n+C^n}\Bigl) \quad \
                    \text{if C}>0 \atop B \quad \text{else}}

    * Log-linear:

        .. math:: E = \text{slope} \cdot \text{log}(C + C_0)

    :math:`B` is the baseline effect

    Parameters
    ----------
    model : Model
        Pharmpy model
    expr : {'linear', 'emax', 'sigmoid', 'step', 'loglin'}
        Name of the PD effect function.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_effect_compartment(model, "linear")
    >>> model.statements.ode_system.find_compartment("EFFECT")
    Compartment(EFFECT, amount=A_EFFECT(t), input=KE0*A_CENTRAL(t)/VC)
    """
    vc, cl = get_central_volume_and_clearance(model)

    odes = get_and_check_odes(model)
    central = odes.central_compartment
    cb = CompartmentalSystemBuilder(odes)

    ke0 = Expr.symbol("KE0")
    met = Expr.symbol('MET')
    model = add_individual_parameter(model, met.name)
    ke0_ass = Assignment.create(ke0, 1 / met)

    effect = Compartment.create("EFFECT", input=ke0 * central.amount / vc)
    cb.add_compartment(effect)
    cb.add_flow(effect, output, ke0)

    model = model.replace(
        statements=Statements(
            model.statements.before_odes
            + ke0_ass
            + CompartmentalSystem(cb)
            + model.statements.after_odes
        )
    )

    model = _add_effect(model, expr, effect.amount)
    return model


def set_direct_effect(model: Model, expr: PDTypes, variable: Optional[str] = None):
    r"""Add an effect to a model.

    Effects are by default using concentratrion, but any user specified
    variable in the model can be used. Implemented PD models are:


    * Linear:

        .. math:: E = B \cdot (1 + \text{slope} \cdot C)

    * Emax:

        .. math:: E = B \cdot \Bigg(1 + \frac {E_{max} \cdot C } { EC_{50} + C}  \Bigg)

    * Step effect:

        .. math:: E = \Biggl \lbrace {B \quad \text{if C} \leq 0 \atop B \cdot (1+ E_{max}) \quad \text{else}}

    * Sigmoidal:

        .. math::  E=\Biggl \lbrace {B \cdot \Bigl(1+\frac{E_{max} \cdot C^n}{EC_{50}^n+C^n}\Bigl) \quad \
                    \text{if C}>0 \atop B \quad \text{else}}

    * Log-linear:

        .. math:: E = \text{slope} \cdot \text{log}(C + C_0)

    :math:`B` is the baseline effect

    Parameters
    ----------
    model : Model
        Pharmpy model
    expr : {'linear', 'emax', 'sigmoid', 'step', 'loglin'}
        Name of PD effect function.
    variable : str
        Name of variable to use (if None concentration will be used)

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_direct_effect(model, "linear")
    >>> model.statements.find_assignment("E")
          ⎛SLOPE⋅A_CENTRAL(t)    ⎞
        B⋅⎜────────────────── + 1⎟
    E =   ⎝        VC            ⎠

    """
    if variable is None:
        vc, cl = get_central_volume_and_clearance(model)
        odes = get_and_check_odes(model)
        conc_expr = odes.central_compartment.amount / vc
        variable_symb = conc_expr
        for s in model.statements.after_odes:
            if s.expression == conc_expr:
                variable_symb = s.symbol
                break
    else:
        variable_symb = Expr.symbol(variable)
        if (
            variable not in model.datainfo.names
            and variable_symb not in model.statements.free_symbols
        ):
            raise ValueError(f'Variable {variable} could not be found')
    model = _add_effect(model, expr, variable_symb)

    return model


def _add_effect(model: Model, expr: str, conc):
    e0 = Expr.symbol("B")
    model = add_individual_parameter(model, e0.name)

    # Add effect E
    if expr == "linear":
        s = Expr.symbol("SLOPE")
        model = add_individual_parameter(model, s.name, lower=-float("inf"))
        E = Assignment(Expr.symbol('E'), e0 * (1 + (s * conc)))
    elif expr == "emax":
        emax = Expr.symbol("E_MAX")
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        ec50 = Expr.symbol("EC_50")
        model = add_individual_parameter(model, ec50.name)
        E = Assignment(Expr.symbol("E"), e0 * (1 + (emax * conc / (ec50 + conc))))
    elif expr == "step":
        emax = Expr.symbol("E_MAX")
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        E = Assignment(Expr.symbol("E"), Expr.piecewise((e0, conc <= 0), (e0 * (1 + emax), True)))
    elif expr == "sigmoid":
        emax = Expr.symbol("E_MAX")
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        ec50 = Expr.symbol("EC_50")
        model = add_individual_parameter(model, ec50.name)
        n = Expr.symbol("N")  # Hill coefficient
        model = add_individual_parameter(model, n.name)
        model = set_initial_estimates(model, {"POP_N": 1})
        E = Assignment.create(
            Expr.symbol("E"),
            Expr.piecewise(
                ((e0 * (1 + (emax * conc**n / (ec50**n + conc**n)))), conc > 0), (e0, True)
            ),
        )
    elif expr == "loglin":
        s = Expr.symbol("SLOPE")
        model = add_individual_parameter(model, s.name, lower=-float("inf"))
        E = Assignment(Expr.symbol("E"), s * (conc + (e0 / s).exp()).log())
    else:
        raise ValueError(f'Unknown model "{expr}".')

    # Add dependent variable Y_2
    y_2 = Expr.symbol('Y_2')
    y = Assignment.create(y_2, E.symbol)
    dvs = model.dependent_variables.replace(y_2, 2)
    model = model.replace(statements=model.statements + E + y, dependent_variables=dvs)

    # Add error model
    model = set_proportional_error_model(model, dv=2, zero_protection=False)

    return model


def add_indirect_effect(
    model: Model,
    expr: Literal['linear', 'emax', 'sigmoid', 'step'],
    prod: bool = True,
):
    r"""Add indirect (turnover) effect

    The concentration :math:`C_c` has an impact on the production or degradation rate of the response  R:

    * Production:

        .. math:: \frac {dR}{dt} = k_{in} \cdot (1 + f(C_c)) - k_{out} \cdot R

    * Degradation:

        .. math:: \frac {dR}{dt} = k_{in} - k_{out} \cdot (1 + f(C_c)) \cdot R

    :math:`k_{in}` and :math:`k_{out}` can either be inhibited or stimulated.
    Baseline :math:`B = R(0) = R_0 = k_{in}/k_{out}`.

    Models:

    * Linear:

        .. math:: f(C_c) = \text{slope} \cdot C_c

    * Emax:

        .. math:: f(C_c) = \frac {E_{max} \cdot C_c } { EC_{50} + C_c }

    * Sigmoidal:

        .. math::  f(C_c) = \frac{E_{max} \cdot C_c^n}{EC_{50}^n+C^n}


    Parameters
    ----------
    model : Model
        Pharmpy model
    prod : bool
        Production (True) (default) or degradation (False)
    expr : {'linear', 'emax', 'sigmoid', 'step'}
        Name of PD effect function.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_indirect_effect(model, expr='linear', prod=True)

    """
    vc, cl = get_central_volume_and_clearance(model)
    odes = get_and_check_odes(model)
    central = odes.central_compartment
    conc_c = central.amount / vc

    response = Compartment.create("RESPONSE")
    a_response = response.amount

    kin = Expr.symbol("K_IN")
    kout = Expr.symbol("K_OUT")
    met = Expr.symbol('MET')
    model = add_individual_parameter(model, met.name)
    b = Expr.symbol("B")  # baseline
    model = add_individual_parameter(model, b.name)

    kout_ass = Assignment.create(kout, 1 / met)
    kin_ass = Assignment.create(kin, kout * b)

    if expr == 'linear':
        s = Expr.symbol("SLOPE")
        model = add_individual_parameter(model, s.name, lower=-float("inf"))
        R = Expr.symbol("SLOPE") * conc_c
    elif expr == 'emax':
        emax = Expr.symbol("E_MAX")
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        ec50 = Expr.symbol("EC_50")
        model = add_individual_parameter(model, ec50.name)
        R = emax * conc_c / (ec50 + conc_c)
    elif expr == 'sigmoid':
        emax = Expr.symbol("E_MAX")
        ec50 = Expr.symbol("EC_50")
        n = Expr.symbol("N")
        model = add_individual_parameter(model, n.name)
        model = set_initial_estimates(model, {"POP_N": 1})
        model = add_individual_parameter(model, ec50.name)
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        R = emax * conc_c**n / (ec50**n + conc_c**n)
    else:
        raise ValueError(f'Unknown model "{expr}".')

    cb = CompartmentalSystemBuilder(odes)
    if prod:
        response = Compartment.create("RESPONSE", input=kin * (1 + R))
        cb.add_compartment(response)
        cb.add_flow(response, output, kout)
    elif not prod:
        response = Compartment.create("RESPONSE", input=kin)
        cb.add_compartment(response)
        cb.add_flow(response, output, kout * (1 + R))

    model = model.replace(
        statements=Statements(
            model.statements.before_odes
            + kout_ass
            + kin_ass
            + CompartmentalSystem(cb)
            + model.statements.after_odes
        )
    )

    model = set_initial_condition(model, "RESPONSE", b)

    # Add dependent variable Y_2
    y_2 = Expr.symbol('Y_2')
    y = Assignment(y_2, a_response)
    dvs = model.dependent_variables.replace(y_2, 2)
    model = model.replace(statements=model.statements + y, dependent_variables=dvs)

    # Add error model
    model = set_proportional_error_model(model, dv=2, zero_protection=False)

    return model.update_source()


def set_baseline_effect(model: Model, expr: str = 'const'):
    r"""Create baseline effect model.

    Currently implemented baseline effects are:

    Constant baseline effect (const):

        .. math:: E = B

    Parameters
    ----------
    model : Model
        Pharmpy model
    expr : str
        Name of baseline effect function.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_baseline_effect(model, expr='const')
    >>> model.statements.find_assignment("E")
    E = B
    """
    e0 = Expr.symbol("B")
    model = add_individual_parameter(model, e0.name)

    E = Assignment(Expr.symbol('E'), e0)

    # Add dependent variable Y_2
    y_2 = Expr.symbol('Y_2')
    y = Assignment.create(y_2, E.symbol)
    dvs = model.dependent_variables.replace(y_2, 2)
    model = model.replace(statements=model.statements + E + y, dependent_variables=dvs)

    # Add error model
    model = set_proportional_error_model(model, dv=2, zero_protection=False)

    return model
