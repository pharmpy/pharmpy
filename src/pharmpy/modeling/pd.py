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
from pharmpy.modeling import create_symbol, get_central_volume_and_clearance, set_initial_condition

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
        Updated Pharmpy model

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
    return model.update_source()


def set_direct_effect(model: Model, expr: PDTypes, variable: Optional[str] = None):
    r"""Add an effect to a model.

    Effects are by default using concentratrion, but any user specified
    variable in the model can be used. Implemented PD models are:


    * Linear:

        .. math:: E = B \cdot (1 + \text{slope} \cdot C)

    * Emax:

        .. math:: E = B \cdot \Bigg(1 + \frac {E_{max} \cdot C } { EC_{50} + C}  \Bigg)

    * Step effect:

        .. math::  E=\Biggl \lbrace {B \cdot (1+ E_{max}) \quad \text{if C}>0 \atop B \quad \text{else}}

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
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_direct_effect(model, "linear")
    >>> model.statements.find_assignment("E")
        SLOPE⋅A_CENTRAL(t)
        ──────────────────
    E =         VC

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
        concentration = True
    else:
        variable_symb = Expr.symbol(variable)
        if (
            variable not in model.datainfo.names
            and variable_symb not in model.statements.free_symbols
        ):
            raise ValueError(f'Variable {variable} could not be found')
        concentration = False
    model = _add_effect(model, expr, variable_symb, concentration)

    return model.update_source()


def _add_baseline_effect(model: Model):
    b = model.statements.find_assignment("B")
    if b is None:
        model = add_individual_parameter(model, "B")
    return model


def _handle_zero(model: Model, expr: Expr):
    idv = model.datainfo.idv_column.symbol
    if expr.is_piecewise():
        args = expr.piecewise_args
        non_zero_arg = args[0] if args[0][0] != 0 else args[1]
        condition = (idv <= 0) | (~non_zero_arg[1])
        expr = non_zero_arg[0]
        expr = Expr.piecewise((Expr.integer(0), condition), (expr, True))
    else:
        expr = Expr.piecewise((Expr.integer(0), idv <= 0), (expr, True))
    return expr


def _add_drug_effect(model: Model, expr: str, conc, zero_handled=True):
    if expr == "linear":
        s = create_symbol(model, "SLOPE")
        model = add_individual_parameter(model, s.name, lower=-float("inf"))
        effect = s * conc
        if not zero_handled:
            effect = _handle_zero(model, effect)
    elif expr == "emax":
        emax = Expr.symbol("E_MAX")
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        x50 = _get_x50(model, conc)
        model = add_individual_parameter(model, x50.name)
        effect = emax * conc / (x50 + conc)
        if not zero_handled:
            effect = _handle_zero(model, effect)
    elif expr == "step":
        emax = Expr.symbol("E_MAX")
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        effect = Expr.piecewise((emax, conc > 0), (Expr.integer(0), True))
        if not zero_handled:
            effect = _handle_zero(model, effect)
    elif expr == "sigmoid":
        emax = Expr.symbol("E_MAX")
        model = add_individual_parameter(model, emax.name, lower=-1.0)
        x50 = _get_x50(model, conc)
        model = add_individual_parameter(model, x50.name)
        n = Expr.symbol("N")  # Hill coefficient
        model = add_individual_parameter(model, n.name)
        model = set_initial_estimates(model, {"POP_N": 1})
        effect = Expr.piecewise(
            ((emax * conc**n / (x50**n + conc**n)), conc > 0), (Expr.integer(0), True)
        )
        if not zero_handled:
            effect = _handle_zero(model, effect)
    elif expr == "loglin":
        s = Expr.symbol("SLOPE")
        e0 = Expr.symbol("B")
        model = add_individual_parameter(model, s.name, lower=-float("inf"))
        effect = s * (conc + (e0 / s).exp()).log()
        if not zero_handled:
            effect = _handle_zero(model, effect)
    else:
        raise ValueError(f'Unknown model "{expr}".')

    E = Assignment(Expr.symbol("E"), effect)
    e_index = model.statements.find_assignment_index("E")
    if e_index is None:
        r_index = model.statements.find_assignment_index("R")
        if r_index is None:
            model = model.replace(statements=model.statements + E)
        else:
            statements = model.statements[0:r_index] + E + model.statements[r_index:]
            model = model.replace(statements=statements)
    else:
        statements = model.statements[0:e_index] + E + model.statements[e_index + 1 :]
        model = model.replace(statements=statements)

    return model


def _get_x50(model, conc):
    x50_name = "EC_50"
    conc_assign = model.statements.find_assignment(conc)
    if conc_assign:
        central = model.statements.ode_system.central_compartment
        elimination_rate = model.statements.ode_system.get_flow(central, output)
        amount = central.amount
        if conc_assign.expression == amount * elimination_rate:
            x50_name = "EDK_50"
        elif conc_assign.expression == amount:
            x50_name = "A_50"
        else:
            x50_name = "EC_50"
    return Expr.symbol(x50_name)


def _add_response(model: Model, expr: str):
    r_index = model.statements.find_assignment_index("R")
    if expr != "loglin":
        if r_index is not None:
            assignment = model.statements[r_index]
            assert isinstance(assignment, Assignment)
            expression = assignment.expression * (Expr.integer(1) + Expr.symbol("E"))
            assignment = Assignment(Expr.symbol("R"), expression)
            statements = model.statements[0:r_index] + assignment + model.statements[r_index + 1 :]
        else:
            b = Expr.symbol("B")
            assignment = Assignment(Expr.symbol("R"), b * (Expr.integer(1) + Expr.symbol("E")))
            statements = model.statements + assignment
        model = model.replace(statements=statements)
    return model


def _add_dependent_variable(model: Model, expr: str):
    dv, *_ = model.dependent_variables

    a = model.statements.get_assignment(dv)
    R = Expr.symbol("R")
    if R not in a.expression.free_symbols:
        # Add dependent variable Y_2
        y_2 = Expr.symbol('Y_2')
        if expr != 'loglin':
            y = Assignment.create(y_2, R)
        else:
            y = Assignment.create(y_2, Expr.symbol("E"))
        dvs = model.dependent_variables.replace(y_2, 2)
        model = model.replace(statements=model.statements + y, dependent_variables=dvs)

        # Add error model
        model = set_proportional_error_model(model, dv=2, zero_protection=False)
    return model


def _add_effect(model: Model, expr: str, conc, zero_handled=True):
    model = _add_baseline_effect(model)
    model = _add_drug_effect(model, expr, conc, zero_handled)
    model = _add_response(model, expr)
    model = _add_dependent_variable(model, expr)
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
        Updated Pharmpy model

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
        Updated Pharmpy model

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


def add_placebo_model(
    model: Model,
    expr: Literal['linear', 'exp', 'hyperbolic'],
    operator: Literal['*', '+', 'prop'] = '*',
):
    r"""Add a placebo or disease progression effect to a model.

    .. warning:: This function is under development.

    * linear

        .. math:: R = B + \text{slope} \cdot \text{TIME}

    * exp

        .. math:: R = B \cdot e^{\frac{-t}{t_D}}

    * hyperbolic

        .. math:: R = B \cdot \frac{t_{50}}{t + t_{50}}

    :math:`B` is the baseline effect

    Parameters
    ----------
    model : Model
        Pharmpy model
    expr : str
        Name of placebo/disease progression effect function.
    operator : str
        Operator to use for combining the baseline with the placebo/disease progression

    Return
    ------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = create_basic_pd_model()
    >>> model = add_placebo_model(model, "linear")
    >>> model.statements.find_assignment("PDP")
    PDP = SLOPE⋅TIME

    """

    r_index = model.statements.find_assignment_index("R")
    if r_index is None:
        raise ValueError("Cannot find response variable R. Is this a PD model?")

    P = Expr.symbol("PDP")

    p_index = model.statements.find_assignment_index("PDP")
    if p_index is not None:
        raise ValueError("PDP already in the model. Not yet supported")

    idv = Expr.symbol(model.datainfo.idv_column.name)
    old_rassign = model.statements.get_assignment("R")

    def operate(lhs, rhs, operator):
        if operator == '*':
            return lhs * rhs
        elif operator == '+':
            return lhs + rhs
        elif operator == 'prop':
            return lhs * (1 + rhs)
        else:
            raise ValueError(f"Unknown operator {operator}")

    if expr == 'linear':
        slope = create_symbol(model, "SLOPE")
        model = add_individual_parameter(model, slope.name, lower=-float("inf"))
        passign_expr = slope * idv
        rassign_expr = operate(old_rassign.expression, P, operator)
    elif expr == 'exp':
        if operator != '*':
            raise ValueError('Only * is supported for exp')
        td = create_symbol(model, "TD")
        model = add_individual_parameter(model, td.name)
        passign_expr = (-idv / td).exp()
        rassign_expr = old_rassign.expression * P
    elif expr == 'hyperbolic':
        t50 = create_symbol(model, "T50")
        model = add_individual_parameter(model, t50.name)
        passign_expr = t50 / (idv + t50)
        rassign_expr = operate(old_rassign.expression, P, operator)
    else:
        raise ValueError(f"Unknown placebo model {expr}")

    passign = Assignment(P, passign_expr)
    new_rassign = Assignment(old_rassign.symbol, rassign_expr)

    r_index = model.statements.get_assignment_index("R")
    statements = (
        model.statements[:r_index] + passign + new_rassign + model.statements[r_index + 1 :]
    )
    model = model.replace(statements=statements)
    model = model.update_source()
    return model
