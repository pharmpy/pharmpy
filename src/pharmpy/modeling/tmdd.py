"""
:meta private:
"""
from pharmpy.deps import sympy
from pharmpy.model import (
    Assignment,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Model,
    output,
)

from .odes import add_individual_parameter, set_first_order_elimination, set_initial_condition
from .parameter_variability import add_iiv


def set_tmdd(model: Model, type: str):
    """Sets target mediated drug disposition

    Sets target mediated drug disposition to a PK model.

    Supported models are full, ib, cr, crib, qss, wagner and mmapp.

    Parameters
    ----------
    model : Model
        Pharmpy model
    type : str
        Type of TMDD model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_tmdd(model, "full")

    """
    type = type.upper()

    if type != "MMAPP":
        model = set_first_order_elimination(model)

    odes = model.statements.ode_system
    central = odes.central_compartment
    central_amount = sympy.Function(central.amount.name)(sympy.Symbol('t'))
    cb = CompartmentalSystemBuilder(odes)

    vc, cl = _get_central_volume_and_cl(model)
    r_0 = sympy.Symbol('R_0')
    model = add_individual_parameter(model, r_0.name)
    model = add_iiv(model, [r_0], 'exp')
    kint = sympy.Symbol('KINT')
    model = add_individual_parameter(model, kint.name)

    if type == "FULL":
        model, kon, koff, kdeg = _create_parameters(model, ['KON', 'KOFF', 'KDEG'])
        target_comp, target_amount, complex_comp, complex_amount = _create_compartments(
            cb, ['TARGET', 'COMPLEX']
        )
        ksyn, ksyn_ass = _create_ksyn()

        cb.add_flow(target_comp, complex_comp, kon * central_amount)
        cb.add_flow(complex_comp, target_comp, koff)
        cb.add_flow(target_comp, output, kdeg)
        cb.add_flow(complex_comp, output, kint)
        cb.set_input(target_comp, ksyn * vc)
        cb.set_input(central, koff * complex_amount - kon * central_amount * target_amount)

        before = model.statements.before_odes + ksyn_ass
        after = model.statements.after_odes
    elif type == "IB":
        model, kdeg, kon = _create_parameters(model, ['KDEG', 'KON'])
        target_comp, target_amount, complex_comp, complex_amount = _create_compartments(
            cb, ['TARGET', 'COMPLEX']
        )
        ksyn, ksyn_ass = _create_ksyn()

        cb.add_flow(target_comp, complex_comp, kon * central_amount / vc)
        cb.add_flow(target_comp, output, kdeg)
        cb.add_flow(complex_comp, output, kint)
        cb.set_input(target_comp, ksyn * vc)
        cb.set_input(central, -kon * central_amount * target_amount / vc)

        before = model.statements.before_odes + ksyn_ass
        after = model.statements.after_odes
    elif type == "CR":
        model, kon, koff = _create_parameters(model, ['KON', 'KOFF'])
        complex_comp, complex_amount = _create_compartments(cb, ['COMPLEX'])

        cb.add_flow(complex_comp, central, koff + kon * central_amount / vc)
        cb.add_flow(complex_comp, output, kint)
        cb.add_flow(central, complex_comp, kon * r_0)

        before = model.statements.before_odes
        after = model.statements.after_odes
    elif type == "CRIB":
        model, kon = _create_parameters(model, ['KON'])
        complex_comp, complex_amount = _create_compartments(cb, ['COMPLEX'])

        cb.add_flow(complex_comp, output, kint)
        cb.add_flow(central, complex_comp, kon * r_0)
        cb.add_flow(complex_comp, central, kon * central_amount / vc)

        before = model.statements.before_odes
        after = model.statements.after_odes
    elif type == "QSS":
        model, kdc, kdeg = _create_parameters(model, ['KDC', 'KDEG'])
        target_comp, target_amount = _create_compartments(cb, ['TARGET'])

        kd = sympy.Symbol('KD')

        ksyn, ksyn_ass = _create_ksyn()
        kd_ass = Assignment(kd, kdc * vc)

        lafree_symb = sympy.Symbol('LAFREE')
        lafree_expr = sympy.Rational(1, 2) * (
            central_amount
            - target_amount
            - kd
            + sympy.sqrt((central_amount - target_amount - kd) ** 2 + 4 * kd * central_amount)
        )
        lafree_ass = Assignment(lafree_symb, lafree_expr)

        # FIXME: Support two and three compartment distribution
        elimination_rate = odes.get_flow(central, output)
        cb.remove_flow(central, output)
        cb.set_input(
            central,
            -lafree_symb * elimination_rate
            - target_amount * kint * lafree_symb / (kd + lafree_symb),
        )
        cb.set_input(
            target_comp,
            ksyn * vc
            - kdeg * target_amount
            - (kint - kdeg) * target_amount * lafree_symb / (kd + lafree_symb),
        )

        # FIXME: should others also have flows?
        central = cb.find_compartment('CENTRAL')
        cb.add_flow(central, output, lafree_symb * elimination_rate)

        lafreef = sympy.Symbol("LAFREEF")
        lafree_final = Assignment(lafreef, lafree_expr)
        before = model.statements.before_odes + (ksyn_ass, kd_ass, lafree_ass)
        after = lafree_final + model.statements.after_odes
        ipred = lafreef / vc
        after = after.reassign(sympy.Symbol('IPRED'), ipred)  # FIXME: Assumes an IPRED
    elif type == 'WAGNER':
        model, km = _create_parameters(model, ['KM'])

        ke = odes.get_flow(central, output)
        kd = km * vc
        rinit = r_0 * vc

        lafree_symb = sympy.Symbol('LAFREE')
        lafree_expr = sympy.Rational(1, 2) * (
            central_amount
            - rinit
            - kd
            + sympy.sqrt((central_amount - rinit - kd) ** 2 + 4 * kd * central_amount)
        )
        lafree_ass = Assignment(lafree_symb, lafree_expr)

        cb.add_flow(central, output, -kint)
        cb.set_input(central, -(ke - kint) * lafree_symb)

        lafreef = sympy.Symbol("LAFREEF")
        lafree_final = Assignment(lafreef, lafree_expr)
        before = model.statements.before_odes + lafree_ass
        after = lafree_final + model.statements.after_odes
        ipred = lafreef / vc
        after = after.reassign(sympy.Symbol('IPRED'), ipred)  # FIXME: Assumes an IPRED
    elif type == 'MMAPP':
        model, kmc, kdeg = _create_parameters(model, ['KMC', 'KDEG'])
        target_comp, target_amount = _create_compartments(cb, ['TARGET'])
        ksyn, ksyn_ass = _create_ksyn()

        target_elim = kdeg + (kint - kdeg) * (central_amount / vc) / (
            kmc * vc + central_amount / vc
        )
        cb.add_flow(target_comp, output, target_elim)
        elim = (cl + target_amount * kint / (central_amount / vc + kmc * vc)) / vc
        cb.add_flow(central, output, elim)
        cb.set_input(target_comp, ksyn * vc)

        before = model.statements.before_odes + ksyn_ass
        after = model.statements.after_odes
    else:
        raise ValueError(f'Unknown TMDD type "{type}".')

    model = model.replace(statements=before + CompartmentalSystem(cb) + after)
    if type == 'WAGNER':
        model = set_initial_condition(model, central.name, r_0 * vc)
    elif type not in ['CR', 'CRIB']:
        model = set_initial_condition(model, "TARGET", r_0 * vc)

    return model.update_source()


def _create_parameters(model, names):
    symbs = []
    for name in names:
        symb = sympy.Symbol(name)
        symbs.append(symb)
        model = add_individual_parameter(model, symb.name)
    return model, *symbs


def _create_compartments(cb, names):
    comps = []
    for name in names:
        comp = Compartment.create(name=name)
        comps.append(comp)
        amount_func = sympy.Function(comp.amount.name)(sympy.Symbol('t'))
        comps.append(amount_func)
        cb.add_compartment(comp)
    return comps


def _get_central_volume_and_cl(model):
    odes = model.statements.ode_system
    central_comp = odes.central_compartment
    elimination_rate = odes.get_flow(central_comp, output)
    numer, denom = elimination_rate.as_numer_denom()
    if denom != 1:
        vc = denom
        cl = numer
    else:
        vc = sympy.Symbol('VC')  # FIXME: What do do here?
        cl = sympy.Integer(1)
    return vc, cl


def _create_ksyn():
    ksyn = sympy.Symbol('KSYN')
    ksyn_ass = Assignment(ksyn, sympy.Symbol("R_0") * sympy.Symbol("KDEG"))
    return ksyn, ksyn_ass
