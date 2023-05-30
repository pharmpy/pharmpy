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

from .odes import add_individual_parameter, set_initial_condition


def set_tmdd(model: Model, type: str):
    """Sets target mediated drug disposition

    Sets target mediated drug disposition to a PK model.

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

    odes = model.statements.ode_system
    central = odes.central_compartment
    central_amount = sympy.Function(central.amount.name)(sympy.Symbol('t'))

    if type == "FULL":
        kon = sympy.Symbol('KON')
        model = add_individual_parameter(model, kon.name)
        koff = sympy.Symbol('KOFF')
        model = add_individual_parameter(model, koff.name)
        kin = sympy.Symbol('KIN')
        model = add_individual_parameter(model, kin.name)
        kout = sympy.Symbol('KOUT')
        model = add_individual_parameter(model, kout.name)
        kpe = sympy.Symbol('KPE')
        model = add_individual_parameter(model, kpe.name)

        cb = CompartmentalSystemBuilder(odes)
        target_comp = Compartment.create(name="TARGET")
        complex_comp = Compartment.create(name="COMPLEX")
        target_amount = sympy.Function(target_comp.amount.name)(sympy.Symbol('t'))
        complex_amount = sympy.Function(complex_comp.amount.name)(sympy.Symbol('t'))
        cb.add_compartment(target_comp)
        cb.add_compartment(complex_comp)
        cb.add_flow(target_comp, complex_comp, kon * central_amount)
        cb.add_flow(complex_comp, target_comp, koff)
        cb.add_flow(target_comp, output, kout)
        cb.add_flow(complex_comp, output, kpe)
        cb.set_input(target_comp, kin)
        cb.set_input(central, koff * complex_amount - kon * central_amount * target_amount)
        cs = CompartmentalSystem(cb)
        model = model.replace(
            statements=model.statements.before_odes + cs + model.statements.after_odes
        )
    elif type == "IB":
        kon = sympy.Symbol('KON')
        model = add_individual_parameter(model, kon.name)
        kin = sympy.Symbol('KIN')
        model = add_individual_parameter(model, kin.name)
        kout = sympy.Symbol('KOUT')
        model = add_individual_parameter(model, kout.name)
        kpe = sympy.Symbol('KPE')
        model = add_individual_parameter(model, kpe.name)

        cb = CompartmentalSystemBuilder(odes)
        target_comp = Compartment.create(name="TARGET")
        complex_comp = Compartment.create(name="COMPLEX")
        target_amount = sympy.Function(target_comp.amount.name)(sympy.Symbol('t'))
        complex_amount = sympy.Function(complex_comp.amount.name)(sympy.Symbol('t'))
        cb.add_compartment(target_comp)
        cb.add_compartment(complex_comp)
        cb.add_flow(target_comp, complex_comp, kon * central_amount)
        cb.add_flow(target_comp, output, kout)
        cb.add_flow(complex_comp, output, kpe)
        cb.set_input(target_comp, kin)
        cb.set_input(central, -kon * central_amount * target_amount)
        cs = CompartmentalSystem(cb)
    elif type == "CR":
        kon = sympy.Symbol('KON')
        model = add_individual_parameter(model, kon.name)
        koff = sympy.Symbol('KOFF')
        model = add_individual_parameter(model, koff.name)
        kin = sympy.Symbol('KIN')
        model = add_individual_parameter(model, kin.name)
        kout = sympy.Symbol('KOUT')
        model = add_individual_parameter(model, kout.name)

        cb = CompartmentalSystemBuilder(odes)
        target_comp = Compartment.create(name="TARGET")
        complex_comp = Compartment.create(name="COMPLEX")
        target_amount = sympy.Function(target_comp.amount.name)(sympy.Symbol('t'))
        complex_amount = sympy.Function(complex_comp.amount.name)(sympy.Symbol('t'))
        cb.add_compartment(target_comp)
        cb.add_compartment(complex_comp)
        cb.add_flow(target_comp, complex_comp, kon * central_amount)
        cb.add_flow(complex_comp, target_comp, koff)
        cb.add_flow(target_comp, output, kout)
        cb.add_flow(complex_comp, output, kout)
        cb.set_input(target_comp, kin)
        cb.set_input(central, koff * complex_amount - kon * central_amount * target_amount)
        cs = CompartmentalSystem(cb)
    elif type == "CRIB":
        kon = sympy.Symbol('KON')
        model = add_individual_parameter(model, kon.name)
        kint = sympy.Symbol('KINT')
        model = add_individual_parameter(model, kint.name)
        rinit = sympy.Symbol('RINIT')
        model = add_individual_parameter(model, rinit.name)

        cb = CompartmentalSystemBuilder(odes)
        complex_comp = Compartment.create(name="COMPLEX")
        complex_amount = sympy.Function(complex_comp.amount.name)(sympy.Symbol('t'))
        cb.add_compartment(complex_comp)
        cb.add_flow(complex_comp, output, kint)
        cb.add_flow(central, complex_comp, kon * rinit)
        cb.add_flow(complex_comp, central, kon * central_amount)
        cs = CompartmentalSystem(cb)
    elif type == "QSS":
        r_0 = sympy.Symbol('R_0')
        model = add_individual_parameter(model, r_0.name)
        kdc = sympy.Symbol('KDC')
        model = add_individual_parameter(model, kdc.name)
        kint = sympy.Symbol('KINT')
        model = add_individual_parameter(model, kint.name)
        kdeg = sympy.Symbol('KDEG')
        model = add_individual_parameter(model, kdeg.name)

        ksyn = sympy.Symbol('KSYN')
        kd = sympy.Symbol('KD')
        central_comp = odes.central_compartment
        elimination_rate = odes.get_flow(central_comp, output)
        numer, denom = elimination_rate.as_numer_denom()
        if denom != 1:
            vc = denom
        else:
            vc = sympy.Symbol('VC')  # FIXME: What do do here?
        ksyn_ass = Assignment(ksyn, r_0 * kdeg)
        kd_ass = Assignment(kd, kdc * vc)

        cb = CompartmentalSystemBuilder(odes)
        target_comp = Compartment.create(name="TARGET")
        cb.add_compartment(target_comp)
        central_amount = sympy.Function(central_comp.amount.name)(sympy.Symbol('t'))
        target_amount = sympy.Function(target_comp.amount.name)(sympy.Symbol('t'))

        lcfree_symb = sympy.Symbol('LCFREE')
        lcfree_expr = sympy.Rational(1, 2) * (
            central_amount
            - target_amount
            - kd
            + sympy.sqrt((central_amount - target_amount - kd) ** 2 + 4 * kd * central_amount)
        )
        lcfree_ass = Assignment(lcfree_symb, lcfree_expr)

        # FIXME: Missing CL/VC (elimination rate) in central_comp input
        # FIXME: Support two and three compartment distribution
        cb.set_input(central_comp, -target_amount * kint * lcfree_symb / (kd + lcfree_symb))
        cb.set_input(
            target_comp,
            ksyn * vc
            - kdeg * target_amount
            - (kint - kdeg) * target_amount * lcfree_symb / (kd + lcfree_symb),
        )

        before = model.statements.before_odes + (ksyn_ass, kd_ass, lcfree_ass)
        cs = CompartmentalSystem(cb)
        model = model.replace(statements=before + cs + model.statements.after_odes)
    else:
        raise ValueError(f'Unknown TMDD type "{type}".')

    if type == "QSS":
        model = set_initial_condition(model, "TARGET", r_0 * vc)

    return model.update_source()
