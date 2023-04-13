"""
:meta private:
"""
from pharmpy.deps import sympy
from pharmpy.model import (
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Model,
    output,
)

from .odes import add_individual_parameter


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
        kd = sympy.Symbol('KD')
        model = add_individual_parameter(model, kd.name)
        kpe = sympy.Symbol('KPE')
        model = add_individual_parameter(model, kpe.name)

        cb = CompartmentalSystemBuilder(odes)
        target_comp = Compartment.create(name="TARGET")
        complex_comp = Compartment.create(name="COMPLEX")
        target_amount = sympy.Function(target_comp.amount.name)(sympy.Symbol('t'))
        complex_amount = sympy.Function(complex_comp.amount.name)(sympy.Symbol('t'))
        cb.add_flow(complex_comp, output, kpe)

        # lcfree = sympy.Rational(1, 2) * (
        #    central_amount
        #    - complex_amount
        #    - kd
        #    + sympy.sqrt((central_amount - complex_amount - kd) ** 2 + 4 * kd * central_amount)
        # )
        cs = CompartmentalSystem(cb)
    else:
        raise ValueError(f'Unknown TMDD type "{type}". Supported is full')

    statements = model.statements.before_odes + cs + model.statements.after_odes
    model = model.replace(statements=statements)
    return model.update_source()
