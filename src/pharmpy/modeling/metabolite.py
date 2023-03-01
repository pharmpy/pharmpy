"""
:meta private:
"""
from pharmpy.internals.deps import sympy
from pharmpy.model import (
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Model,
    output,
)

from .odes import add_individual_parameter


def add_metabolite(model: Model):
    """Adds a metabolite compartment to a model

    The flow from the central compartment to the metabolite compartment
    will be unidirectional.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_metabolite(model)

    """
    qm1 = sympy.Symbol('QM1')
    model = add_individual_parameter(model, qm1.name)
    clm1 = sympy.Symbol('CLM1')
    model = add_individual_parameter(model, clm1.name)
    vm1 = sympy.Symbol('VM1')
    model = add_individual_parameter(model, vm1.name)

    odes = model.statements.ode_system
    central = odes.central_compartment
    ke = odes.get_flow(central, output)
    cl, vc = ke.as_numer_denom()
    cb = CompartmentalSystemBuilder(odes)
    metacomp = Compartment.create(name="METABOLITE")
    cb.add_compartment(metacomp)
    cb.add_flow(central, metacomp, qm1 / vc)
    cb.add_flow(metacomp, output, clm1 / vm1)
    cs = CompartmentalSystem(cb)
    statements = model.statements.before_odes + cs + model.statements.after_odes
    model = model.replace(statements=statements)
    return model
