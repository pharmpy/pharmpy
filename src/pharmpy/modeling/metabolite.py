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

from .error import set_proportional_error_model
from .odes import _find_noncov_theta, add_individual_parameter
from .parameters import set_initial_estimates


def add_metabolite(model: Model, drug_dvid: int = 1):
    """Adds a metabolite compartment to a model

    The flow from the central compartment to the metabolite compartment
    will be unidirectional.

    Parameters
    ----------
    model : Model
        Pharmpy model
    drug_dvid : int
        DVID for drug (assuming all other DVIDs being for metabolites)

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
    clm = sympy.Symbol('CLM')
    model = add_individual_parameter(model, clm.name)
    vm = sympy.Symbol('VM')
    model = add_individual_parameter(model, vm.name)

    odes = model.statements.ode_system
    central = odes.central_compartment
    ke = odes.get_flow(central, output)
    cl, vc = ke.as_numer_denom()

    if vc != 1:
        pop_cl = _find_noncov_theta(model, cl)
        pop_vc = _find_noncov_theta(model, vc)
        pop_clm_init = model.parameters[pop_cl].init * 2.0
        pop_vm_init = model.parameters[pop_vc].init * 0.5
        model = set_initial_estimates(model, {'POP_CLM': pop_clm_init, 'POP_VM': pop_vm_init})

    cb = CompartmentalSystemBuilder(odes)
    metacomp = Compartment.create(name="METABOLITE")
    cb.add_compartment(metacomp)
    cb.add_flow(central, metacomp, ke)
    cb.add_flow(metacomp, output, clm / vm)
    cb.remove_flow(central, output)
    cs = CompartmentalSystem(cb)

    # dvid_col = model.datainfo.typeix['dvid'][0]
    # dvids = dvid_col.categories

    conc = Assignment(sympy.Symbol('CONC_M'), metacomp.amount / vm)
    y_m = sympy.Symbol('Y_M')
    y = Assignment(y_m, conc.symbol)
    original_y = next(iter(model.dependent_variables))
    ind = model.statements.after_odes.find_assignment_index(original_y)
    old_after = model.statements.after_odes
    new_after = old_after[: ind + 1] + y + old_after[ind + 1 :]
    error = conc + new_after

    dvs = model.dependent_variables.replace(y_m, 2)  # FIXME: Should be next DVID in categories
    statements = model.statements.before_odes + cs + error
    model = model.replace(statements=statements, dependent_variables=dvs)
    model = set_proportional_error_model(model, dv=2, zero_protection=False)
    return model
