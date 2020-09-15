import sympy

from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment, Bolus, CompartmentalSystem, ExplicitODESystem, Infusion


def explicit_odes(model):
    """Convert model from compartmental system to explicit ODE system
       or do nothing if it already has an explicit ODE system
    """
    statements = model.statements
    odes = statements.ode_system
    if isinstance(odes, CompartmentalSystem):
        eqs, ics = odes.to_explicit_odes()
        new = ExplicitODESystem(eqs, ics)
        statements[model.statements.index(odes)] = new
        model.statements = statements
    return model


def _have_zero_order_absorption(model):
    """Check if ode system describes a zero order absorption

       currently defined as having Infusion dose with rate not in dataset
    """
    odes = model.statements.ode_system
    dosing = odes.find_dosing()
    dose = dosing.dose
    if isinstance(dose, Infusion):
        if dose.rate is None:
            value = dose.duration
        else:
            value = dose.rate
        if isinstance(value, sympy.Symbol) or isinstance(value, str):
            name = str(value)
            if name not in model.dataset.columns:
                return True
    return False


def absorption(model, order, rate=None):
    """Set or change the absorption for a model

    Parameters
    ----------
    model
        Model to set or change absorption for
    order
        'bolus', 0 or 1
    """
    statements = model.statements
    odes = statements.ode_system
    if not isinstance(odes, CompartmentalSystem):
        raise ValueError("Setting absorption is not supported for ExplicitODESystem")

    depot = odes.find_depot()
    order = str(order)
    if order == 'bolus':
        if depot:
            to_comp, _ = odes.get_compartment_flows(depot)[0]
            to_comp.dose = depot.dose
            ka = odes.get_flow(depot, odes.find_central())
            odes.remove_compartment(depot)
            symbols = ka.free_symbols
            for s in symbols:
                statements.remove_symbol_definition(s, odes)
            model.statements = statements
            model.remove_unused_parameters_and_rvs()
        elif _have_zero_order_absorption(model):
            dose_comp = odes.find_dosing()
            old_symbols = dose_comp.free_symbols
            dose_comp.dose = Bolus(dose_comp.dose.amount)
            unneeded_symbols = old_symbols - dose_comp.dose.free_symbols
            for symb in unneeded_symbols:
                statements.remove_symbol_definition(symb, odes)
            model.remove_unused_parameters_and_rvs()
    elif order == '0':
        if not _have_zero_order_absorption(model):
            pass
    elif order == '1':
        if not depot:
            dose_comp = odes.find_dosing()
            depot = odes.add_compartment('DEPOT')
            depot.dose = dose_comp.dose
            dose_comp.dose = None
            mat_param = Parameter('TVMAT', init=0.1, lower=0)
            model.parameters.add(mat_param)
            imat = Assignment('MAT', mat_param.symbol)
            model.statements = model.statements.insert(0, imat)     # FIXME: Don't set again
            odes.add_flow(depot, dose_comp, 1 / mat_param.symbol)
            model.remove_unused_parameters_and_rvs()
    else:
        raise ValueError(f'Requested order {order} but only orders bolus, 0 and 1 are supported')

    return model
