from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment, CompartmentalSystem, ExplicitODESystem


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


def absorption(model, order, rate=None):
    """Set or change the absorption for a model

    Parameters
    ----------
    model
        Model to set or change absorption for
    order
        0 or 1
    """
    statements = model.statements
    odes = statements.ode_system
    if not isinstance(odes, CompartmentalSystem):
        raise ValueError("Setting absorption is not supported for ExplicitODESystem")

    depot = odes.find_depot()
    if order == 0:
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
    elif order == 1:
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
    else:
        raise ValueError(f'Requested order {order} but only orders r0 and 1 are supported')

    return model
